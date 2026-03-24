import os
import sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AdvTrain import *
from base_utils import get_underlying_model
from datasets import Dataset
from config import get_task_config, get_train_batch_size, get_num_proc, get_dataloader_num_workers
from paths_config import DATA_DIR, MERGED_DATA_DIR, T5_SMALL_DIR, get_checkpoint_task_dir
from perf_monitor import PerfMonitor
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, TensorDataset

_t5_path = T5_SMALL_DIR if os.path.exists(T5_SMALL_DIR) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)


def _is_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ


def _setup_distributed():
    if not _is_torchrun():
        return False, 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("Step 4 DDP 推理需要 CUDA（与 Step 3/5 一致，后端 NCCL）。")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def _teardown_distributed(was_initialized: bool):
    if was_initialized and dist.is_initialized():
        dist.destroy_process_group()


def _run_one_task(
    task_idx: int,
    batch_size: int,
    nproc: int,
    device_ids: list,
    distributed: bool,
    rank: int,
    world_size: int,
    local_rank: int,
    log_file: str,
):
    task_config = get_task_config(task_idx)
    auxiliary = task_config["auxiliary"]
    target = task_config["target"]
    save_file = os.path.join(get_checkpoint_task_dir(task_idx), "model.pth")

    if distributed:
        device = f"cuda:{local_rank}"
        if batch_size % world_size != 0:
            raise ValueError(
                f"DDP 推理要求全局 --batch-size ({batch_size}) 能被进程数 ({world_size}) 整除，"
                "与 Step 3/5 一致；单卡请 DDP_NPROC=1。"
            )
        local_batch = batch_size // world_size
        if rank == 0:
            print(
                f"[Step4 DDP] task={task_idx} global_batch={batch_size} per_rank_batch={local_batch} world_size={world_size}",
                flush=True,
            )
    else:
        device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() and device_ids else "cpu"
        local_batch = batch_size

    path = os.path.join(MERGED_DATA_DIR, str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    train_df["item"] = train_df["item"].astype(str)
    nuser = train_df["user_idx"].max() + 1
    nitem = train_df["item_idx"].max() + 1
    config = {
        "task_idx": task_idx,
        "device": device,
        "log_file": log_file,
        "save_file": save_file,
        "learning_rate": task_config["lr"],
        "epochs": 50,
        "batch_size": batch_size,
        "emsize": 768,
        "nlayers": 2,
        "nhid": 2048,
        "ntoken": 32128,
        "dropout": 0.2,
        "nuser": nuser,
        "nitem": nitem,
        "coef": task_config["coef"],
        "nhead": 2,
    }

    suser_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, auxiliary, "user_profiles.npy")), dtype=torch.float, device=device
    )
    sitem_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, auxiliary, "item_profiles.npy")), dtype=torch.float, device=device
    )
    sdomain_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, auxiliary, "domain.npy")), dtype=torch.float, device=device
    )
    tuser_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, target, "user_profiles.npy")), dtype=torch.float, device=device
    )
    titem_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, target, "item_profiles.npy")), dtype=torch.float, device=device
    )
    tdomain_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, target, "domain.npy")), dtype=torch.float, device=device
    )

    domain_profiles = torch.cat([sdomain_profiles.unsqueeze(0), tdomain_profiles.unsqueeze(0)], dim=0)
    user_profiles = torch.cat([tuser_profiles, suser_profiles], dim=0)
    item_profiles = torch.cat([titem_profiles, sitem_profiles], dim=0)
    model = Model(
        config.get("nuser"),
        config.get("nitem"),
        config.get("ntoken"),
        config.get("emsize"),
        config.get("nhead"),
        config.get("nhid"),
        config.get("nlayers"),
        config.get("dropout"),
        user_profiles,
        item_profiles,
        domain_profiles,
    ).to(device)
    _map = device if isinstance(device, str) else f"cuda:{device}"
    model.load_state_dict(torch.load(config.get("save_file"), map_location=_map, weights_only=True))

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    elif torch.cuda.is_available() and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    target_df = train_df[train_df["domain"] == "target"].copy()
    target_df["domain"] = "auxiliary"
    target_dataset = Dataset.from_pandas(target_df)
    processor = Processor(auxiliary, target)
    encoded_data = target_dataset.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
    encoded_data.set_format("torch")

    n_samples = len(encoded_data)
    row_idx = torch.arange(n_samples, dtype=torch.long)
    full_dataset = TensorDataset(
        row_idx,
        encoded_data["user_idx"],
        encoded_data["item_idx"],
        encoded_data["rating"],
        encoded_data["explanation_idx"],
        encoded_data["domain_idx"],
    )

    if distributed:
        chunk = (n_samples + world_size - 1) // world_size
        s = rank * chunk
        e = min(s + chunk, n_samples)
        test_dataset = Subset(full_dataset, list(range(s, e)))
    else:
        test_dataset = full_dataset

    test_nw = get_dataloader_num_workers("test")
    pin_mem = torch.cuda.is_available()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=local_batch,
        shuffle=False,
        num_workers=test_nw,
        pin_memory=pin_mem,
        persistent_workers=test_nw > 0,
        prefetch_factor=2 if test_nw > 0 else None,
    )

    _model = get_underlying_model(model)
    model.eval()
    local_prediction_exps = []
    local_reference_exps = []
    local_entropy_values = []

    dev_id = local_rank if distributed else (device_ids[0] if device_ids else 0)
    perf = None
    if rank == 0 or not distributed:
        perf = PerfMonitor(
            device=dev_id,
            log_file=log_file,
            num_proc=nproc,
            test_num_workers=test_nw,
        )
        perf.start()
        perf.epoch_start()

    with torch.no_grad():
        for batch in test_dataloader:
            _, user_idx, item_idx, rating, tgt_output, domain_idx = batch
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(
                (user_idx, item_idx, rating, tgt_output, domain_idx), device
            )
            _model.recommend(user_idx, item_idx, domain_idx)
            pred_exps, entropy = _model.generate(user_idx, item_idx, domain_idx)
            local_prediction_exps.extend(tokenizer.batch_decode(pred_exps, skip_special_tokens=True))
            local_reference_exps.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
            local_entropy_values.extend(entropy.cpu().numpy().tolist())

    if distributed:
        dist.barrier()
        payload = (local_prediction_exps, local_reference_exps, local_entropy_values)
        if rank == 0:
            gather_list = [None] * world_size
            dist.gather_object(payload, object_gather_list=gather_list, dst=0)
        else:
            dist.gather_object(payload, dst=0)

        if rank == 0:
            prediction_exps = []
            reference_exps = []
            entropy_values = []
            for part in gather_list:
                p, r, e = part
                prediction_exps.extend(p)
                reference_exps.extend(r)
                entropy_values.extend(e)
            assert len(entropy_values) == n_samples, (
                f"rank0 汇总条数 {len(entropy_values)} 与 target 样本数 {n_samples} 不一致"
            )
        else:
            prediction_exps = reference_exps = entropy_values = None
    else:
        prediction_exps = local_prediction_exps
        reference_exps = local_reference_exps
        entropy_values = local_entropy_values

    if perf is not None:
        perf.epoch_end(1, len(test_dataloader))
        perf.finish()

    if distributed:
        dist.barrier()

    if not distributed or rank == 0:
        filtered_indices = filter_by_entropy(entropy_values)
        filtered_prediction_exps = [prediction_exps[i] for i in filtered_indices]
        filtered_target_df = target_df.iloc[filtered_indices].copy()
        filtered_target_df["explanation"] = filtered_prediction_exps

        updated_train_df = train_df[train_df["domain"] == "target"].copy()
        final_df = pd.concat([updated_train_df, filtered_target_df])
        _tdir = get_checkpoint_task_dir(task_idx)
        os.makedirs(_tdir, exist_ok=True)
        final_df.to_csv(os.path.join(_tdir, "factuals_counterfactuals.csv"), index=False)
        print(f"Task {task_idx}: 已写入 {os.path.join(_tdir, 'factuals_counterfactuals.csv')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="非 torchrun 时：逗号分隔 GPU，多卡走 DataParallel；torchrun DDP 下忽略，以 LOCAL_RANK 为准",
    )
    parser.add_argument(
        "--task", type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8], metavar="N", help="仅跑指定任务 1-8"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="全局 batch（DDP 时须能被 WORLD_SIZE 整除）")
    parser.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="PerfMonitor 结构化日志路径；不传则沿用当前目录 save.log（与 run_step4.sh 传入的绝对路径对齐 Step 3/5）",
    )
    args = parser.parse_args()

    batch_size = args.batch_size if args.batch_size is not None else get_train_batch_size()
    nproc = args.num_proc if args.num_proc is not None else get_num_proc()
    device_ids = [int(x.strip()) for x in args.gpus.split(",")]

    distributed, rank, world_size, local_rank = _setup_distributed()
    if distributed and rank == 0:
        print(
            f"[Step4] torchrun DDP: WORLD_SIZE={world_size}，忽略 --gpus {args.gpus}，请用 CUDA_VISIBLE_DEVICES。",
            flush=True,
        )

    task_range = [args.task] if args.task else range(1, 9)
    seed = 3407
    torch.manual_seed(seed)

    try:
        for task_idx in task_range:
            lf = args.log_file if args.log_file is not None else "save.log"
            lf = os.path.abspath(os.path.expanduser(lf))
            _run_one_task(
                task_idx, batch_size, nproc, device_ids, distributed, rank, world_size, local_rank, log_file=lf
            )
    finally:
        _teardown_distributed(distributed)
