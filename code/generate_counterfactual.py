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

_t5_path = T5_SMALL_DIR if os.path.exists(T5_SMALL_DIR) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0", help="逗号分隔的 GPU ID，如 '0,1' 使用多卡 DataParallel")
    parser.add_argument("--task", type=int, default=None, choices=[1,2,3,4,5,6,7,8], metavar="N", help="仅跑指定任务 1-8，不传则跑全部")
    parser.add_argument("--batch-size", type=int, default=None, help="推理批次大小（不传则用 config.train_batch_size）")
    parser.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数，不传则用 config.num_proc")
    args = parser.parse_args()
    batch_size = args.batch_size if args.batch_size is not None else get_train_batch_size()
    nproc = args.num_proc if args.num_proc is not None else get_num_proc()
    device_ids = [int(x.strip()) for x in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() and device_ids else "cpu"
    task_range = [args.task] if args.task else range(1, 9)

    seed = 3407
    torch.manual_seed(seed)
    for task_idx in task_range:
        task_config = get_task_config(task_idx)
        auxiliary = task_config["auxiliary"]
        target = task_config["target"]
        log_file = "save.log"
        save_file = os.path.join(get_checkpoint_task_dir(task_idx), "model.pth")
        epochs = 50
        coef = task_config["coef"]
        learning_rate = task_config["lr"]
        adv = task_config["adv"]
        path = os.path.join(MERGED_DATA_DIR, str(task_idx))
        train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
        train_df['item'] = train_df['item'].astype(str)
        nuser = train_df['user_idx'].max() + 1
        nitem = train_df['item_idx'].max() + 1
        config = {
            "task_idx": task_idx,
            "device": device,
            "log_file": log_file,
            "save_file":save_file,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "emsize": 768,
            "nlayers": 2,
            "nhid": 2048,
            "ntoken": 32128,
            "dropout": 0.2,
            "nuser": nuser,
            "nitem": nitem, 
            "coef": coef, 
            "nhead": 2
            }
        # load domain, user, item profiles from auxiliary
        suser_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, auxiliary, "user_profiles.npy")), dtype=torch.float, device=config.get("device"))
        sitem_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, auxiliary, "item_profiles.npy")), dtype=torch.float, device=config.get("device"))
        sdomain_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, auxiliary, "domain.npy")), dtype=torch.float, device=config.get("device"))

        # load target
        tuser_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, target, "user_profiles.npy")), dtype=torch.float, device=config.get("device"))
        titem_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, target, "item_profiles.npy")), dtype=torch.float, device=config.get("device"))
        tdomain_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, target, "domain.npy")), dtype=torch.float, device=config.get("device"))

        domain_profiles = torch.cat([sdomain_profiles.unsqueeze(0), tdomain_profiles.unsqueeze(0)], dim=0)
        user_profiles = torch.cat([tuser_profiles, suser_profiles], dim=0)
        item_profiles = torch.cat([titem_profiles, sitem_profiles], dim=0)
        model = Model(config.get("nuser"), config.get("nitem"), config.get("ntoken"), config.get("emsize"), config.get("nhead"), config.get("nhid"), config.get("nlayers"), config.get("dropout"), user_profiles, item_profiles, domain_profiles).to(device)
        _map = device if isinstance(device, str) else f"cuda:{device}"
        model.load_state_dict(torch.load(config.get("save_file"), map_location=_map, weights_only=True))
        if torch.cuda.is_available() and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)

        target_df = train_df[train_df['domain'] == 'target'].copy()
        target_df["domain"] = "auxiliary"
        target_dataset = Dataset.from_pandas(target_df)
        processor = Processor(auxiliary, target)
        encoded_data = target_dataset.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
        encoded_data.set_format("torch")

        test_dataset = TensorDataset(
            encoded_data['user_idx'],
            encoded_data['item_idx'],
            encoded_data['rating'],
            encoded_data['explanation_idx'],
            encoded_data['domain_idx']
        )
        test_nw = get_dataloader_num_workers("test")
        pin_mem = torch.cuda.is_available()
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=test_nw,
            pin_memory=pin_mem,
            persistent_workers=test_nw > 0,
            prefetch_factor=2 if test_nw > 0 else None,
        )
        _model = get_underlying_model(model)
        model.eval()
        prediction_exps = []
        reference_exps = []
        entropy_values = []

        dev_id = device_ids[0] if device_ids else 0
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
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
                pred_ratings = _model.recommend(user_idx, item_idx, domain_idx)
                pred_exps, entropy = _model.generate(user_idx, item_idx, domain_idx)
                prediction_exps.extend(tokenizer.batch_decode(pred_exps, skip_special_tokens=True))
                reference_exps.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
                entropy_values.extend(entropy.cpu().numpy())
        perf.epoch_end(1, len(test_dataloader))
        perf.finish()

        filtered_indices = filter_by_entropy(entropy_values)
        filtered_prediction_exps = [prediction_exps[i] for i in filtered_indices]
        filtered_target_df = target_df.iloc[filtered_indices].copy()
        filtered_target_df['explanation'] = filtered_prediction_exps

        updated_train_df = train_df[train_df['domain'] == 'target'].copy()
        final_df = pd.concat([updated_train_df, filtered_target_df])
        _tdir = get_checkpoint_task_dir(task_idx)
        os.makedirs(_tdir, exist_ok=True)
        final_df.to_csv(os.path.join(_tdir, "factuals_counterfactuals.csv"), index=False)