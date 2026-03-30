"""
Step4 执行体核心（ENGINE）：反事实生成主逻辑。

由 ``executors.step4_entry`` 在 torchrun 下调用（code/ 下历史薄壳名保持不变）。
用户入口请使用 ``python code/d4c.py step4 …``。
"""
import hashlib
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
from executors.adv_train_core import *
from base_utils import get_underlying_model
from config import (
    BASE_TRAINING_DEFAULTS,
    TASK_DEFAULTS,
    get_dataloader_num_workers,
    get_dataloader_prefetch_factor,
    get_num_proc,
)
from datasets import Dataset, load_from_disk
from paths_config import (
    DATA_DIR,
    MERGED_DATA_DIR,
    T5_SMALL_DIR,
    append_log_dual,
    get_checkpoint_task_dir,
    get_d4c_root,
)
from perf_monitor import PerfMonitor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, TensorDataset

_STEP4_REQUIRES_TORCHRUN_MSG = (
    "step4 runner 仅支持 torchrun / python -m torch.distributed.run DDP。\n"
    "用户日常（仓库根）: python code/d4c.py step4 …\n"
    "请勿在非 torchrun 环境下直接启动薄壳脚本。\n"
    "高级排障（须 torchrun，在 code/ 目录）见 docs/D4C_Scripts_and_Runtime_Guide.md 附录。\n"
    "多卡请设置 CUDA_VISIBLE_DEVICES 并使 nproc_per_node 与可见 GPU 数一致。"
)

# 编码逻辑 / Processor 口径变更时递增，避免误读旧 Arrow 缓存
_STEP4_ENCODE_CACHE_VERSION = "v1"

_t5_path = T5_SMALL_DIR if os.path.exists(T5_SMALL_DIR) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)


def _require_torchrun_env_vars() -> None:
    for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
        if k not in os.environ:
            raise RuntimeError(_STEP4_REQUIRES_TORCHRUN_MSG)


def _setup_distributed():
    _require_torchrun_env_vars()
    if not torch.cuda.is_available():
        raise RuntimeError("Step 4 DDP 推理需要 CUDA（与 Step 3/5 一致，后端 NCCL）。")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _teardown_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _step4_encoded_cache_dir(
    task_idx: int,
    aug_csv_path: str,
    auxiliary: str,
    target: str,
    t5_resolved: str,
    processor_max_length: int,
) -> str:
    """稳定缓存目录：任务 + 数据文件身份 + 域配置 + tokenizer 路径 + Processor 长度。"""
    st = os.stat(aug_csv_path)
    payload = (
        f"{_STEP4_ENCODE_CACHE_VERSION}|task={task_idx}|aux={auxiliary}|tgt={target}|"
        f"t5={os.path.abspath(t5_resolved)}|csv={os.path.abspath(aug_csv_path)}|"
        f"sz={st.st_size}|mt={int(st.st_mtime_ns)}|maxlen={processor_max_length}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return os.path.join(get_d4c_root(), "cache", "step4_encoded", str(task_idx), digest)


def _dataset_saved_to_disk(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def _step4_pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _step4_partial_format_choice() -> str:
    """parquet（需 pyarrow）/ csv / auto（有 pyarrow 则 parquet）。"""
    v = os.environ.get("D4C_STEP4_PARTIAL_FORMAT", "auto").strip().lower()
    if v in ("parquet", "csv", "auto"):
        return v
    return "auto"


def _step4_partial_suffix_and_kind(fmt_choice: str) -> tuple[str, str]:
    if fmt_choice == "csv":
        return ".csv", "csv"
    if fmt_choice == "parquet":
        if _step4_pyarrow_available():
            return ".parquet", "parquet"
        return ".csv", "csv"
    if _step4_pyarrow_available():
        return ".parquet", "parquet"
    return ".csv", "csv"


def _step4_write_partial_df(df: pd.DataFrame, path: str, kind: str) -> None:
    if kind == "parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def _step4_read_partial_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path, encoding="utf-8")


class Step4PerfLogger:
    """各 rank 独立文件，最小侵入性能埋点（与 PerfMonitor 并存）。"""

    def __init__(self, log_file: str, task_idx: int, rank: int):
        log_dir = os.path.dirname(os.path.abspath(os.path.expanduser(log_file)))
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, f"step4_perf_task{task_idx}_rank{rank}.log")
        self.rank = rank

    def line(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{ts}] [rank{self.rank}] {msg}\n"
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
        print(text, end="", flush=True)


def _format_seconds_human(seconds: float) -> str:
    """人类可读时长：秒或分钟。"""
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60.0:.2f} min"


def _step4_pct(part: float, whole: float, ndigits: int = 1) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, ndigits)


def _log_step4_final_summary(
    *,
    task_idx: int,
    world_size: int,
    n_rows: int,
    log_file: str,
    plog: Step4PerfLogger,
    step4_end_to_end_wall_s: float,
    preprocess_wall_s: float,
    inference_loop_wall_s: float,
    decode_local_wall_s: float,
    merge_wall_s: float,
    filter_wall_s: float,
    csv_write_wall_s: float,
    barrier_after_inference_wall_s: float,
    collective_gather_paths_wall_s: float,
    trainer_epoch_time_s: float | None,
    inference_only_avg_step_ms: float,
) -> None:
    """rank0：train.log / stdout / step4_perf 三处写入 Step4 端到端摘要（不改变训练器原有 Epoch 输出）。"""
    e2e = step4_end_to_end_wall_s
    decode_tail_s = decode_local_wall_s
    infer_pct = _step4_pct(inference_loop_wall_s, e2e)
    decode_pct = _step4_pct(decode_tail_s, e2e)
    csv_pct = _step4_pct(csv_write_wall_s, e2e)
    trainer_s_str = f"{float(trainer_epoch_time_s):.4f}" if trainer_epoch_time_s is not None else "n/a"

    mach = (
        "[rank0] step4_final_summary\n"
        f"  step4_end_to_end_wall_s={e2e:.4f}\n"
        f"  total_wall_s__alias_of_step4_e2e={e2e:.4f}\n"
        f"  end_to_end_wall_s__alias_of_step4_e2e={e2e:.4f}\n"
        f"  epoch_time_scope=inference_only\n"
        f"  step4_total_scope=full_task\n"
        f"  preprocess_wall_s={preprocess_wall_s:.4f}\n"
        f"  inference_loop_wall_s={inference_loop_wall_s:.4f}\n"
        f"  decode_tail_wall_s={decode_tail_s:.4f}\n"
        f"  merge_wall_s={merge_wall_s:.4f}\n"
        f"  filter_wall_s={filter_wall_s:.4f}\n"
        f"  csv_write_wall_s={csv_write_wall_s:.4f}\n"
        f"  barrier_after_inference_wall_s={barrier_after_inference_wall_s:.4f}\n"
        f"  collective_gather_paths_wall_s={collective_gather_paths_wall_s:.4f}\n"
        f"  inference_share_pct={infer_pct}\n"
        f"  decode_share_pct={decode_pct}\n"
        f"  csv_share_pct={csv_pct}\n"
        f"  n_rows={n_rows}\n"
        f"  world_size={world_size}\n"
        f"  inference_only_avg_step_ms={inference_only_avg_step_ms:.4f}\n"
        f"  trainer_epoch_time_s={trainer_s_str}\n"
        f"  note=trainer_epoch_time_excludes_decode_merge_csv_tail"
    )
    for ln in mach.splitlines():
        msg = ln.removeprefix("[rank0] ") if ln.startswith("[rank0] ") else ln
        plog.line(msg)
    append_log_dual(log_file, mach + "\n")

    trainer_h = (
        _format_seconds_human(float(trainer_epoch_time_s))
        if trainer_epoch_time_s is not None
        else "n/a"
    )
    human_lines = [
        f"Trainer epoch time shown above: {trainer_h} "
        f"(trainer_epoch_time_s={trainer_s_str} s; scope=inference_only / PerfMonitor epoch wall).",
        f"Actual step4 end-to-end time: {_format_seconds_human(e2e)} "
        f"(step4_end_to_end_wall_s={e2e:.4f} s; scope=full_task incl. preprocess, decode, merge, csv).",
    ]
    for _ln in human_lines:
        plog.line(_ln)
    append_log_dual(log_file, "\n".join(f"[rank0] {_ln}" for _ln in human_lines) + "\n")

    block = (
        "\n========== Step 4 End-to-End Summary ==========\n"
        f"Task: {task_idx}\n"
        f"Rows: {n_rows}\n"
        f"World size: {world_size}\n"
        f"epoch_time_scope=inference_only | step4_total_scope=full_task\n"
        f"Trainer epoch time shown above: {trainer_h} (trainer_epoch_time_s={trainer_s_str} s)\n"
        f"Actual step4 end-to-end time: {_format_seconds_human(e2e)} (step4_end_to_end_wall_s={e2e:.2f} s)\n"
        f"Preprocess: {preprocess_wall_s:.2f} s ({_format_seconds_human(preprocess_wall_s)})\n"
        f"Inference loop: {inference_loop_wall_s:.2f} s ({_format_seconds_human(inference_loop_wall_s)})\n"
        f"trainer_epoch_time_s (PerfMonitor, inference-only): {trainer_s_str} s ({trainer_h})\n"
        f"Barrier after inference: {barrier_after_inference_wall_s:.2f} s\n"
        f"Decode tail (local tokenizer): {decode_tail_s:.2f} s ({_format_seconds_human(decode_tail_s)})\n"
        f"Merge (rank0 read partials + sort/validate): {merge_wall_s:.2f} s\n"
        f"Entropy filter: {filter_wall_s:.2f} s\n"
        f"Collective gather paths: {collective_gather_paths_wall_s:.4f} s\n"
        f"CSV write: {csv_write_wall_s:.2f} s ({_format_seconds_human(csv_write_wall_s)})\n"
        f"step4_end_to_end_wall_s (primary total): {e2e:.2f} s ({_format_seconds_human(e2e)})\n"
        f"Inference / decode / CSV share of step4 e2e: {infer_pct}% / {decode_pct}% / {csv_pct}%\n"
        "Note: trainer_epoch_time_s matches log line 'Epoch 1 | time: Xm' scope (inference loop only).\n"
        "Warning: the 'Epoch 1 time: Xm' line is not the full step4 task duration.\n"
        "==============================================\n"
    )
    print(block, flush=True)
    append_log_dual(log_file, block)


def _decode_pred_token_rows(token_rows, chunk_size: int = 4096, progress_plog: Step4PerfLogger | None = None):
    """各 rank 本地 batch_decode：先一次性打成 (n, max_len) int64 再用 torch.from_numpy 按 chunk 解码。"""
    if not token_rows:
        return []
    cs = int(os.environ.get("D4C_STEP4_DECODE_CHUNK", str(chunk_size)))
    if cs < 1:
        cs = chunk_size
    n = len(token_rows)
    _pid = tokenizer.pad_token_id
    pad_id = int(_pid) if _pid is not None else 0

    t_pack0 = time.perf_counter()
    max_len = 0
    for r in token_rows:
        lr = len(r)
        if lr > max_len:
            max_len = lr
    if max_len == 0:
        return [""] * n
    mat = np.full((n, max_len), pad_id, dtype=np.int64)
    for i, r in enumerate(token_rows):
        if not r:
            continue
        L = len(r)
        mat[i, :L] = r
    pack_wall_s = time.perf_counter() - t_pack0
    if progress_plog is not None:
        progress_plog.line(
            f"decode_token_pack_wall_s={pack_wall_s:.4f} decode_input_rows={n} max_seq_len={max_len}"
        )

    num_chunks = (n + cs - 1) // cs if n else 0
    log_every = int(os.environ.get("D4C_STEP4_DECODE_CHUNK_LOG_EVERY", "0"))
    out = [""] * n
    t_prog0 = time.perf_counter()
    for chunk_i, s in enumerate(range(0, n, cs)):
        e = min(s + cs, n)
        block = np.ascontiguousarray(mat[s:e])
        t = torch.from_numpy(block)
        decoded = tokenizer.batch_decode(t, skip_special_tokens=True)
        out[s:e] = decoded
        if progress_plog is not None and log_every > 0:
            done = chunk_i + 1
            if done % log_every == 0 or done == num_chunks:
                progress_plog.line(
                    f"decode_chunk_progress chunks={done}/{num_chunks} "
                    f"cum_wall_s={time.perf_counter() - t_prog0:.4f}"
                )
    return out


def _run_one_task(
    task_idx: int,
    batch_size: int,
    nproc: int,
    rank: int,
    world_size: int,
    local_rank: int,
    log_file: str,
):
    step4_e2e_start = time.perf_counter()
    task_config = TASK_DEFAULTS[task_idx]
    auxiliary = task_config["auxiliary"]
    target = task_config["target"]
    _task_ckpt_dir = get_checkpoint_task_dir(task_idx)
    # gather 之后不再用 dist.barrier 等 rank0 decode/CSV；用文件握手让 rank!=0 在下一轮任务/销毁进程组前等待 rank0 写完 CSV（纯 CPU，避免 NCCL 长时间 barrier 超时）。
    sync_ready_path = os.path.join(_task_ckpt_dir, f".step4_ddp_task_{task_idx}_ready")
    if rank == 0 and world_size > 1:
        try:
            os.remove(sync_ready_path)
        except OSError:
            pass

    save_file = os.path.join(_task_ckpt_dir, "model.pth")

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

    plog = Step4PerfLogger(log_file, task_idx, rank)

    path = os.path.join(MERGED_DATA_DIR, str(task_idx))
    aug_csv = os.path.join(path, "aug_train.csv")
    train_df = pd.read_csv(aug_csv)
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
        "ntoken": len(tokenizer),
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

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    target_df = train_df[train_df["domain"] == "target"].copy()
    target_df["domain"] = "auxiliary"
    target_dataset = Dataset.from_pandas(target_df)
    processor = Processor(auxiliary, target)
    proc_max_len = int(processor.max_length)

    cache_dir = _step4_encoded_cache_dir(task_idx, aug_csv, auxiliary, target, _t5_path, proc_max_len)
    plog.line(f"step4_encode_cache_dir={cache_dir}")

    perf = None
    if rank == 0:
        perf = PerfMonitor(
            device=local_rank,
            log_file=log_file,
            num_proc=nproc,
            test_num_workers=get_dataloader_num_workers("test"),
        )
        perf.start()

    # ---------- 阶段 preprocess：仅 rank0 tokenize + save；其余 rank barrier 后 load ----------
    t_pre0 = time.perf_counter()
    plog.line("tokenize_phase_start")
    encoded_data = None
    cache_hit = False  # 仅 rank0 语义；rank!=0 见 preprocess 日志 replica
    tokenize_wall = 0.0

    if rank == 0:
        t_tok0 = time.perf_counter()
        if _dataset_saved_to_disk(cache_dir):
            try:
                encoded_data = load_from_disk(cache_dir)
                if len(encoded_data) == len(target_df):
                    cache_hit = True
                else:
                    encoded_data = None
            except Exception:
                encoded_data = None
        if encoded_data is None:
            cache_hit = False
            encoded_data = target_dataset.map(
                lambda sample: processor(sample), num_proc=nproc, desc="Tokenize"
            )
            os.makedirs(cache_dir, exist_ok=True)
            encoded_data.save_to_disk(cache_dir)
        tokenize_wall = time.perf_counter() - t_tok0
        plog.line(
            f"tokenize_end cache_hit={cache_hit} tokenize_wall_s={tokenize_wall:.4f} n_rows={len(encoded_data)}"
        )
    else:
        plog.line("tokenize_skip_waiting_barrier (rank!=0)")

    t_barrier0 = time.perf_counter()
    dist.barrier()
    barrier_preprocess = time.perf_counter() - t_barrier0
    plog.line(f"barrier_after_tokenize wall_s={barrier_preprocess:.4f}")

    if rank != 0:
        t_ld = time.perf_counter()
        encoded_data = load_from_disk(cache_dir)
        tokenize_wall = time.perf_counter() - t_ld
        plog.line(f"load_from_disk wall_s={tokenize_wall:.4f} n_rows={len(encoded_data)}")

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

    chunk = (n_samples + world_size - 1) // world_size
    s = rank * chunk
    e = min(s + chunk, n_samples)
    test_dataset = Subset(full_dataset, list(range(s, e)))

    t_dl0 = time.perf_counter()
    test_nw = get_dataloader_num_workers("test")
    pin_mem = torch.cuda.is_available()
    _pf_test = get_dataloader_prefetch_factor(test_nw, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=local_batch,
        shuffle=False,
        num_workers=test_nw,
        pin_memory=pin_mem,
        persistent_workers=test_nw > 0,
        prefetch_factor=_pf_test,
    )
    dataloader_build_wall = time.perf_counter() - t_dl0

    preprocess_wall = time.perf_counter() - t_pre0
    _cache_log = f"cache_hit={cache_hit}" if rank == 0 else "replica_load_from_disk=True"
    plog.line(
        f"preprocess_summary preprocess_wall_s={preprocess_wall:.4f} "
        f"dataloader_build_wall_s={dataloader_build_wall:.4f} "
        f"tokenize_or_load_wall_s_rank={tokenize_wall:.4f} {_cache_log}"
    )

    _model = get_underlying_model(model)
    model.eval()
    local_pred_token_rows = []
    local_entropy_values = []
    local_row_indices = []

    if rank == 0:
        perf.test_num_workers = test_nw
        perf.epoch_start()

    log_interval = max(1, int(os.environ.get("D4C_STEP4_PERF_LOG_INTERVAL", "10")))
    t_prev_end = time.perf_counter()
    first_batch_wait = None
    step_idx = 0
    t_loop_start = time.perf_counter()

    with torch.no_grad():
        for batch in test_dataloader:
            t_batch_start = time.perf_counter()
            if first_batch_wait is None:
                first_batch_wait = t_batch_start - t_prev_end
                plog.line(f"first_batch_wait_s={first_batch_wait:.4f}")

            data_wait_s = t_batch_start - t_prev_end

            batch_row_idx, user_idx, item_idx, rating, tgt_output, domain_idx = batch
            t_gather0 = time.perf_counter()
            user_idx, item_idx, rating, _tgt_input, tgt_output, domain_idx = _model.gather(
                (user_idx, item_idx, rating, tgt_output, domain_idx), device
            )
            gather_h2d_s = time.perf_counter() - t_gather0

            # Step4 生成路径不依赖 recommend() 的 rating；generate() 内部自包含 encoder。
            # 去掉 recommend() 可省一次完整 transformer 前向（与 generate 第一步结构重复）。
            t_gen0 = time.perf_counter()
            pred_exps, entropy = _model.generate(user_idx, item_idx, domain_idx)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generate_wall_s = time.perf_counter() - t_gen0

            t_ent0 = time.perf_counter()
            ent_cpu = entropy.detach().cpu()
            ent_list = ent_cpu.numpy().tolist()
            entropy_sync_wall_s = time.perf_counter() - t_ent0

            decode_wall_s = 0.0
            local_row_indices.extend(batch_row_idx.detach().cpu().reshape(-1).tolist())
            local_pred_token_rows.extend(pred_exps.detach().cpu().tolist())
            local_entropy_values.extend(ent_list)

            t_end = time.perf_counter()
            step_wall_s = t_end - t_batch_start
            bsz = int(user_idx.size(0))
            samples_per_sec = bsz / step_wall_s if step_wall_s > 0 else 0.0
            peak_mb = (
                torch.cuda.max_memory_allocated(local_rank) / (1024**2) if torch.cuda.is_available() else 0.0
            )

            step_idx += 1
            if step_idx % log_interval == 0 or step_idx == 1:
                plog.line(
                    f"step={step_idx} step_wall_s={step_wall_s:.4f} data_wait_s={data_wait_s:.4f} "
                    f"gather_h2d_s={gather_h2d_s:.4f} generate_wall_s={generate_wall_s:.4f} "
                    f"decode_wall_s={decode_wall_s:.4f} entropy_sync_wall_s={entropy_sync_wall_s:.4f} "
                    f"samples_per_sec={samples_per_sec:.2f} max_mem_alloc_MB={peak_mb:.1f}"
                )

            t_prev_end = t_end

    inference_loop_wall_s = time.perf_counter() - t_loop_start
    plog.line(f"inference_loop_wall_s={inference_loop_wall_s:.4f} steps={step_idx}")
    if first_batch_wait is None:
        plog.line("first_batch_wait_s=n/a (empty_dataloader)")

    t_barrier1 = time.perf_counter()
    dist.barrier()
    barrier_after_loop_s = time.perf_counter() - t_barrier1
    plog.line(f"barrier_after_inference wall_s={barrier_after_loop_s:.4f}")

    if len(local_row_indices) != len(local_pred_token_rows) or len(local_entropy_values) != len(
        local_pred_token_rows
    ):
        raise RuntimeError(
            f"rank{rank} 本地 row_idx / token / entropy 条数不一致: "
            f"{len(local_row_indices)} {len(local_pred_token_rows)} {len(local_entropy_values)}"
        )

    # PerfMonitor 仅统计推理循环，避免尾部 decode/merge 拉长“单步均值”
    trainer_epoch_time_s: float | None = None
    if rank == 0:
        _rec = perf.epoch_end(1, len(test_dataloader), emit_log=True)
        trainer_epoch_time_s = float(_rec["epoch_time"])
        t_pf0 = time.perf_counter()
        perf.finish()
        rank0_perf_finish_wall_s = time.perf_counter() - t_pf0
        plog.line(f"rank0_perf_finish_wall_s={rank0_perf_finish_wall_s:.4f}")
        _epoch_scope_note = (
            "Note: Epoch time above is trainer/PerfMonitor wall time and excludes "
            "step4 decode/merge/csv tail."
        )
        print(_epoch_scope_note, flush=True)
        append_log_dual(log_file, _epoch_scope_note + "\n")

    partial_dir = os.path.join(_task_ckpt_dir, ".step4_partials")
    os.makedirs(partial_dir, exist_ok=True)
    partial_base = os.path.join(
        partial_dir, f"step4_partial_task{task_idx}_rank{rank}_pid{os.getpid()}"
    )
    fmt_choice = _step4_partial_format_choice()
    partial_suffix, partial_kind = _step4_partial_suffix_and_kind(fmt_choice)
    if fmt_choice == "parquet" and partial_kind == "csv":
        plog.line("step4_partial_fallback_csv reason=pyarrow_missing")
    partial_path = partial_base + partial_suffix

    _prev_torch_threads = torch.get_num_threads()
    try:
        _dt = os.environ.get("D4C_STEP4_DECODE_THREADS", "0").strip()
        if _dt:
            _nti = int(_dt)
            if _nti > 0:
                torch.set_num_threads(_nti)
                plog.line(f"decode_torch_num_threads={_nti}")
        t_dec_local0 = time.perf_counter()
        local_explanations = _decode_pred_token_rows(local_pred_token_rows, progress_plog=plog)
        decode_local_wall_s = time.perf_counter() - t_dec_local0
    finally:
        torch.set_num_threads(_prev_torch_threads)

    n_loc = len(local_explanations)
    _dchunk = int(os.environ.get("D4C_STEP4_DECODE_CHUNK", "4096"))
    _nchunks = (n_loc + _dchunk - 1) // _dchunk if n_loc else 0
    plog.line(
        f"decode_local_wall_s={decode_local_wall_s:.4f} decode_input_rows={n_loc} "
        f"decode_chunk_size={_dchunk} decode_num_chunks={_nchunks}"
    )

    t_before_partial_phase = time.perf_counter()
    t_prep0 = time.perf_counter()
    part_df = pd.DataFrame(
        {"row_idx": local_row_indices, "entropy": local_entropy_values, "explanation": local_explanations}
    )
    partial_prep_wall_s = time.perf_counter() - t_prep0
    plog.line(f"partial_df_prep_wall_s={partial_prep_wall_s:.4f}")

    t_pw0 = time.perf_counter()
    _step4_write_partial_df(part_df, partial_path, partial_kind)
    partial_write_wall_s = time.perf_counter() - t_pw0
    plog.line(
        f"partial_write_wall_s={partial_write_wall_s:.4f} partial_kind={partial_kind} path={partial_path}"
    )

    t_gather_paths0 = time.perf_counter()
    pre_gather_wait_wall_s = t_gather_paths0 - t_before_partial_phase
    if world_size == 1:
        path_gather = [(partial_path,)]
    else:
        path_payload = (partial_path,)
        if rank == 0:
            path_gather = [None] * world_size
            dist.gather_object(path_payload, object_gather_list=path_gather, dst=0)
        else:
            dist.gather_object(path_payload, dst=0)
    collective_gather_paths_wall_s = time.perf_counter() - t_gather_paths0
    plog.line(f"pre_gather_wait_wall_s={pre_gather_wait_wall_s:.4f}")
    plog.line(f"collective_gather_paths_wall_s={collective_gather_paths_wall_s:.4f}")
    plog.line("barrier_before_csv_removed=True")

    decode_merge_rank0_wall_s = 0.0
    rank0_read_partials_wall_s = 0.0
    rank0_sort_validate_wall_s = 0.0
    rank0_filter_wall_s = 0.0
    csv_write_wall_s = 0.0

    if rank == 0:
        t_read0 = time.perf_counter()
        dfs = []
        for item in path_gather:
            pth = item[0] if isinstance(item, (list, tuple)) else item
            dfs.append(_step4_read_partial_df(pth))
        rank0_read_partials_wall_s = time.perf_counter() - t_read0
        plog.line(f"rank0_read_partials_wall_s={rank0_read_partials_wall_s:.4f} n_files={len(dfs)}")

        t_sv0 = time.perf_counter()
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        if len(merged) != n_samples:
            raise RuntimeError(
                f"rank0 合并后行数 {len(merged)} 与 target 样本数 {n_samples} 不一致"
            )
        idx_arr = merged["row_idx"].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(idx_arr, np.arange(n_samples, dtype=np.int64)):
            raise RuntimeError("rank0 合并后 row_idx 未覆盖 0..n-1 或存在重复/缺失")
        entropy_values = merged["entropy"].astype(float).tolist()
        prediction_exps = merged["explanation"].astype(str).tolist()
        rank0_sort_validate_wall_s = time.perf_counter() - t_sv0
        plog.line(f"rank0_sort_validate_wall_s={rank0_sort_validate_wall_s:.4f} n={len(merged)}")

        decode_merge_rank0_wall_s = rank0_read_partials_wall_s + rank0_sort_validate_wall_s
        plog.line(f"decode_merge_rank0_wall_s={decode_merge_rank0_wall_s:.4f}")

        t_filt0 = time.perf_counter()
        filtered_indices = filter_by_entropy(entropy_values)
        rank0_filter_wall_s = time.perf_counter() - t_filt0
        plog.line(f"rank0_filter_wall_s={rank0_filter_wall_s:.4f}")

        t_csv0 = time.perf_counter()
        filtered_prediction_exps = [prediction_exps[i] for i in filtered_indices]
        filtered_target_df = target_df.iloc[filtered_indices].copy()
        filtered_target_df["explanation"] = filtered_prediction_exps

        updated_train_df = train_df[train_df["domain"] == "target"].copy()
        final_df = pd.concat([updated_train_df, filtered_target_df])
        os.makedirs(_task_ckpt_dir, exist_ok=True)
        final_df.to_csv(os.path.join(_task_ckpt_dir, "factuals_counterfactuals.csv"), index=False)
        csv_write_wall_s = time.perf_counter() - t_csv0
        print(
            f"Task {task_idx}: 已写入 {os.path.join(_task_ckpt_dir, 'factuals_counterfactuals.csv')}",
            flush=True,
        )
        plog.line(f"csv_write_wall_s={csv_write_wall_s:.4f}")

        for item in path_gather:
            pth = item[0] if isinstance(item, (list, tuple)) else item
            try:
                os.remove(pth)
            except OSError:
                pass

        if world_size > 1:
            with open(sync_ready_path, "w", encoding="utf-8") as sf:
                sf.write("1\n")
                sf.flush()
                os.fsync(sf.fileno())
    else:
        if world_size > 1:
            plog.line(
                "post_gather_no_more_collectives_exit_path=True "
                "post_csv_ready_wait_reason=rank0_merge_filter_csv_before_next_collective"
            )
            t_wait0 = time.perf_counter()
            while not os.path.exists(sync_ready_path):
                time.sleep(0.05)
            plog.line(f"post_csv_ready_wait_wall_s={time.perf_counter() - t_wait0:.4f}")

    decode_tail_wall_s = decode_local_wall_s + (
        (decode_merge_rank0_wall_s + rank0_filter_wall_s) if rank == 0 else 0.0
    )
    inf_avg_ms = (inference_loop_wall_s / step_idx * 1000.0) if step_idx else 0.0
    step4_end_to_end_wall_s = time.perf_counter() - step4_e2e_start

    if rank == 0:
        _log_step4_final_summary(
            task_idx=task_idx,
            world_size=world_size,
            n_rows=n_samples,
            log_file=log_file,
            plog=plog,
            step4_end_to_end_wall_s=step4_end_to_end_wall_s,
            preprocess_wall_s=preprocess_wall,
            inference_loop_wall_s=inference_loop_wall_s,
            decode_local_wall_s=decode_local_wall_s,
            merge_wall_s=decode_merge_rank0_wall_s,
            filter_wall_s=rank0_filter_wall_s,
            csv_write_wall_s=csv_write_wall_s,
            barrier_after_inference_wall_s=barrier_after_loop_s,
            collective_gather_paths_wall_s=collective_gather_paths_wall_s,
            trainer_epoch_time_s=trainer_epoch_time_s,
            inference_only_avg_step_ms=inf_avg_ms,
        )

    plog.line(
        f"step4_perf_summary "
        f"inference_loop_wall_s={inference_loop_wall_s:.4f} "
        f"decode_tail_wall_s={decode_tail_wall_s:.4f} "
        f"csv_write_wall_s={csv_write_wall_s:.4f} "
        f"step4_end_to_end_wall_s={step4_end_to_end_wall_s:.4f} "
        f"total_wall_s__alias_of_step4_e2e={step4_end_to_end_wall_s:.4f} "
        f"inference_only_avg_step_ms={inf_avg_ms:.4f} "
        f"note=primary_total_is_step4_end_to_end_wall_s_full_task_preprocess_through_csv"
    )

    plog.line("step4_task_done")


