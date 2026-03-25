import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from cpu_utils import effective_cpu_count

# ---------------------------------------------------------------------------
# 类型与轻量配置对象（供静态检查与逐步向「入口 resolve、后续只读」演进）
# ---------------------------------------------------------------------------


class TaskConfig(TypedDict):
    """单任务默认表 + 命名预设可覆盖 lr/coef/adv 后的结构（auxiliary/target 仅来自 task_configs）。"""

    auxiliary: str
    target: str
    lr: Union[int, float]
    coef: Union[int, float]
    adv: Union[int, float]


class TrainingPresetRow(TypedDict, total=False):
    """命名预设中允许出现的字段（全局一条或 per-task 子 dict）。"""

    train_batch_size: int
    epochs: int
    full_eval_every_epochs: int
    min_lr_ratio: float
    adv: float
    lr: float
    coef: float
    gradient_accumulation_steps: int
    per_device_train_batch_size: int


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    """
    不含 CLI 的只读快照：由当前环境变量 + 命名预设 + 模块默认解析。
    训练脚本仍应在入口用「CLI 非 None 则覆盖」；本对象便于日志与调试统一打印。
    """

    task_idx: Optional[int]
    global_train_batch_size: int
    epochs: int
    min_lr_ratio: float
    full_eval_every_epochs_uniform: int


T = TypeVar("T")


def _training_resolve_value(
    *,
    cli: Optional[T],
    env_names: Tuple[str, ...],
    env_reader: Optional[Callable[[str], T]] = None,
    task_idx: Optional[int],
    preset_key: str,
    code_default: T,
    preset_reader: Optional[Callable[[Any], T]] = None,
    after_preset: Optional[Callable[[T], T]] = None,
) -> T:
    """
    统一优先级（在 config 层可见部分）：CLI > ENV（按 env_names 顺序）> TRAINING_PRESET 切片 > 代码默认。

    说明：各训练脚本的 CLI 参数在 parse 后若需覆盖，由调用方传入 cli；本模块多数 getter 不含 CLI，
    则 cli 恒为 None，退化为 ENV > preset > code。
    """
    if cli is not None:
        return cli  # type: ignore[return-value]
    for name in env_names:
        if name in os.environ:
            raw = os.environ[name]
            if env_reader is not None:
                v = env_reader(raw)
            else:
                raise RuntimeError(f"env_reader 未提供但存在环境变量 {name}")
            return v
    row = _active_train_preset_slice(task_idx)
    if row and preset_key in row:
        raw = row[preset_key]
        if preset_reader is not None:
            v = preset_reader(raw)
        else:
            v = raw  # type: ignore[assignment]
        if after_preset is not None:
            v = after_preset(v)
        return v  # type: ignore[return-value]
    return code_default


def _read_env_float(name: str) -> float:
    return float(os.environ[name])


def _read_env_int(name: str) -> int:
    return int(os.environ[name])


def _preset_int_min(raw: Any, minimum: int) -> int:
    return max(minimum, int(raw))


def _coerce_task_param_numeric(v: Any) -> Union[int, float]:
    """与 task_configs 中 coef 等为整数时的 format 输出兼容：整数值用 int，否则 float。"""
    fv = float(v)
    if math.isfinite(fv) and fv.is_integer() and abs(fv) <= 2**53:
        return int(fv)
    return fv


# ---------------------------------------------------------------------------
# 训练模式（reproduction / optimized）
# ---------------------------------------------------------------------------


def _normalized_train_mode() -> str:
    """
    训练工程模式（运行时读取环境变量，便于 CLI 在 parse_args 后写入 D4C_TRAIN_MODE）。
    - reproduction / repro / paper / default：论文复现取向（默认）。
    - optimized：大 batch / DDP 工程优化取向，非严格复现。
    """
    m = os.environ.get("D4C_TRAIN_MODE", "reproduction").strip().lower()
    if m in ("reproduction", "repro", "paper", "default"):
        return "reproduction"
    if m == "optimized":
        return "optimized"
    return "reproduction"


def get_train_mode() -> str:
    return _normalized_train_mode()


def get_exact_reproduction() -> bool:
    return get_train_mode() == "reproduction"


def get_allow_large_batch() -> bool:
    """optimized 模式下语义上允许更大 global batch（实际 batch 仍由 CLI/config 决定）。"""
    return get_train_mode() == "optimized"


def _env_int_choice(name: str, repro_default: int, opt_default: int) -> int:
    if name in os.environ:
        return max(1, int(os.environ[name]))
    return opt_default if get_train_mode() == "optimized" else repro_default


def get_lr_scheduler_type() -> str:
    """
    none：保持 valid 变差时手工减半学习率（原行为）。
    warmup_cosine：按步数 warmup + cosine（与手工减半互斥，由训练循环实现）。
    未设置环境变量时：reproduction 为 none；optimized 默认为 warmup_cosine。
    """
    if "D4C_LR_SCHEDULER" in os.environ:
        v = os.environ["D4C_LR_SCHEDULER"].strip().lower()
        if v in ("none", "off", "disabled"):
            return "none"
        if v in ("warmup_cosine", "warmup-cosine", "cosine"):
            return "warmup_cosine"
        return "none"
    return "warmup_cosine" if get_train_mode() == "optimized" else "none"


def get_warmup_epochs() -> float:
    """未指定 D4C_WARMUP_STEPS / D4C_WARMUP_RATIO 时，用 epoch 数换算 warmup steps（兼容旧配置）。"""
    return float(os.environ.get("D4C_WARMUP_EPOCHS", "1.0"))


def get_d4c_warmup_steps_optional() -> Optional[int]:
    """显式 warmup 步数；未设置环境变量则 None。D4C_WARMUP_STEPS 优先于 D4C_WARMUP_RATIO。"""
    if "D4C_WARMUP_STEPS" not in os.environ:
        return None
    v = int(os.environ["D4C_WARMUP_STEPS"])
    return v if v > 0 else None


def get_d4c_warmup_ratio_optional() -> Optional[float]:
    """显式 warmup 占计划总步数比例；未设置则 None。"""
    if "D4C_WARMUP_RATIO" not in os.environ:
        return None
    return float(os.environ["D4C_WARMUP_RATIO"])


def get_min_lr_ratio(task_idx: Optional[int] = None) -> float:
    """
    cosine 末端倍率下限（相对 initial_lr）。
    - reproduction：模块默认 0.0（可衰减到底，贴近旧论文行为）
    - optimized：模块默认 0.1（生成/解释任务末端保留一定步长，减轻 lr 过小导致欠拟合）
    命名预设 / D4C_MIN_LR_RATIO 仍优先于上述默认。
    """
    _default = 0.1 if get_train_mode() == "optimized" else 0.0
    return _training_resolve_value(
        cli=None,
        env_names=("D4C_MIN_LR_RATIO",),
        env_reader=_read_env_float,
        task_idx=task_idx,
        preset_key="min_lr_ratio",
        code_default=_default,
        preset_reader=float,
    )


def get_scheduler_initial_lr(cli_learning_rate: float) -> float:
    """优化器使用的初始 LR；D4C_INITIAL_LR 优先于 CLI --learning_rate。"""
    if "D4C_INITIAL_LR" in os.environ:
        return float(os.environ["D4C_INITIAL_LR"])
    return float(cli_learning_rate)


def _full_eval_uniform_interval_from_env() -> Optional[int]:
    """若设置了固定间隔环境变量则返回 N（>=0），否则 None。"""
    if "D4C_FULL_EVAL_EVERY" in os.environ:
        return max(0, int(os.environ["D4C_FULL_EVAL_EVERY"]))
    if "D4C_FULL_BLEU_EVAL_EVERY" in os.environ:
        return max(0, int(os.environ["D4C_FULL_BLEU_EVAL_EVERY"]))
    return None


def _full_eval_uniform_interval_from_preset(task_idx: Optional[int]) -> Optional[int]:
    """命名预设中的固定 full eval 间隔；无键返回 None（与旧逻辑一致）。"""
    row = _active_train_preset_slice(task_idx)
    if not row or "full_eval_every_epochs" not in row:
        return None
    return max(0, int(row["full_eval_every_epochs"]))


def get_full_eval_every_epochs(task_idx: Optional[int] = None) -> int:
    """
    固定间隔：仅当设置了 D4C_FULL_EVAL_EVERY 或 D4C_FULL_BLEU_EVAL_EVERY 时返回 N；否则为 0。
    optimized 且未设上述变量时，训练入口应改用 resolve_full_bleu_eval_training() 得到分阶段 schedule。
    """
    v = _full_eval_uniform_interval_from_env()
    if v is not None:
        return v
    u = _full_eval_uniform_interval_from_preset(task_idx)
    if u is not None:
        return u
    return 0


def get_full_eval_phased_schedule_default() -> Optional[Tuple[int, int, int]]:
    """
    optimized 且未设置 D4C_FULL_EVAL_* 固定间隔时启用分阶段 full BLEU：
    epoch<=boundary 每 early 轮一次；epoch>boundary 每 late 轮一次（默认 5 / 10 / 2）。
    可用 D4C_FULL_EVAL_EARLY_EVERY、D4C_FULL_EVAL_PHASE_END_EPOCH、D4C_FULL_EVAL_LATE_EVERY 覆盖。
    """
    if get_train_mode() != "optimized":
        return None
    if _full_eval_uniform_interval_from_env() is not None:
        return None
    early = max(1, int(os.environ.get("D4C_FULL_EVAL_EARLY_EVERY", "5")))
    boundary = max(1, int(os.environ.get("D4C_FULL_EVAL_PHASE_END_EPOCH", "10")))
    late = max(1, int(os.environ.get("D4C_FULL_EVAL_LATE_EVERY", "2")))
    return (early, boundary, late)


def should_run_full_bleu_eval_epoch(
    epoch_1_based: int,
    full_eval_every: int,
    phased: Optional[Tuple[int, int, int]],
) -> bool:
    """epoch_1_based 为从 1 开始的 epoch 序号。"""
    if phased is not None:
        early, boundary, late = phased
        e = epoch_1_based
        if e <= boundary:
            return e > 0 and e % early == 0
        return (e - boundary) % late == 0
    if full_eval_every <= 0:
        return False
    return epoch_1_based % full_eval_every == 0


def resolve_full_bleu_eval_training(
    cli_full_eval_every: Optional[int],
    *,
    task_idx: Optional[int] = None,
) -> Tuple[int, Optional[Tuple[int, int, int]]]:
    """
    返回 (uniform_every, phased)。uniform_every>0 时每 N epoch 一次 full BLEU；phased 非 None 时忽略 uniform（应为 0）。
    CLI 优先，其次环境变量固定间隔，再次命名预设（可按 task_idx），再次 optimized 分阶段默认。
    """
    if cli_full_eval_every is not None:
        return max(0, int(cli_full_eval_every)), None
    u = _full_eval_uniform_interval_from_env()
    if u is not None:
        return u, None
    u = _full_eval_uniform_interval_from_preset(task_idx)
    if u is not None:
        return u, None
    phased = get_full_eval_phased_schedule_default()
    if phased is not None:
        return 0, phased
    return 0, None


def get_full_bleu_eval_every_epochs(task_idx: Optional[int] = None) -> int:
    """与 get_full_eval_every_epochs(task_idx) 同义（兼容旧名）；支持 per-task 预设与 D4C_PRESET_TASK_ID。"""
    return get_full_eval_every_epochs(task_idx)


def snapshot_resolved_training_config(task_idx: Optional[int] = None) -> ResolvedTrainingConfig:
    """当前 ENV + 命名预设 + 模块默认下的只读快照（不含 CLI）。"""
    return ResolvedTrainingConfig(
        task_idx=task_idx,
        global_train_batch_size=get_train_batch_size(task_idx),
        epochs=get_epochs(task_idx),
        min_lr_ratio=get_min_lr_ratio(task_idx),
        full_eval_every_epochs_uniform=get_full_eval_every_epochs(task_idx),
    )


def get_quick_eval_max_samples(resolved_bleu4_max_samples: int) -> int:
    """
    quick eval 子集大小（每 epoch）。D4C_QUICK_EVAL_MAX_SAMPLES 优先；否则用 TRAIN_BLEU4 / CLI 解析结果 resolved_bleu4_max_samples。
    """
    if "D4C_QUICK_EVAL_MAX_SAMPLES" in os.environ:
        return max(64, int(os.environ["D4C_QUICK_EVAL_MAX_SAMPLES"]))
    return max(64, int(resolved_bleu4_max_samples))


def apply_optimized_torch_backends() -> bool:
    """
    optimized 且未禁用 D4C_DDP_FAST 时启用 TF32、cudnn.benchmark（加速，数值与论文不完全一致）。
    须在 CUDA 可用且尽量在 set_device 之后调用。
    """
    import torch

    if get_train_mode() != "optimized":
        return False
    v = os.environ.get("D4C_DDP_FAST", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return True


# ---------------------------------------------------------------------------
# 全局默认（模块级常量）
# ---------------------------------------------------------------------------

# 全局配置（Step 1+2 嵌入）
# 可通过 run_step1_step2.sh --embed-batch-size N 或 run_preprocess_and_embed.py --embed-batch-size N 覆盖
embed_batch_size = 1024  # compute_embeddings 嵌入批次大小，显存不足时可减小（64/128），多卡可增大（512/1024）

# 全局配置（CPU 并行）
# 与作业实际可用核数对齐：优先 sched_getaffinity（ cgroup/cpuset 下常为 nproc ），非 Linux 则 os.cpu_count()
# 也可 export RUNNING_CPU_COUNT=12 显式指定（与 shell `nproc` 一致）
# 可通过 run_step3/4/5.sh --num-proc N 或各 Python 脚本 --num-proc N 覆盖
_num_cpu = effective_cpu_count()
# 默认可并联上限：与常见单节点 GPU 作业「最多约 12 核」对齐；更大机器请 export MAX_PARALLEL_CPU=16 等
_MAX_PARALLEL_CPU = max(1, int(os.environ.get("MAX_PARALLEL_CPU", "12")))
# datasets.map（Tokenize）等并行进程数；与 PyTorch DataLoader 的 num_workers 独立，见 get_dataloader_num_workers()
num_proc = min(_num_cpu, _MAX_PARALLEL_CPU)

# 全局配置（Step 3/4/5 训练与推理）
# 可通过 run_step3/4/5.sh --batch-size N 或各 Python 脚本 --batch-size N 覆盖 CLI；本模块侧为 ENV > 命名预设 > 下列默认值。
#
# train_batch_size（模块默认）：训练用「全局有效 batch」G——每个优化步在全体 rank 上合计的样本数。
#   关系式：G = per_device_micro_batch × world_size × gradient_accumulation_steps
#   其中 per_device_micro_batch 常记为 P（每 rank 每步 forward 的样本数，即 DataLoader batch_size）。
train_batch_size = 2048  # 默认 G；显存不足时请配合梯度累积或 D4C_PER_DEVICE_BATCH_SIZE / per_device_train_batch_size 预设
# 每优化步内累积多少个微批再 optimizer.step；1 表示不累积。亦可用 D4C_GRADIENT_ACCUMULATION_STEPS 或命名预设。
gradient_accumulation_steps = 1
# eval_batch_size：推理/验证用「全局 batch」（各 rank 合计）；DDP 下每 rank 批大小 = eval_batch_size / world_size
eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "2560"))
epochs = 50  # Step 3 域对抗预训练、Step 5 主训练的默认轮数（CLI > 预设 > 本值）

# 早停与选模（可被 CLI 或环境变量覆盖；见 AdvTrain / run-d4c）
# TRAIN_MIN_EPOCHS：至少训练多少 epoch 后才允许因 valid 变差而早停（reproduction 默认 30；optimized 默认 8）
# TRAIN_EARLY_STOP_PATIENCE：valid 连续变差次数阈值（reproduction 默认 15；optimized 默认 6）
# TRAIN_EARLY_STOP_PATIENCE_FULL：optimized 下按 full BLEU eval 次数计的早停耐心（未设则与 TRAIN_EARLY_STOP_PATIENCE 相同）
# TRAIN_EARLY_STOP_PATIENCE_LOSS：dual_bleu 下仅针对 valid_loss 变差计数的早停耐心（未设 CLI 时见 resolve_early_stop_patience_loss）
# TRAIN_BLEU4_MAX_SAMPLES：按 BLEU-4 选模时 quick 评估最多条数（reproduction 默认 2048；optimized 默认 512）
# D4C_TRAIN_MODE=optimized 时若未设置上述环境变量，则采用 optimized 一侧默认值（见 getter）

task_configs: Dict[int, TaskConfig] = {
    1: {
        "auxiliary": "AM_Electronics",
        "target": "AM_CDs",
        "lr": 5e-4,
        "coef": 1,
        "adv": 0.01,
    },
    2: {
        "auxiliary": "AM_Movies",
        "target": "AM_CDs",
        "lr": 1e-3,
        "coef": 0.1,
        "adv": 0.01,
    },
    3: {
        "auxiliary": "AM_CDs",
        "target": "AM_Electronics",
        "lr": 5e-4,
        "coef": 0.5,
        "adv": 0.1,
    },
    4: {
        "auxiliary": "AM_Movies",
        "target": "AM_Electronics",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01,
    },
    5: {
        "auxiliary": "AM_CDs",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01,
    },
    6: {
        "auxiliary": "AM_Electronics",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01,
    },
    7: {
        "auxiliary": "Yelp",
        "target": "TripAdvisor",
        "lr": 1e-4,
        "coef": 0.5,
        "adv": 0.01,
    },
    8: {
        "auxiliary": "TripAdvisor",
        "target": "Yelp",
        "lr": 5e-4,
        "coef": 1,
        "adv": 0.01,
    },
}


def resolve_task_idx_from_aux_target(auxiliary: str, target: str) -> Optional[int]:
    """由 auxiliary/target 反查任务号 1–8；未知组合返回 None。"""
    for tid, cfg in task_configs.items():
        if cfg["auxiliary"] == auxiliary and cfg["target"] == target:
            return int(tid)
    return None


# ---------------------------------------------------------------------------
# 命名训练预设（可选）
#
# 使用：export D4C_TRAIN_PRESET=<键名>
#
# 两种形态（可同时保留多套预设）：
#   1) 全局一条：顶层直接为字段 dict，如 {"train_batch_size": 1024, "epochs": 30, ...}，对所有任务相同。
#   2) 按任务：顶层键为整数 1..8，值为该任务的字段 dict；各任务可分别改 batch / epochs / lr / coef / adv 等。
#
# 在「未通过 CLI / 环境变量显式覆盖」时，下列 getter 会采用当前任务对应切片：
#   - 全局训练 batch（G）→ get_train_batch_size(task_idx)
#   - gradient_accumulation_steps / per_device_train_batch_size → 见 get_* 与 resolve_ddp_train_microbatch_layout
#   - epochs → get_epochs(task_idx)
#   - full_eval_every_epochs → resolve_full_bleu_eval_training(..., task_idx=) / get_full_eval_every_epochs(task_idx)
#   - min_lr_ratio → get_min_lr_ratio(task_idx)（仍低于 D4C_MIN_LR_RATIO 环境变量）
#   - lr / coef / adv → get_task_config(task_idx) / format_step3_task_params_line()
#
# task_idx 来源：函数实参，或环境变量 D4C_PRESET_TASK_ID（shell 的 run_step3 每任务会设置）。
# 总优先级：CLI > 对应环境变量 > 预设切片 > 代码默认（mode 相关项在各自 getter 内处理）。
# ---------------------------------------------------------------------------

_TRAINING_PRESET_ALLOWED_KEYS: FrozenSet[str] = frozenset(
    {
        "train_batch_size",
        "epochs",
        "full_eval_every_epochs",
        "min_lr_ratio",
        "adv",
        "lr",
        "coef",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
    }
)

_TRAINING_PRESET_INT_KEYS: FrozenSet[str] = frozenset(
    {
        "train_batch_size",
        "epochs",
        "full_eval_every_epochs",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
    }
)

_TRAINING_PRESET_FLOAT_KEYS: FrozenSet[str] = frozenset(
    {
        "min_lr_ratio",
        "adv",
        "lr",
        "coef",
    }
)


def _validate_training_presets(presets: Dict[str, Any], *, name: str = "TRAINING_PRESETS") -> None:
    """模块加载时校验：任务键 1..8、字段名合法、数值类型基本合理。"""

    def _check_row(row: Dict[str, Any], ctx: str) -> None:
        unknown = set(row.keys()) - _TRAINING_PRESET_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{name} {ctx} 含有未知字段 {sorted(unknown)}；"
                f"允许: {sorted(_TRAINING_PRESET_ALLOWED_KEYS)}"
            )
        for k, v in row.items():
            if k in _TRAINING_PRESET_INT_KEYS:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"{name} {ctx} 字段 {k!r} 应为整数类型，当前为 {type(v).__name__}")
                if isinstance(v, float) and not float(v).is_integer():
                    raise ValueError(f"{name} {ctx} 字段 {k!r} 应为整数值，当前为 {v!r}")
            elif k in _TRAINING_PRESET_FLOAT_KEYS:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"{name} {ctx} 字段 {k!r} 应为数值类型，当前为 {type(v).__name__}")
                fv = float(v)
                if not math.isfinite(fv):
                    raise ValueError(f"{name} {ctx} 字段 {k!r} 应为有限数值，当前为 {v!r}")

    for preset_name, blob in presets.items():
        if not isinstance(blob, dict):
            raise TypeError(f'{name} 中预设 {preset_name!r} 应为 dict，当前为 {type(blob).__name__}')
        keys = list(blob.keys())
        if keys and all(isinstance(k, int) and 1 <= k <= 8 for k in keys):
            for tid, row in blob.items():
                if not isinstance(row, dict):
                    raise TypeError(
                        f'{name} 预设 {preset_name!r} 的任务 {tid} 值应为 dict，当前为 {type(row).__name__}'
                    )
                _check_row(row, f"预设 {preset_name!r} task={tid}")
        else:
            _check_row(blob, f"预设 {preset_name!r}（全局）")


_GB1024_EP30_FE2_ROW: Dict[str, Any] = {
    "train_batch_size": 1024,
    "epochs": 30,
    "full_eval_every_epochs": 2,
    "min_lr_ratio": 0.1,
    "adv": 0.005,
}

TRAINING_PRESETS: Dict[str, Any] = {
    "gb1024_ep30_fe2": {i: dict(_GB1024_EP30_FE2_ROW) for i in range(1, 9)},
}

_validate_training_presets(TRAINING_PRESETS)


def _named_train_preset_blob() -> Optional[Dict[Any, Any]]:
    name = (os.environ.get("D4C_TRAIN_PRESET") or "").strip()
    if not name:
        return None
    raw = TRAINING_PRESETS.get(name)
    return raw if isinstance(raw, dict) else None


def _preset_blob_kind(blob: Dict[Any, Any]) -> str:
    keys = list(blob.keys())
    if keys and all(isinstance(k, int) and 1 <= k <= 8 for k in keys):
        return "per_task"
    return "global"


def _resolve_preset_task_idx(explicit: Optional[int]) -> Optional[int]:
    if explicit is not None:
        return int(explicit)
    raw = (os.environ.get("D4C_PRESET_TASK_ID") or "").strip()
    if raw.isdigit():
        v = int(raw)
        if 1 <= v <= 8:
            return v
    return None


def _active_train_preset_slice(task_idx: Optional[int]) -> Optional[Dict[str, Any]]:
    blob = _named_train_preset_blob()
    if not blob:
        return None
    kind = _preset_blob_kind(blob)
    if kind == "global":
        return blob
    tid = _resolve_preset_task_idx(task_idx)
    if tid is None:
        return None
    row = blob.get(tid)
    return row if isinstance(row, dict) else None


def training_preset_is_per_task() -> bool:
    """为真时，未传 --batch-size 的 *_optimized.sh 不应统一注入 batch（由 Python 按任务解析）。"""
    blob = _named_train_preset_blob()
    if not blob:
        return False
    return _preset_blob_kind(blob) == "per_task"


def get_task_config(task_idx: int) -> Optional[TaskConfig]:
    base = task_configs.get(task_idx)
    if base is None:
        return None
    p = _active_train_preset_slice(int(task_idx))
    if not p:
        return base
    merged: TaskConfig = dict(base)
    for k in ("lr", "coef", "adv"):
        if k in p:
            merged[k] = _coerce_task_param_numeric(p[k])  # type: ignore[assignment]
    return merged


def format_step3_task_params_line(task_idx: int) -> str:
    """供 run_step3.sh / run_step5.sh 解析：auxiliary target lr coef adv（空格分隔）。"""
    c = get_task_config(task_idx)
    if not c:
        return ""
    return f"{c['auxiliary']} {c['target']} {c['lr']} {c['coef']} {c['adv']}"


def get_train_batch_size(task_idx: Optional[int] = None) -> int:
    """训练「全局有效 batch」G（每优化步跨所有 rank 的样本总数）；CLI 覆盖在脚本侧。"""
    return _training_resolve_value(
        cli=None,
        env_names=(),
        task_idx=task_idx,
        preset_key="train_batch_size",
        code_default=train_batch_size,
        preset_reader=lambda raw: _preset_int_min(raw, 1),
    )


def get_gradient_accumulation_steps(task_idx: Optional[int] = None) -> int:
    """梯度累积步数 A（每优化步内的微批次数）；D4C_GRADIENT_ACCUMULATION_STEPS 优先于命名预设与模块默认。"""
    return _training_resolve_value(
        cli=None,
        env_names=("D4C_GRADIENT_ACCUMULATION_STEPS",),
        env_reader=lambda s: _preset_int_min(s, 1),
        task_idx=task_idx,
        preset_key="gradient_accumulation_steps",
        code_default=gradient_accumulation_steps,
        preset_reader=lambda raw: _preset_int_min(raw, 1),
    )


def get_per_device_train_batch_size_optional(task_idx: Optional[int] = None) -> Optional[int]:
    """
    若设置，则与全局 batch G、world_size 一起推导出梯度累积步数 A。
    优先级：D4C_PER_DEVICE_BATCH_SIZE > 命名预设 per_device_train_batch_size（无模块级默认，未设返回 None）。
    """
    if "D4C_PER_DEVICE_BATCH_SIZE" in os.environ:
        return max(1, int(os.environ["D4C_PER_DEVICE_BATCH_SIZE"]))
    row = _active_train_preset_slice(task_idx)
    if row and "per_device_train_batch_size" in row:
        return max(1, int(row["per_device_train_batch_size"]))
    return None


def resolve_ddp_train_microbatch_layout(
    global_batch_size: int,
    world_size: int,
    *,
    per_device_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    task_idx: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    由「全局有效 batch」G 解析 (G, P, A)，满足 G = P × world_size × A。

    - P：每 rank、每 optimizer step 的 DataLoader 微批大小（per-rank micro-batch）。
    - A：gradient_accumulation_steps。
    - 若给定 per_device_batch_size（CLI 或 get_per_device_train_batch_size_optional），则 A = G / (P × world_size)，
      且当同时显式指定 gradient_accumulation_steps（非 None 的实参）时必须与推导的 A 一致。
    - 否则 A 来自 gradient_accumulation_steps 实参，若无实参则用 get_gradient_accumulation_steps(task_idx)，
      再令 P = G / (world_size × A)。
    """
    G = int(global_batch_size)
    W = int(world_size)
    if G < 1:
        raise ValueError(f"global_batch_size 须 >= 1，当前为 {G}")
    if W < 1:
        raise ValueError(f"world_size 须 >= 1，当前为 {W}")

    explicit_accum_arg = gradient_accumulation_steps is not None

    if per_device_batch_size is not None:
        P = max(1, int(per_device_batch_size))
        prod = P * W
        if G % prod != 0:
            raise ValueError(
                f"全局 batch ({G}) 须能被 per_device_batch_size×world_size ({prod}) 整除，"
                f"请调整 --batch-size、--per-device-batch-size 或进程数。"
            )
        A = G // prod
        if explicit_accum_arg:
            A_req = max(1, int(gradient_accumulation_steps))
            if A_req != A:
                raise ValueError(
                    f"per_device_batch_size={P} 与 world_size={W} 推得 gradient_accumulation_steps={A}，"
                    f"与显式指定的 {A_req} 不一致；请只指定一侧或改成相容组合。"
                )
        return G, P, A

    if explicit_accum_arg:
        A = max(1, int(gradient_accumulation_steps))  # type: ignore[arg-type]
    else:
        A = get_gradient_accumulation_steps(task_idx)

    denom = W * A
    if G % denom != 0:
        raise ValueError(
            f"全局 batch ({G}) 须能被 world_size×gradient_accumulation_steps ({denom}) 整除；"
            f"当前 world_size={W}，accum={A}。可设置 D4C_PER_DEVICE_BATCH_SIZE / --per-device-batch-size "
            f"以在保持全局 batch 不变的前提下换用更小的每卡微批。"
        )
    P = G // denom
    return G, P, A


def get_embed_batch_size():
    """返回 embedding 计算的 batch_size，供 run_preprocess_and_embed / compute_embeddings 使用"""
    return embed_batch_size


def get_eval_batch_size():
    """返回 eval 推理阶段的「全局 batch」；DDP 下每 rank 批大小 = 该值 / world_size"""
    return eval_batch_size


def get_epochs(task_idx: Optional[int] = None) -> int:
    """返回训练轮数，供 Step 3/5 使用。"""
    return _training_resolve_value(
        cli=None,
        env_names=(),
        task_idx=task_idx,
        preset_key="epochs",
        code_default=epochs,
        preset_reader=lambda raw: _preset_int_min(raw, 1),
    )


def get_train_min_epochs():
    """至少训练轮数后再允许早停；未设置 TRAIN_MIN_EPOCHS 时依 D4C_TRAIN_MODE 取默认。"""
    return _env_int_choice("TRAIN_MIN_EPOCHS", 30, 8)


def get_train_early_stop_patience():
    """valid 连续变差多少次触发早停；未设置 TRAIN_EARLY_STOP_PATIENCE 时依 mode 取默认。"""
    return _env_int_choice("TRAIN_EARLY_STOP_PATIENCE", 15, 6)


def _first_env_int_max1(names: Tuple[str, ...]) -> Optional[int]:
    for n in names:
        if n in os.environ:
            return max(1, int(os.environ[n]))
    return None


def get_train_early_stop_patience_full() -> int:
    """
    optimized + full BLEU 周期评估时：连续多少次 **full eval** 未刷新 best 则早停（quick eval 不参与计数）。
    未设置 TRAIN_EARLY_STOP_PATIENCE_FULL / D4C_EARLY_STOP_PATIENCE_FULL 时，与 TRAIN_EARLY_STOP_PATIENCE / get_train_early_stop_patience() 同义。
    """
    v = _first_env_int_max1(("TRAIN_EARLY_STOP_PATIENCE_FULL", "D4C_EARLY_STOP_PATIENCE_FULL"))
    if v is not None:
        return v
    return get_train_early_stop_patience()


def resolve_early_stop_patience_loss(
    cli_patience_loss: Optional[int],
    cli_patience: Optional[int],
) -> int:
    """
    dual_bleu 下 valid_loss 连续变差早停耐心，与 full BLEU 的 patience_full 独立。
    优先级：--early-stop-patience-loss > TRAIN_EARLY_STOP_PATIENCE_LOSS / D4C_EARLY_STOP_PATIENCE_LOSS
    > --early-stop-patience > get_train_early_stop_patience()。
    """
    if cli_patience_loss is not None:
        return max(1, int(cli_patience_loss))
    v = _first_env_int_max1(("TRAIN_EARLY_STOP_PATIENCE_LOSS", "D4C_EARLY_STOP_PATIENCE_LOSS"))
    if v is not None:
        return v
    if cli_patience is not None:
        return max(1, int(cli_patience))
    return get_train_early_stop_patience()


def get_train_bleu4_max_samples():
    """BLEU-4 quick 评估最大条数；未设置 TRAIN_BLEU4_MAX_SAMPLES 时依 mode 取默认。"""
    if "TRAIN_BLEU4_MAX_SAMPLES" in os.environ:
        return max(64, int(os.environ["TRAIN_BLEU4_MAX_SAMPLES"]))
    return 512 if get_train_mode() == "optimized" else 2048


def get_num_proc():
    """返回 datasets.map（Tokenize）等使用的并行进程数；与 DataLoader 的 num_workers 无关，见 get_dataloader_num_workers()"""
    return num_proc


def get_max_parallel_cpu():
    """与 MAX_PARALLEL_CPU 一致（默认 12）；DDP 每 rank 的 DataLoader worker 上限等可复用。"""
    return _MAX_PARALLEL_CPU


def get_dataloader_num_workers(split="train"):
    """
    PyTorch DataLoader 的 num_workers，与 datasets.map 的 num_proc 独立。
    split: 'train' | 'valid' | 'test' — 训练/验证/推理测试可适当区分上限。
    单路 worker 数不超过 _MAX_PARALLEL_CPU（默认 12），避免在 12 核节点上过度抢占。
    """
    n = _num_cpu or 8
    cap_t = min(_MAX_PARALLEL_CPU, 16)  # 训练侧单 DataLoader 上限
    cap_v = min(max(4, _MAX_PARALLEL_CPU // 2), 8)  # 验证/测试略保守
    if split == "train":
        return min(max(2, n // 2), cap_t)
    if split in ("valid", "test"):
        return min(max(1, n // 4), cap_v)
    return min(max(1, n // 4), cap_v)


@contextmanager
def hf_datasets_progress_bar(enabled: bool):
    """仅 rank0 显示 datasets.map 的 tqdm；其他 rank 关闭，避免 torchrun 多进程重复进度条与日志。"""
    if enabled:
        yield
        return
    try:
        from datasets.utils.logging import disable_progress_bar
    except ImportError:
        yield
        return
    disable_progress_bar()
    yield


def get_dataloader_prefetch_factor(num_workers: int):
    """num_workers==0 时必须为 None。默认 4，可用环境变量 DATALOADER_PREFETCH_FACTOR 覆盖。"""
    if num_workers <= 0:
        return None
    return max(2, int(os.environ.get("DATALOADER_PREFETCH_FACTOR", "4")))


def get_ddp_train_num_workers_per_rank(world_size: int) -> int:
    """
    DDP 下每个训练进程的 DataLoader worker 数。
    在 world_size × workers 不超过 MAX_PARALLEL_CPU 均分份额的前提下，
    尽量用满 get_dataloader_num_workers('train')，比「仅按 world_size 缩小」更易利用空闲 CPU。
    """
    ws = max(int(world_size), 1)
    dl_train = get_dataloader_num_workers("train")
    share = max(1, get_max_parallel_cpu() // ws)
    return max(1, min(dl_train, share))
