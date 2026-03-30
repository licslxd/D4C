import math
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from cpu_utils import effective_cpu_count
from training_runtime_inputs import collect_training_runtime_overrides_from_args

# 执行层：DDP_NPROC / torchrun --nproc_per_node 仅在 shell 解析；Python 以 WORLD_SIZE 为准。
# D4C_NUM_PROC 为 CPU 侧（如 datasets.map）并行度，属 runtime 链，勿与 DDP_NPROC 混淆。详见 docs/D4C_Scripts_and_Runtime_Guide.md。

# import 后由 train_logging.flush_preset_load_events 写入训练日志（摘要侧）
PRESET_LOAD_EVENTS: List[str] = []


def record_preset_event(msg: str) -> None:
    PRESET_LOAD_EVENTS.append(msg)


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


class RuntimePresetRow(TypedDict, total=False):
    """命名 runtime 预设（CPU / DataLoader 并发）；与训练 TRAINING_PRESETS 独立。"""

    max_parallel_cpu: int
    num_proc: int
    dataloader_num_workers_train: int
    dataloader_num_workers_valid: int
    dataloader_num_workers_test: int
    dataloader_prefetch_factor_train: int
    dataloader_prefetch_factor_valid: int
    dataloader_prefetch_factor_test: int
    dataloader_workers_train_per_rank_cap: int


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
class BaseTrainingDefaults:
    """
    训练相关**代码默认**的单一来源（不含 task 表、不含 preset、不在 import 时读 ENV）。

    解析链目标形态：BASE_TRAINING_DEFAULTS → ENV / 命名预设 / CLI → 解析结果。
    """

    epochs: int = 50
    train_batch_size: int = 2048
    gradient_accumulation_steps: int = 1
    min_lr_ratio: float = 0.1
    train_min_epochs: int = 8
    train_early_stop_patience: int = 6
    train_bleu4_max_samples: int = 512
    lr_scheduler: str = "warmup_cosine"
    warmup_epochs: float = 1.0
    # eval 全局 batch 代码默认；运行时仍优先读 EVAL_BATCH_SIZE
    eval_batch_size: int = 2560
    # 无任务 lr 时 learning_rate resolve 链末级 fallback（在 build_resolved_training_config 内使用）
    initial_learning_rate: float = 1e-3


BASE_TRAINING_DEFAULTS = BaseTrainingDefaults()
DEFAULT_TRAINING_CONFIG = BASE_TRAINING_DEFAULTS
"""与 BASE_TRAINING_DEFAULTS 同义别名，便于语义上称「默认训练配置」。"""

# 历史兼容：与 ``from config import train_batch_size`` / ``epochs`` 等保持一致（值均来自 BASE_TRAINING_DEFAULTS）
train_batch_size = BASE_TRAINING_DEFAULTS.train_batch_size
gradient_accumulation_steps = BASE_TRAINING_DEFAULTS.gradient_accumulation_steps
epochs = BASE_TRAINING_DEFAULTS.epochs


T = TypeVar("T")


def _preset_int_min(raw: Any, minimum: int) -> int:
    return max(minimum, int(raw))


def _coerce_task_param_numeric(v: Any) -> Union[int, float]:
    """与 task_configs 中 coef 等为整数时的 format 输出兼容：整数值用 int，否则 float。"""
    fv = float(v)
    if math.isfinite(fv) and fv.is_integer() and abs(fv) <= 2**53:
        return int(fv)
    return fv


# ---------------------------------------------------------------------------
# presets/*.yaml：优先加载（相对仓库根 presets/）；缺失、错误或缺 PyYAML 时回退下方内置表
# ---------------------------------------------------------------------------


def _d4c_presets_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_yaml_file(path: Path) -> Any:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_TASK_ROW_KEYS: FrozenSet[str] = frozenset({"auxiliary", "target", "lr", "coef", "adv"})


def _normalize_task_row_yaml(v: Any, *, ctx: str) -> TaskConfig:
    if not isinstance(v, dict):
        raise TypeError(f"{ctx} 须为 dict，当前为 {type(v).__name__}")
    unk = set(v.keys()) - _TASK_ROW_KEYS
    if unk:
        raise ValueError(f"{ctx} 含有未知字段 {sorted(unk)}")
    miss = _TASK_ROW_KEYS - set(v.keys())
    if miss:
        raise ValueError(f"{ctx} 缺少字段 {sorted(miss)}")
    return {
        "auxiliary": str(v["auxiliary"]),
        "target": str(v["target"]),
        "lr": _coerce_task_param_numeric(v["lr"]),
        "coef": _coerce_task_param_numeric(v["coef"]),
        "adv": _coerce_task_param_numeric(v["adv"]),
    }


def _try_load_task_defaults_from_yaml() -> Optional[Dict[int, TaskConfig]]:
    try:
        import yaml  # noqa: F401
    except ImportError:
        record_preset_event("WARN presets/tasks: PyYAML not installed -> builtin task_configs")
        warnings.warn(
            "未安装 PyYAML，跳过 presets/tasks/*.yaml，使用内置 task_configs / TASK_DEFAULTS。",
            UserWarning,
            stacklevel=2,
        )
        return None
    root = _d4c_presets_repo_root() / "presets" / "tasks"
    if not root.is_dir():
        record_preset_event("SKIP presets/tasks: directory missing -> builtin task_configs")
        return None
    paths = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not paths:
        record_preset_event("SKIP presets/tasks: no yaml files -> builtin task_configs")
        return None
    merged: Dict[int, TaskConfig] = {}
    for path in paths:
        try:
            raw = _load_yaml_file(path)
            if not isinstance(raw, dict):
                raise TypeError(f"根须为 mapping，当前为 {type(raw).__name__}")
            for k, v in raw.items():
                tid = int(k)
                if tid < 1 or tid > 8:
                    raise ValueError(f"任务号须在 1..8，收到 {tid!r}（键 {k!r}）")
                merged[tid] = _normalize_task_row_yaml(v, ctx=f"{path.name} task {tid}")
        except Exception as e:
            record_preset_event(f"WARN presets/tasks: load failed {path} -> builtin ({e})")
            warnings.warn(
                f"加载 tasks YAML 失败，回退内置 task_configs: {path}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return None
    need = set(range(1, 9))
    if set(merged.keys()) != need:
        record_preset_event(
            f"WARN presets/tasks: keys must be 1..8, got {sorted(merged.keys())} -> builtin"
        )
        warnings.warn(
            f"presets/tasks 合并后须恰好包含任务 1..8，当前键为 {sorted(merged.keys())}，回退内置。",
            UserWarning,
            stacklevel=2,
        )
        return None
    record_preset_event(f"OK presets/tasks: loaded {len(paths)} yaml file(s)")
    return merged


def _coerce_training_preset_top_level(data: Dict[Any, Any]) -> Any:
    keys = list(data.keys())
    if not keys:
        return data
    if all(isinstance(k, int) and 1 <= k <= 8 for k in keys):
        return {int(k): dict(v) if isinstance(v, dict) else v for k, v in data.items()}
    if all(isinstance(k, str) and k.isdigit() and 1 <= int(k) <= 8 for k in keys):
        return {int(k): dict(v) if isinstance(v, dict) else v for k, v in data.items()}
    return dict(data)


def _try_load_training_presets_from_yaml() -> Optional[Dict[str, Any]]:
    try:
        import yaml  # noqa: F401
    except ImportError:
        record_preset_event("WARN presets/training: PyYAML not installed -> builtin TRAINING_PRESETS")
        warnings.warn(
            "未安装 PyYAML，跳过 presets/training/*.yaml，使用内置 TRAINING_PRESETS。",
            UserWarning,
            stacklevel=2,
        )
        return None
    root = _d4c_presets_repo_root() / "presets" / "training"
    if not root.is_dir():
        record_preset_event("SKIP presets/training: directory missing -> builtin TRAINING_PRESETS")
        return None
    paths = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not paths:
        record_preset_event("SKIP presets/training: no yaml files -> builtin TRAINING_PRESETS")
        return None
    out: Dict[str, Any] = {}
    for path in paths:
        try:
            raw = _load_yaml_file(path)
            if raw is None:
                raise ValueError("文件为空或仅注释")
            if not isinstance(raw, dict):
                raise TypeError(f"根须为 mapping，当前为 {type(raw).__name__}")
            out[path.stem] = _coerce_training_preset_top_level(raw)
        except Exception as e:
            record_preset_event(f"WARN presets/training: load failed {path} -> builtin ({e})")
            warnings.warn(
                f"加载训练预设 YAML 失败，回退内置 TRAINING_PRESETS: {path}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return None
    record_preset_event(f"OK presets/training: loaded {len(paths)} preset file(s)")
    return out


def _try_load_runtime_presets_from_yaml() -> Optional[Dict[str, Dict[str, int]]]:
    try:
        import yaml  # noqa: F401
    except ImportError:
        record_preset_event("WARN presets/runtime: PyYAML not installed -> builtin RUNTIME_PRESETS")
        warnings.warn(
            "未安装 PyYAML，跳过 presets/runtime/*.yaml，使用内置 RUNTIME_PRESETS。",
            UserWarning,
            stacklevel=2,
        )
        return None
    root = _d4c_presets_repo_root() / "presets" / "runtime"
    if not root.is_dir():
        record_preset_event("SKIP presets/runtime: directory missing -> builtin RUNTIME_PRESETS")
        return None
    paths = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not paths:
        record_preset_event("SKIP presets/runtime: no yaml files -> builtin RUNTIME_PRESETS")
        return None
    out: Dict[str, Dict[str, int]] = {}
    for path in paths:
        try:
            raw = _load_yaml_file(path)
            if raw is None:
                raise ValueError("文件为空或仅注释")
            if not isinstance(raw, dict):
                raise TypeError(f"根须为 mapping，当前为 {type(raw).__name__}")
            row: Dict[str, int] = {}
            for kk, vv in raw.items():
                row[str(kk)] = int(vv)
            out[path.stem] = row
        except Exception as e:
            record_preset_event(f"WARN presets/runtime: load failed {path} -> builtin ({e})")
            warnings.warn(
                f"加载 runtime 预设 YAML 失败，回退内置 RUNTIME_PRESETS: {path}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return None
    record_preset_event(f"OK presets/runtime: loaded {len(paths)} preset file(s)")
    return out


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


def apply_ddp_fast_torch_backends() -> bool:
    """
    D4C_DDP_FAST 未禁用时在 CUDA 上启用 TF32、cudnn.benchmark（加速，数值可能因硬件略有差异）。
    须在 CUDA 可用且尽量在 set_device 之后调用。
    """
    import torch

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
# 与作业实际可用核数对齐：优先 sched_getaffinity；不在 import 时冻结，见 _get_num_cpu() / get_num_proc()
# 也可 export RUNNING_CPU_COUNT=12；更大机器请 export MAX_PARALLEL_CPU=16 等
# runtime 解析链的代码默认上限（无 MAX_PARALLEL_CPU 且无 D4C_RUNTIME_PRESET 覆盖时与历史默认 12 一致）
_RUNTIME_BASE_MAX_PARALLEL_CPU = 12


def _get_num_cpu() -> int:
    return int(effective_cpu_count() or 8)


def _resolve_max_parallel_cpu_cli(max_parallel_cli: Optional[int] = None) -> int:
    """runtime_base → runtime_preset → MAX_PARALLEL_CPU → 可选 CLI。"""
    v = max(1, int(_RUNTIME_BASE_MAX_PARALLEL_CPU))
    rp = _active_runtime_preset_slice()
    if rp and "max_parallel_cpu" in rp:
        v = max(1, int(rp["max_parallel_cpu"]))
    if "MAX_PARALLEL_CPU" in os.environ:
        v = max(1, int(os.environ["MAX_PARALLEL_CPU"]))
    if max_parallel_cli is not None:
        v = max(1, int(max_parallel_cli))
    return v


def _get_max_parallel_cpu() -> int:
    return _resolve_max_parallel_cpu_cli(None)

# 全局配置（Step 3/4/5 训练与推理）
# 代码默认已收敛至 BASE_TRAINING_DEFAULTS；本段仅说明语义与覆盖方式。
#
# train_batch_size / epochs / gradient_accumulation_steps：见文件顶部别名与 get_*（CLI > ENV > 命名预设 > BASE）。
#   G = per_device_micro_batch × world_size × gradient_accumulation_steps（P 即 DataLoader batch_size）。
# eval 全局 batch：get_eval_batch_size() 运行时读 EVAL_BATCH_SIZE，未设则用 BASE_TRAINING_DEFAULTS.eval_batch_size。
# 早停与 BLEU 采样默认值：BASE_TRAINING_DEFAULTS.train_min_epochs / train_early_stop_patience / train_bleu4_max_samples。

_TASK_CONFIGS_BUILTIN: Dict[int, TaskConfig] = {
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

task_configs: Dict[int, TaskConfig] = _TASK_CONFIGS_BUILTIN
_LOADED_TASK_YAML = _try_load_task_defaults_from_yaml()
if _LOADED_TASK_YAML is not None:
    task_configs = _LOADED_TASK_YAML

TASK_DEFAULTS: Dict[int, TaskConfig] = task_configs


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
# 训练入口请使用 build_resolved_training_config（含 CLI）。下列 shell 辅助函数仅无 CLI 时读取 ENV/预设：
#   - get_train_batch_size(task_idx) / get_epochs(task_idx)
#
# task_idx 来源：函数实参，或环境变量 D4C_PRESET_TASK_ID（shell 的 run_step3_optimized.sh 等每任务会设置）。
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


# step3 / step5：1..8 每任务单独一份完整 dict（手写展开，无推导式），便于后续逐 task 微调。
# 二者数值相同；step5 为各行独立副本，避免运行时误改一处影响另一预设。
# train_batch_size/epochs/full_eval_every_epochs/min_lr_ratio/adv 各任务相同；lr/coef 与对应 task_configs 对齐。
_TRAINING_PRESET_STEP3_TABLE: Dict[int, Any] = {
    1: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 5e-4,
        "coef": 1,
    },
    2: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.1,
    },
    3: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 5e-4,
        "coef": 0.5,
    },
    4: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.5,
    },
    5: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.5,
    },
    6: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.5,
    },
    7: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-4,
        "coef": 0.5,
    },
    8: {
        "train_batch_size": 1024,
        "epochs": 30,
        "full_eval_every_epochs": 2,
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 5e-4,
        "coef": 1,
    },
}

_TRAINING_PRESETS_BUILTIN: Dict[str, Any] = {
    "step3": _TRAINING_PRESET_STEP3_TABLE,
    "step5": {tid: dict(row) for tid, row in _TRAINING_PRESET_STEP3_TABLE.items()},
}

TRAINING_PRESETS: Dict[str, Any] = _TRAINING_PRESETS_BUILTIN
_LOADED_TRAINING_YAML = _try_load_training_presets_from_yaml()
if _LOADED_TRAINING_YAML is not None:
    TRAINING_PRESETS = _LOADED_TRAINING_YAML

_validate_training_presets(TRAINING_PRESETS)

# ---------------------------------------------------------------------------
# 命名 runtime 预设（可选）
#
# 使用：export D4C_RUNTIME_PRESET=<键名>
# 解析优先级（与训练预设平行）：runtime_base → runtime_preset → ENV → CLI（仅 build_resolved 中带 CLI）
# 不设置 D4C_RUNTIME_PRESET 时行为与改造前一致（仍走 base + MAX_PARALLEL_CPU 等 ENV）。
#
# 线程类环境变量（OMP_NUM_THREADS / MKL_NUM_THREADS / TOKENIZERS_PARALLELISM）请在 shell 中设置；
# 本模块不在 import 时覆盖它们。
# ---------------------------------------------------------------------------

_RUNTIME_PRESET_ALLOWED_KEYS: FrozenSet[str] = frozenset(
    {
        "max_parallel_cpu",
        "num_proc",
        "ddp_world_size",
        "dataloader_num_workers_train",
        "dataloader_num_workers_valid",
        "dataloader_num_workers_test",
        "dataloader_prefetch_factor_train",
        "dataloader_prefetch_factor_valid",
        "dataloader_prefetch_factor_test",
        "dataloader_workers_train_per_rank_cap",
    }
)

_RUNTIME_PRESET_INT_KEYS: FrozenSet[str] = _RUNTIME_PRESET_ALLOWED_KEYS


def _validate_runtime_presets(presets: Dict[str, Any], *, name: str = "RUNTIME_PRESETS") -> None:
    for preset_name, blob in presets.items():
        if not isinstance(blob, dict):
            raise TypeError(f'{name} 中预设 {preset_name!r} 应为 dict，当前为 {type(blob).__name__}')
        unknown = set(blob.keys()) - _RUNTIME_PRESET_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{name} 预设 {preset_name!r} 含有未知字段 {sorted(unknown)}；"
                f"允许: {sorted(_RUNTIME_PRESET_ALLOWED_KEYS)}"
            )
        for k, v in blob.items():
            if k in _RUNTIME_PRESET_INT_KEYS:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"{name} 预设 {preset_name!r} 字段 {k!r} 应为整数类型，当前为 {type(v).__name__}")
                if isinstance(v, float) and not float(v).is_integer():
                    raise ValueError(f"{name} 预设 {preset_name!r} 字段 {k!r} 应为整数值，当前为 {v!r}")
                iv = int(v)
                if iv < 0 and k != "num_proc":  # num_proc 必须 >=1，单独在值上约束
                    raise ValueError(f"{name} 预设 {preset_name!r} 字段 {k!r} 不可为负，当前为 {iv}")
                if k == "num_proc" and iv < 1:
                    raise ValueError(f"{name} 预设 {preset_name!r} num_proc 须 >= 1，当前为 {iv}")
                if k == "ddp_world_size" and iv < 1:
                    raise ValueError(f"{name} 预设 {preset_name!r} ddp_world_size 须 >= 1，当前为 {iv}")


_RUNTIME_PRESETS_BUILTIN: Dict[str, Dict[str, int]] = {
    "gpu01_single_12c": {
        "max_parallel_cpu": 10,
        "num_proc": 6,
        "dataloader_num_workers_train": 4,
        "dataloader_num_workers_valid": 2,
        "dataloader_num_workers_test": 2,
        "dataloader_prefetch_factor_train": 2,
        "dataloader_prefetch_factor_valid": 2,
        "dataloader_prefetch_factor_test": 2,
    },
    "gpu01_ddp2_12c": {
        "max_parallel_cpu": 10,
        "num_proc": 4,
        "dataloader_workers_train_per_rank_cap": 3,
        "dataloader_num_workers_valid": 2,
        "dataloader_num_workers_test": 2,
        "dataloader_prefetch_factor_train": 2,
        "dataloader_prefetch_factor_valid": 2,
        "dataloader_prefetch_factor_test": 2,
    },
}

RUNTIME_PRESETS: Dict[str, Dict[str, int]] = _RUNTIME_PRESETS_BUILTIN
_LOADED_RUNTIME_YAML = _try_load_runtime_presets_from_yaml()
if _LOADED_RUNTIME_YAML is not None:
    RUNTIME_PRESETS = _LOADED_RUNTIME_YAML

_validate_runtime_presets(RUNTIME_PRESETS)

_unknown_runtime_preset_warned: Set[str] = set()


def _named_runtime_preset_blob() -> Optional[Dict[str, Any]]:
    name = (os.environ.get("D4C_RUNTIME_PRESET") or "").strip()
    if not name:
        return None
    raw = RUNTIME_PRESETS.get(name)
    if raw is None:
        import warnings

        if name not in _unknown_runtime_preset_warned:
            _unknown_runtime_preset_warned.add(name)
            warnings.warn(
                f"未知 D4C_RUNTIME_PRESET={name!r}，忽略 runtime 预设（按未设置 preset 继续）。",
                UserWarning,
                stacklevel=2,
            )
        return None
    return raw if isinstance(raw, dict) else None


def _active_runtime_preset_slice() -> Optional[Dict[str, Any]]:
    """当前激活的 runtime 预设 dict（全局单条）；无预设或未识别名字时为 None。"""
    return _named_runtime_preset_blob()


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
    """仅返回 TASK_DEFAULTS 表项（不含预设合并）；训练时以 build_resolved_training_config 为准。"""
    return TASK_DEFAULTS.get(int(task_idx))


def resolve_ddp_train_microbatch_layout(
    global_batch_size: int,
    world_size: int,
    *,
    per_device_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
) -> Tuple[int, int, int]:
    """由全局 batch G 解析 (G, P, A)，满足 G = P × world_size × A；梯度累积须由 resolve 层显式传入。"""
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

    if not explicit_accum_arg:
        raise ValueError("未指定 per_device_batch_size 时必须提供 gradient_accumulation_steps")
    A = max(1, int(gradient_accumulation_steps))
    denom = W * A
    if G % denom != 0:
        raise ValueError(
            f"全局 batch ({G}) 须能被 world_size×gradient_accumulation_steps ({denom}) 整除；"
            f"当前 world_size={W}，accum={A}。"
        )
    P = G // denom
    return G, P, A


def get_embed_batch_size():
    """返回 embedding 计算的 batch_size，供 run_preprocess_and_embed / compute_embeddings 使用"""
    return embed_batch_size


def get_eval_batch_size(cli: Optional[int] = None) -> int:
    """eval 全局 batch；优先级：cli > EVAL_BATCH_SIZE > BASE_TRAINING_DEFAULTS.eval_batch_size。"""
    if cli is not None:
        return max(1, int(cli))
    return max(
        1,
        int(
            os.environ.get(
                "EVAL_BATCH_SIZE",
                str(BASE_TRAINING_DEFAULTS.eval_batch_size),
            )
        ),
    )


def _first_env_int_max1(names: Tuple[str, ...]) -> Optional[int]:
    for n in names:
        if n in os.environ:
            return max(1, int(os.environ[n]))
    return None


def _fixed_dataloader_prefetch_factor(num_workers: int) -> Optional[int]:
    if num_workers <= 0:
        return None
    return max(2, 4)


def _auto_derive_dataloader_num_workers(split: str, cap_parallel: int) -> int:
    """无 runtime preset / 无对应 ENV 时的 DataLoader workers 推导（与历史逻辑一致）。"""
    _cap = int(cap_parallel)
    n = _get_num_cpu()
    cap_t = min(_cap, 16)
    cap_v = min(max(4, _cap // 2), 8)
    if split == "train":
        return min(max(2, n // 2), cap_t)
    if split in ("valid", "test"):
        return min(max(1, n // 4), cap_v)
    return min(max(1, n // 4), cap_v)


def _dataloader_workers_env_key(split: str) -> Optional[str]:
    if split == "train":
        return "D4C_DATALOADER_WORKERS_TRAIN"
    if split == "valid":
        return "D4C_DATALOADER_WORKERS_VALID"
    if split == "test":
        return "D4C_DATALOADER_WORKERS_TEST"
    return None


def _dataloader_prefetch_env_key(split: Optional[str]) -> Optional[str]:
    if split == "train":
        return "D4C_PREFETCH_TRAIN"
    if split in ("valid", "eval"):
        return "D4C_PREFETCH_VALID"
    if split == "test":
        return "D4C_PREFETCH_TEST"
    return None


def _resolve_dataloader_num_workers_for_split(split: str, cli: Optional[int]) -> int:
    """runtime 自动推导 → runtime_preset → ENV → 可选 CLI。"""
    mp = _resolve_max_parallel_cpu_cli(None)
    nw = _auto_derive_dataloader_num_workers(split, mp)
    rp = _active_runtime_preset_slice()
    wkey = f"dataloader_num_workers_{split}" if split in ("train", "valid", "test") else None
    if rp and wkey and wkey in rp:
        nw = max(0, int(rp[wkey]))
    ek = _dataloader_workers_env_key(split)
    if ek and ek in os.environ:
        nw = max(0, int(os.environ[ek]))
    if cli is not None:
        nw = max(0, int(cli))
    return nw


def _resolve_num_proc_cli(num_proc_cli: Optional[int]) -> int:
    """derived(min(cpu,max_parallel)) → runtime_preset → D4C_NUM_PROC → CLI。"""
    mp = _resolve_max_parallel_cpu_cli(None)
    v = min(_get_num_cpu(), mp)
    rp = _active_runtime_preset_slice()
    if rp and "num_proc" in rp:
        v = max(1, int(rp["num_proc"]))
        v = min(v, _get_num_cpu())
    if "D4C_NUM_PROC" in os.environ:
        v = max(1, int(os.environ["D4C_NUM_PROC"]))
        v = min(v, _get_num_cpu())
    if num_proc_cli is not None:
        v = max(1, int(num_proc_cli))
        v = min(v, _get_num_cpu())
    return v


def _resolve_ddp_train_num_workers_per_rank_cli(world_size: int, cli_cap: Optional[int]) -> int:
    ws = max(int(world_size), 1)
    dl_train = _resolve_dataloader_num_workers_for_split("train", None)
    share = max(1, _resolve_max_parallel_cpu_cli(None) // ws)
    out = max(1, min(dl_train, share))
    rp = _active_runtime_preset_slice()
    if rp and "dataloader_workers_train_per_rank_cap" in rp:
        cap = max(1, int(rp["dataloader_workers_train_per_rank_cap"]))
        out = max(1, min(out, cap))
    if "D4C_DATALOADER_TRAIN_PER_RANK_CAP" in os.environ:
        cap = max(1, int(os.environ["D4C_DATALOADER_TRAIN_PER_RANK_CAP"]))
        out = max(1, min(out, cap))
    if cli_cap is not None:
        cap = max(1, int(cli_cap))
        out = max(1, min(out, cap))
    return out


def get_num_proc() -> int:
    """datasets.map（Tokenize）并行进程数；与 DataLoader num_workers 独立。runtime_preset / D4C_NUM_PROC 可覆盖。"""
    return _resolve_num_proc_cli(None)


def get_max_parallel_cpu() -> int:
    """并行 CPU 上限；runtime_preset → MAX_PARALLEL_CPU。"""
    return _get_max_parallel_cpu()


def get_dataloader_num_workers(split="train"):
    """
    PyTorch DataLoader 的 num_workers，与 datasets.map 的 num_proc 独立。
    split: 'train' | 'valid' | 'test'。
    """
    return _resolve_dataloader_num_workers_for_split(str(split), None)


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


def get_dataloader_prefetch_factor(num_workers: int, split: Optional[str] = None):
    """
    num_workers==0 时为 None。
    若给定 split 且 runtime preset（或 D4C_PREFETCH_*）有值则用之；否则与历史一致为 prefetch=4。
    """
    if num_workers <= 0:
        return None
    rp = _active_runtime_preset_slice()
    if split == "train" and rp and "dataloader_prefetch_factor_train" in rp:
        return max(2, int(rp["dataloader_prefetch_factor_train"]))
    if split in ("valid", "eval") and rp and "dataloader_prefetch_factor_valid" in rp:
        return max(2, int(rp["dataloader_prefetch_factor_valid"]))
    if split == "test" and rp and "dataloader_prefetch_factor_test" in rp:
        return max(2, int(rp["dataloader_prefetch_factor_test"]))
    ek = _dataloader_prefetch_env_key(split)
    if ek and ek in os.environ:
        return max(2, int(os.environ[ek]))
    return _fixed_dataloader_prefetch_factor(num_workers)


def get_ddp_train_num_workers_per_rank(world_size: int) -> int:
    """
    DDP 下每个训练进程的 DataLoader worker 数。
    world_size × workers 不超过 max_parallel_cpu 均分份额；可用 dataloader_workers_train_per_rank_cap 收紧。
    """
    return _resolve_ddp_train_num_workers_per_rank_cli(world_size, None)


def _runtime_cli_val(
    runtime_overrides: Optional[Dict[str, Any]],
    args: Any,
    key: str,
) -> Any:
    """优先使用入口收集的 runtime_overrides（与 args 同源、非 None 字段），等价于显式 override dict。"""
    if runtime_overrides is not None and key in runtime_overrides:
        return runtime_overrides[key]
    return getattr(args, key, None)


@dataclass(frozen=True)
class FinalTrainingConfig:
    """入口 resolve 之后的冻结训练配置；训练循环只读此对象。"""

    task_idx: int
    auxiliary: str
    target: str
    preset_name: Optional[str]
    world_size: int
    sources: Tuple[Tuple[str, str], ...]

    learning_rate: float
    scheduler_initial_lr: float
    initial_lr: float
    epochs: int

    train_batch_size: int
    batch_size_global: int
    batch_size: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int

    num_proc: int
    max_parallel_cpu: int
    runtime_preset_name: Optional[str]
    dataloader_num_workers_train: int
    dataloader_num_workers_valid: int
    dataloader_num_workers_test: int
    dataloader_prefetch_factor_train: Optional[int]
    dataloader_prefetch_factor_valid: Optional[int]
    dataloader_prefetch_factor_test: Optional[int]

    min_lr_ratio: float
    lr_scheduler: str
    scheduler_type: str
    warmup_epochs: float
    d4c_warmup_steps: Optional[int]
    d4c_warmup_ratio: Optional[float]

    eval_batch_size: int
    min_epochs: int
    train_min_epochs: int
    early_stop_patience: int
    early_stop_patience_full: int
    early_stop_patience_loss: int

    full_eval_every_epochs: int
    full_bleu_eval_every_epochs: int
    full_eval_phased: Optional[Tuple[int, int, int]]

    checkpoint_metric: str
    dual_bleu_eval: bool
    bleu4_max_samples: int
    quick_eval_max_samples: int

    coef: float
    eta: float
    adversarial_coef: float
    adversarial_alpha: float
    adversarial_beta: float
    adversarial_schedule_enabled: bool
    adversarial_start_epoch: int
    adversarial_warmup_epochs: int
    adversarial_coef_target: float

    emsize: int = 768
    nlayers: int = 2
    nhid: int = 2048
    # 词表大小在 run-d4c 入口用 len(tokenizer) 覆盖；此处仅作占位默认值
    ntoken: int = 32128
    dropout: float = 0.2
    nhead: int = 2
    label_smoothing: float = 0.1
    repetition_penalty: float = 1.15
    generate_temperature: float = 0.8
    generate_top_p: float = 0.9
    max_explanation_length: int = 25
    decode_strategy: str = "greedy"
    decode_seed: Optional[int] = None
    nuser: int = 0
    nitem: int = 0

    device: int = 0
    device_ids: Tuple[int, ...] = ()
    save_file: str = ""
    log_file: Optional[str] = None
    ddp_world_size: int = 1
    ddp_find_unused_parameters: bool = True
    rank0_only_logging: bool = True
    run_id: str = ""
    ddp_fast_backends: bool = False

    logger: Any = None
    valid_dataset: Any = None

    def to_log_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("logger", None)
        d.pop("valid_dataset", None)
        d["sources"] = dict(self.sources)
        return d


def build_resolved_training_config(
    args: Any,
    *,
    task_idx: int,
    world_size: int,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> FinalTrainingConfig:
    """
    唯一训练配置解析入口：base → TASK_DEFAULTS → 命名预设 → ENV → CLI，一次性冻结为 FinalTrainingConfig。
    """
    ro = runtime_overrides if runtime_overrides is not None else collect_training_runtime_overrides_from_args(args)
    src: Dict[str, str] = {}

    def rv(key: str) -> Any:
        return _runtime_cli_val(ro, args, key)

    tid = int(task_idx)
    tc = TASK_DEFAULTS.get(tid)
    if tc is None:
        raise ValueError(f"无效 task_idx={tid}，TASK_DEFAULTS 中无此任务")
    auxiliary = str(tc["auxiliary"])
    target = str(tc["target"])
    row = _active_train_preset_slice(tid)
    preset_nm = (os.environ.get("D4C_TRAIN_PRESET") or "").strip() or None

    # ----- global train batch G -----
    G = int(BASE_TRAINING_DEFAULTS.train_batch_size)
    src["train_batch_size"] = "base"
    if row and "train_batch_size" in row:
        G = _preset_int_min(row["train_batch_size"], 1)
        src["train_batch_size"] = "preset"
    if "D4C_TRAIN_BATCH_SIZE" in os.environ:
        G = _preset_int_min(os.environ["D4C_TRAIN_BATCH_SIZE"], 1)
        src["train_batch_size"] = "env"
    elif "D4C_OPT_BATCH_SIZE" in os.environ:
        G = _preset_int_min(os.environ["D4C_OPT_BATCH_SIZE"], 1)
        src["train_batch_size"] = "env"
    if rv("batch_size") is not None:
        G = int(rv("batch_size"))
        src["train_batch_size"] = "cli"

    # ----- per-device P -----
    p_opt: Optional[int] = None
    src["per_device_train_batch_size"] = "base"
    if row and "per_device_train_batch_size" in row:
        p_opt = max(1, int(row["per_device_train_batch_size"]))
        src["per_device_train_batch_size"] = "preset"
    if "D4C_PER_DEVICE_BATCH_SIZE" in os.environ:
        p_opt = max(1, int(os.environ["D4C_PER_DEVICE_BATCH_SIZE"]))
        src["per_device_train_batch_size"] = "env"
    if rv("per_device_batch_size") is not None:
        p_opt = max(1, int(rv("per_device_batch_size")))
        src["per_device_train_batch_size"] = "cli"

    # ----- gradient accumulation (merged before layout) -----
    A0 = int(BASE_TRAINING_DEFAULTS.gradient_accumulation_steps)
    src["gradient_accumulation_steps"] = "base"
    if row and "gradient_accumulation_steps" in row:
        A0 = _preset_int_min(row["gradient_accumulation_steps"], 1)
        src["gradient_accumulation_steps"] = "preset"
    if "D4C_GRADIENT_ACCUMULATION_STEPS" in os.environ:
        A0 = _preset_int_min(os.environ["D4C_GRADIENT_ACCUMULATION_STEPS"], 1)
        src["gradient_accumulation_steps"] = "env"
    a_cli_raw = rv("gradient_accumulation_steps")
    a_cli = int(a_cli_raw) if a_cli_raw is not None else None
    if a_cli is not None:
        A0 = max(1, int(a_cli))
        src["gradient_accumulation_steps"] = "cli"

    if p_opt is not None:
        accum_for_layout: Optional[int] = a_cli
    else:
        accum_for_layout = A0
    G, P, A = resolve_ddp_train_microbatch_layout(
        G,
        world_size,
        per_device_batch_size=p_opt,
        gradient_accumulation_steps=accum_for_layout,
    )
    eff = P * int(world_size) * A
    if eff != G:
        raise RuntimeError(f"内部错误: 期望 G={G} 与 P×W×A={eff} 一致")

    # ----- epochs -----
    ep = int(BASE_TRAINING_DEFAULTS.epochs)
    src["epochs"] = "base"
    if row and "epochs" in row:
        ep = _preset_int_min(row["epochs"], 1)
        src["epochs"] = "preset"
    if "D4C_EPOCHS" in os.environ:
        ep = _preset_int_min(os.environ["D4C_EPOCHS"], 1)
        src["epochs"] = "env"
    if rv("epochs") is not None:
        ep = int(rv("epochs"))
        src["epochs"] = "cli"

    # ----- learning rate：BASE → task → preset → ENV → CLI（scheduler 覆盖 learning_rate）-----
    initial_f = float(BASE_TRAINING_DEFAULTS.initial_learning_rate)
    src["learning_rate"] = "base"
    initial_f = float(tc["lr"])
    src["learning_rate"] = "task"
    if row and "lr" in row:
        initial_f = float(_coerce_task_param_numeric(row["lr"]))
        src["learning_rate"] = "preset"
    if "D4C_INITIAL_LR" in os.environ:
        initial_f = float(os.environ["D4C_INITIAL_LR"])
        src["learning_rate"] = "env"
    cli_lr = rv("learning_rate")
    cli_sched = rv("scheduler_initial_lr")
    if cli_lr is not None:
        initial_f = float(cli_lr)
        src["learning_rate"] = "cli"
    if cli_sched is not None:
        initial_f = float(cli_sched)
        src["learning_rate"] = "cli"

    # ----- min_lr_ratio -----
    min_lr = float(BASE_TRAINING_DEFAULTS.min_lr_ratio)
    src["min_lr_ratio"] = "base"
    if row and "min_lr_ratio" in row:
        min_lr = float(row["min_lr_ratio"])
        src["min_lr_ratio"] = "preset"
    if "D4C_MIN_LR_RATIO" in os.environ:
        min_lr = float(os.environ["D4C_MIN_LR_RATIO"])
        src["min_lr_ratio"] = "env"
    if rv("min_lr_ratio") is not None:
        min_lr = float(rv("min_lr_ratio"))
        src["min_lr_ratio"] = "cli"

    # ----- lr_scheduler -----
    lr_sched = str(BASE_TRAINING_DEFAULTS.lr_scheduler)
    src["lr_scheduler"] = "base"
    if "D4C_LR_SCHEDULER" in os.environ:
        v = os.environ["D4C_LR_SCHEDULER"].strip().lower()
        if v in ("warmup_cosine", "warmup-cosine", "cosine"):
            lr_sched = "warmup_cosine"
        elif v in ("none", "off", "disabled"):
            lr_sched = "none"
        else:
            lr_sched = "none"
        src["lr_scheduler"] = "env"
    cli_ls = rv("lr_scheduler")
    if cli_ls is not None and str(cli_ls).strip():
        v = str(cli_ls).strip().lower()
        if v in ("none", "off", "disabled"):
            lr_sched = "none"
        elif v in ("warmup_cosine", "warmup-cosine", "cosine"):
            lr_sched = "warmup_cosine"
        else:
            lr_sched = "none"
        src["lr_scheduler"] = "cli"

    # ----- warmup_epochs -----
    wu_ep = float(BASE_TRAINING_DEFAULTS.warmup_epochs)
    src["warmup_epochs"] = "base"
    if "D4C_WARMUP_EPOCHS" in os.environ:
        wu_ep = float(os.environ["D4C_WARMUP_EPOCHS"])
        src["warmup_epochs"] = "env"
    if rv("warmup_epochs") is not None:
        wu_ep = float(rv("warmup_epochs"))
        src["warmup_epochs"] = "cli"

    # ----- warmup steps / ratio -----
    wsteps: Optional[int] = None
    src["d4c_warmup_steps"] = "base"
    if "D4C_WARMUP_STEPS" in os.environ:
        v = int(os.environ["D4C_WARMUP_STEPS"])
        wsteps = v if v > 0 else None
        src["d4c_warmup_steps"] = "env"
    if rv("warmup_steps") is not None:
        v = int(rv("warmup_steps"))
        wsteps = v if v > 0 else None
        src["d4c_warmup_steps"] = "cli"

    wratio: Optional[float] = None
    src["d4c_warmup_ratio"] = "base"
    if "D4C_WARMUP_RATIO" in os.environ:
        wratio = float(os.environ["D4C_WARMUP_RATIO"])
        src["d4c_warmup_ratio"] = "env"
    if rv("warmup_ratio") is not None:
        wratio = float(rv("warmup_ratio"))
        src["d4c_warmup_ratio"] = "cli"

    # ----- eval batch -----
    eval_bs = int(BASE_TRAINING_DEFAULTS.eval_batch_size)
    src["eval_batch_size"] = "base"
    if "EVAL_BATCH_SIZE" in os.environ:
        eval_bs = max(1, int(os.environ["EVAL_BATCH_SIZE"]))
        src["eval_batch_size"] = "env"
    if rv("eval_batch_size") is not None:
        eval_bs = max(1, int(rv("eval_batch_size")))
        src["eval_batch_size"] = "cli"

    # ----- early stop -----
    min_ep = max(1, int(BASE_TRAINING_DEFAULTS.train_min_epochs))
    src["min_epochs"] = "base"
    if "TRAIN_MIN_EPOCHS" in os.environ:
        min_ep = max(1, int(os.environ["TRAIN_MIN_EPOCHS"]))
        src["min_epochs"] = "env"
    if rv("min_epochs") is not None:
        min_ep = max(1, int(rv("min_epochs")))
        src["min_epochs"] = "cli"

    esp = max(1, int(BASE_TRAINING_DEFAULTS.train_early_stop_patience))
    src["early_stop_patience"] = "base"
    if "TRAIN_EARLY_STOP_PATIENCE" in os.environ:
        esp = max(1, int(os.environ["TRAIN_EARLY_STOP_PATIENCE"]))
        src["early_stop_patience"] = "env"
    _esp = rv("early_stop_patience")
    if _esp is not None:
        esp = max(1, int(_esp))
        src["early_stop_patience"] = "cli"

    esp_full_raw = rv("early_stop_patience_full")
    if esp_full_raw is not None:
        esp_full = max(1, int(esp_full_raw))
        src["early_stop_patience_full"] = "cli"
    elif _esp is not None:
        esp_full = max(1, int(_esp))
        src["early_stop_patience_full"] = "cli"
    else:
        v = _first_env_int_max1(("TRAIN_EARLY_STOP_PATIENCE_FULL", "D4C_EARLY_STOP_PATIENCE_FULL"))
        if v is not None:
            esp_full = v
            src["early_stop_patience_full"] = "env"
        else:
            esp_full = esp
            src["early_stop_patience_full"] = src["early_stop_patience"]

    if rv("early_stop_patience_loss") is not None:
        esp_loss = max(1, int(rv("early_stop_patience_loss")))
        src["early_stop_patience_loss"] = "cli"
    else:
        v = _first_env_int_max1(("TRAIN_EARLY_STOP_PATIENCE_LOSS", "D4C_EARLY_STOP_PATIENCE_LOSS"))
        if v is not None:
            esp_loss = v
            src["early_stop_patience_loss"] = "env"
        elif _esp is not None:
            esp_loss = max(1, int(_esp))
            src["early_stop_patience_loss"] = "cli"
        else:
            esp_loss = esp
            src["early_stop_patience_loss"] = src["early_stop_patience"]

    # ----- BLEU samples -----
    b4 = max(64, int(BASE_TRAINING_DEFAULTS.train_bleu4_max_samples))
    src["bleu4_max_samples"] = "base"
    if "TRAIN_BLEU4_MAX_SAMPLES" in os.environ:
        b4 = max(64, int(os.environ["TRAIN_BLEU4_MAX_SAMPLES"]))
        src["bleu4_max_samples"] = "env"
    if rv("bleu4_max_samples") is not None:
        b4 = max(64, int(rv("bleu4_max_samples")))
        src["bleu4_max_samples"] = "cli"

    qeval = b4
    src["quick_eval_max_samples"] = "base"
    if "D4C_QUICK_EVAL_MAX_SAMPLES" in os.environ:
        qeval = max(64, int(os.environ["D4C_QUICK_EVAL_MAX_SAMPLES"]))
        src["quick_eval_max_samples"] = "env"
    if rv("quick_eval_max_samples") is not None:
        qeval = max(64, int(rv("quick_eval_max_samples")))
        src["quick_eval_max_samples"] = "cli"

    # ----- full BLEU eval schedule -----
    fe = 0
    phased: Optional[Tuple[int, int, int]] = None
    src["full_eval_schedule"] = "base"
    cli_fe = rv("full_eval_every")
    if cli_fe is not None:
        fe = max(0, int(cli_fe))
        phased = None
        src["full_eval_schedule"] = "cli"
    elif "D4C_FULL_EVAL_EVERY" in os.environ:
        fe = max(0, int(os.environ["D4C_FULL_EVAL_EVERY"]))
        phased = None
        src["full_eval_schedule"] = "env"
    elif "D4C_FULL_BLEU_EVAL_EVERY" in os.environ:
        fe = max(0, int(os.environ["D4C_FULL_BLEU_EVAL_EVERY"]))
        phased = None
        src["full_eval_schedule"] = "env"
    elif row and "full_eval_every_epochs" in row:
        fe = max(0, int(row["full_eval_every_epochs"]))
        phased = None
        src["full_eval_schedule"] = "preset"
    else:
        early = max(1, int(os.environ.get("D4C_FULL_EVAL_EARLY_EVERY", "5")))
        boundary = max(1, int(os.environ.get("D4C_FULL_EVAL_PHASE_END_EPOCH", "10")))
        late = max(1, int(os.environ.get("D4C_FULL_EVAL_LATE_EVERY", "2")))
        phased = (early, boundary, late)
        src["full_eval_schedule"] = "env"

    ckpt_metric = str(getattr(args, "checkpoint_metric", "bleu4"))
    dual = ckpt_metric == "bleu4" and (fe > 0 or phased is not None)

    # ----- coef / adversarial_coef -----
    coef_f = float(_coerce_task_param_numeric(tc["coef"]))
    src["coef"] = "task"
    if row and "coef" in row:
        coef_f = float(_coerce_task_param_numeric(row["coef"]))
        src["coef"] = "preset"
    if rv("coef") is not None:
        coef_f = float(rv("coef"))
        src["coef"] = "cli"

    adv_f = float(_coerce_task_param_numeric(tc["adv"]))
    src["adversarial_coef"] = "task"
    if row and "adv" in row:
        adv_f = float(_coerce_task_param_numeric(row["adv"]))
        src["adversarial_coef"] = "preset"
    if rv("adv") is not None:
        adv_f = float(rv("adv"))
        src["adversarial_coef"] = "cli"

    # ----- eta (run-d4c) -----
    eta_f = 1e-3
    src["eta"] = "base"
    eta_cli = getattr(args, "eta", None)
    if eta_cli is not None:
        eta_f = float(eta_cli)
        src["eta"] = "cli"

    # ----- adversarial schedule (仅 resolve 读 ENV，训练期不再读) -----
    start = rv("adversarial_start_epoch")
    src["adversarial_schedule"] = "base"
    if start is None and "D4C_ADVERSARIAL_START_EPOCH" in os.environ:
        start = int(os.environ["D4C_ADVERSARIAL_START_EPOCH"])
        src["adversarial_schedule"] = "env"
    if start is None:
        adv_sched_on = False
        adv_skip = 0
        adv_warm = 0
        adv_target = adv_f
        src["adversarial_schedule_enabled"] = "base"
    else:
        adv_sched_on = True
        adv_skip = max(0, int(start))
        w = rv("adversarial_warmup_epochs")
        if w is None and "D4C_ADVERSARIAL_WARMUP_EPOCHS" in os.environ:
            w = int(os.environ["D4C_ADVERSARIAL_WARMUP_EPOCHS"])
        if w is None:
            adv_warm = 0
        else:
            adv_warm = max(0, int(w))
        t = rv("adversarial_coef_target")
        if t is None and "D4C_ADVERSARIAL_COEF_TARGET" in os.environ:
            t = float(os.environ["D4C_ADVERSARIAL_COEF_TARGET"])
        if t is None:
            adv_target = adv_f
        else:
            adv_target = float(t)
        src["adversarial_schedule_enabled"] = "cli" if rv("adversarial_start_epoch") is not None else "env"

    # ----- runtime preset 元数据（D4C_RUNTIME_PRESET 原样记录；未知名时 blob 已为 None，行为等同未设 preset）-----
    runtime_preset_nm = (os.environ.get("D4C_RUNTIME_PRESET") or "").strip() or None

    # ----- max_parallel_cpu（runtime_base → runtime_preset → MAX_PARALLEL_CPU）-----
    max_par_v = max(1, int(_RUNTIME_BASE_MAX_PARALLEL_CPU))
    src["max_parallel_cpu"] = "base"
    rp_rt = _active_runtime_preset_slice()
    if rp_rt and "max_parallel_cpu" in rp_rt:
        max_par_v = max(1, int(rp_rt["max_parallel_cpu"]))
        src["max_parallel_cpu"] = "runtime_preset"
    if "MAX_PARALLEL_CPU" in os.environ:
        max_par_v = max(1, int(os.environ["MAX_PARALLEL_CPU"]))
        src["max_parallel_cpu"] = "env"

    # ----- num_proc（derived → runtime_preset → D4C_NUM_PROC → CLI）-----
    num_proc_v = min(_get_num_cpu(), max_par_v)
    src["num_proc"] = "derived"
    if rp_rt and "num_proc" in rp_rt:
        num_proc_v = max(1, int(rp_rt["num_proc"]))
        num_proc_v = min(num_proc_v, _get_num_cpu())
        src["num_proc"] = "runtime_preset"
    if "D4C_NUM_PROC" in os.environ:
        num_proc_v = max(1, int(os.environ["D4C_NUM_PROC"]))
        num_proc_v = min(num_proc_v, _get_num_cpu())
        src["num_proc"] = "env"
    if rv("num_proc") is not None:
        num_proc_v = max(1, int(rv("num_proc")))
        num_proc_v = min(num_proc_v, _get_num_cpu())
        src["num_proc"] = "cli"

    ws = max(int(world_size), 1)
    nw_train = _resolve_ddp_train_num_workers_per_rank_cli(ws, None)
    if rp_rt and (
        "dataloader_num_workers_train" in rp_rt or "dataloader_workers_train_per_rank_cap" in rp_rt
    ):
        src["dataloader_num_workers_train"] = "runtime_preset"
    elif "D4C_DATALOADER_WORKERS_TRAIN" in os.environ or "D4C_DATALOADER_TRAIN_PER_RANK_CAP" in os.environ:
        src["dataloader_num_workers_train"] = "env"
    else:
        src["dataloader_num_workers_train"] = "derived"

    dl_valid_base = _resolve_dataloader_num_workers_for_split("valid", None)
    nw_valid = max(1, min(dl_valid_base, nw_train))
    if rp_rt and "dataloader_num_workers_valid" in rp_rt:
        src["dataloader_num_workers_valid"] = "runtime_preset"
    elif "D4C_DATALOADER_WORKERS_VALID" in os.environ:
        src["dataloader_num_workers_valid"] = "env"
    else:
        src["dataloader_num_workers_valid"] = "derived"

    nw_test = _resolve_dataloader_num_workers_for_split("test", None)
    if rp_rt and "dataloader_num_workers_test" in rp_rt:
        src["dataloader_num_workers_test"] = "runtime_preset"
    elif "D4C_DATALOADER_WORKERS_TEST" in os.environ:
        src["dataloader_num_workers_test"] = "env"
    else:
        src["dataloader_num_workers_test"] = "derived"

    pf_t = get_dataloader_prefetch_factor(nw_train, split="train")
    pf_v = get_dataloader_prefetch_factor(nw_valid, split="valid")
    pf_test = get_dataloader_prefetch_factor(nw_test, split="test")
    src["dataloader_prefetch_factor_train"] = (
        "runtime_preset"
        if rp_rt and "dataloader_prefetch_factor_train" in rp_rt
        else ("env" if "D4C_PREFETCH_TRAIN" in os.environ else "derived")
    )
    src["dataloader_prefetch_factor_valid"] = (
        "runtime_preset"
        if rp_rt and "dataloader_prefetch_factor_valid" in rp_rt
        else ("env" if "D4C_PREFETCH_VALID" in os.environ else "derived")
    )
    src["dataloader_prefetch_factor_test"] = (
        "runtime_preset"
        if rp_rt and "dataloader_prefetch_factor_test" in rp_rt
        else ("env" if "D4C_PREFETCH_TEST" in os.environ else "derived")
    )

    sources_tuple = tuple(sorted(src.items()))

    return FinalTrainingConfig(
        task_idx=tid,
        auxiliary=auxiliary,
        target=target,
        preset_name=preset_nm,
        world_size=int(world_size),
        sources=sources_tuple,
        learning_rate=initial_f,
        scheduler_initial_lr=initial_f,
        initial_lr=initial_f,
        epochs=ep,
        train_batch_size=G,
        batch_size_global=G,
        batch_size=P,
        per_device_train_batch_size=P,
        gradient_accumulation_steps=A,
        effective_global_batch_size=eff,
        num_proc=num_proc_v,
        max_parallel_cpu=max_par_v,
        runtime_preset_name=runtime_preset_nm,
        dataloader_num_workers_train=nw_train,
        dataloader_num_workers_valid=nw_valid,
        dataloader_num_workers_test=nw_test,
        dataloader_prefetch_factor_train=pf_t,
        dataloader_prefetch_factor_valid=pf_v,
        dataloader_prefetch_factor_test=pf_test,
        min_lr_ratio=min_lr,
        lr_scheduler=lr_sched,
        scheduler_type=lr_sched,
        warmup_epochs=wu_ep,
        d4c_warmup_steps=wsteps,
        d4c_warmup_ratio=wratio,
        eval_batch_size=eval_bs,
        min_epochs=min_ep,
        train_min_epochs=min_ep,
        early_stop_patience=esp,
        early_stop_patience_full=esp_full,
        early_stop_patience_loss=esp_loss,
        full_eval_every_epochs=fe,
        full_bleu_eval_every_epochs=fe,
        full_eval_phased=phased,
        checkpoint_metric=ckpt_metric,
        dual_bleu_eval=dual,
        bleu4_max_samples=b4,
        quick_eval_max_samples=qeval,
        coef=coef_f,
        eta=eta_f,
        adversarial_coef=adv_f,
        adversarial_alpha=adv_f,
        adversarial_beta=adv_f,
        adversarial_schedule_enabled=adv_sched_on,
        adversarial_start_epoch=adv_skip,
        adversarial_warmup_epochs=adv_warm,
        adversarial_coef_target=adv_target,
    )


def get_train_batch_size(task_idx: Optional[int] = None) -> int:
    """
    全局训练 batch G；供 shell 等**无 argparse**场景（无 CLI 层）。
    顺序：BASE → TRAINING_PRESET 切片 → ``D4C_TRAIN_BATCH_SIZE`` / ``D4C_OPT_BATCH_SIZE``。
    """
    G = int(BASE_TRAINING_DEFAULTS.train_batch_size)
    row = _active_train_preset_slice(task_idx)
    if row and "train_batch_size" in row:
        G = _preset_int_min(row["train_batch_size"], 1)
    if "D4C_TRAIN_BATCH_SIZE" in os.environ:
        G = _preset_int_min(os.environ["D4C_TRAIN_BATCH_SIZE"], 1)
    elif "D4C_OPT_BATCH_SIZE" in os.environ:
        G = _preset_int_min(os.environ["D4C_OPT_BATCH_SIZE"], 1)
    return G


def get_epochs(task_idx: Optional[int] = None) -> int:
    """
    训练轮数；供 shell 等**无 argparse**场景。
    顺序：BASE → 预设 → ``D4C_EPOCHS``。
    """
    ep = int(BASE_TRAINING_DEFAULTS.epochs)
    row = _active_train_preset_slice(task_idx)
    if row and "epochs" in row:
        ep = _preset_int_min(row["epochs"], 1)
    if "D4C_EPOCHS" in os.environ:
        ep = _preset_int_min(os.environ["D4C_EPOCHS"], 1)
    return ep


def __getattr__(name: str) -> Any:
    """兼容旧代码 ``from config import num_proc`` / ``eval_batch_size``（改为运行时解析）。"""
    if name == "num_proc":
        return get_num_proc()
    if name == "eval_batch_size":
        return get_eval_batch_size()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
