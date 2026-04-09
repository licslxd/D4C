import json
import math
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from dataclasses import asdict, dataclass, field, replace
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from cpu_utils import effective_cpu_count
from training_hardware_inputs import collect_training_hardware_overrides_from_args

# 执行层：DDP_NPROC / torchrun --nproc_per_node 仅在 shell 解析；Python 以 WORLD_SIZE 为准。
# D4C_NUM_PROC 为 CPU 侧（如 datasets.map）并行度，属 hardware 层（与 D4C_HARDWARE_PROFILE_JSON 同源语义），勿与 DDP_NPROC 混淆。详见 docs/D4C_Scripts_and_Runtime_Guide.md。

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


class HardwarePresetRow(TypedDict, total=False):
    """命名 hardware 预设（CPU / DataLoader / DDP 布局等）；与训练 TRAINING_PRESETS 独立。"""

    max_parallel_cpu: int
    num_proc: int
    ddp_world_size: int
    dataloader_num_workers_train: int
    dataloader_num_workers_valid: int
    dataloader_num_workers_test: int
    dataloader_prefetch_factor_train: int
    dataloader_prefetch_factor_valid: int
    dataloader_prefetch_factor_test: int
    dataloader_workers_train_per_rank_cap: int
    omp_num_threads: int
    mkl_num_threads: int
    tokenizers_parallelism: bool


class TrainingPresetRow(TypedDict, total=False):
    """命名预设中允许出现的字段（全局一条或 per-task 子 dict）。"""

    train_batch_size: int
    train_label_max_length: int
    epochs: int
    full_bleu_eval: Dict[str, Any]
    min_lr_ratio: float
    adv: float
    lr: float
    coef: float
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    train_dynamic_padding: bool
    loss_weight_repeat_ul: float
    loss_weight_terminal_clean: float
    terminal_clean_span: int
    full_bleu_decode_strategy: str


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
    # eval 全局 batch 的代码默认；torchrun 子进程内实际值来自父进程写入的 effective training_row（见 build_resolved_training_config）
    eval_batch_size: int = 2560
    # 无任务 lr 时 learning_rate resolve 链末级 fallback（在 build_resolved_training_config 内使用）
    initial_learning_rate: float = 1e-3
    # 训练标签（teacher forcing）最大 token 长度；与 decode 的 max_explanation_length 解耦
    train_label_max_length: int = 64


BASE_TRAINING_DEFAULTS = BaseTrainingDefaults()
DEFAULT_TRAINING_CONFIG = BASE_TRAINING_DEFAULTS
"""与 BASE_TRAINING_DEFAULTS 同义别名，便于语义上称「默认训练配置」。"""

# 模块级便捷常量：与 ``from config import train_batch_size`` / ``epochs`` 等一致（值均来自 BASE_TRAINING_DEFAULTS）
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


def _normalize_cuda_visible_devices_yaml(val: Any) -> Optional[str]:
    """hardware preset 中的 cuda_visible_devices：规范化逗号分隔设备列表；空则 None。"""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    return ",".join(parts)


def _coerce_hardware_yaml_value(key: str, val: Any) -> Any:
    """hardware YAML 标量：整数字段用 int；tokenizers_parallelism 用 bool。"""
    k = str(key)
    if k == "tokenizers_parallelism":
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ("true", "1", "yes", "on")
    if val is None:
        raise ValueError(f"hardware 字段 {k!r} 不可为 null")
    if isinstance(val, bool):
        raise TypeError(f"hardware 字段 {k!r} 为 bool，仅 tokenizers_parallelism 允许布尔类型")
    return int(val)


def _try_load_hardware_presets_from_yaml() -> Optional[Dict[str, Dict[str, Any]]]:
    try:
        import yaml  # noqa: F401
    except ImportError:
        record_preset_event("WARN presets/hardware: PyYAML not installed -> builtin HARDWARE_PRESETS")
        warnings.warn(
            "未安装 PyYAML，跳过 presets/hardware/*.yaml，使用内置 HARDWARE_PRESETS。",
            UserWarning,
            stacklevel=2,
        )
        return None
    root = _d4c_presets_repo_root() / "presets" / "hardware"
    if not root.is_dir():
        record_preset_event("SKIP presets/hardware: directory missing -> builtin HARDWARE_PRESETS")
        return None
    paths = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not paths:
        record_preset_event("SKIP presets/hardware: no yaml files -> builtin HARDWARE_PRESETS")
        return None
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        try:
            raw = _load_yaml_file(path)
            if raw is None:
                raise ValueError("文件为空或仅注释")
            if not isinstance(raw, dict):
                raise TypeError(f"根须为 mapping，当前为 {type(raw).__name__}")
            row: Dict[str, Any] = {}
            for kk, vv in raw.items():
                sk = str(kk)
                if sk == "cuda_visible_devices":
                    cvd = _normalize_cuda_visible_devices_yaml(vv)
                    if cvd:
                        row[sk] = cvd
                    continue
                row[sk] = _coerce_hardware_yaml_value(kk, vv)
            out[path.stem] = row
        except Exception as e:
            record_preset_event(f"WARN presets/hardware: load failed {path} -> builtin ({e})")
            warnings.warn(
                f"加载 hardware 预设 YAML 失败，回退内置 HARDWARE_PRESETS: {path}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return None
    record_preset_event(f"OK presets/hardware: loaded {len(paths)} preset file(s)")
    return out


@dataclass(frozen=True)
class FullBleuEvalResolved:
    """由 training_row.full_bleu_eval 解析得到的唯一 full BLEU 调度语义。"""

    mode: str  # "off" | "interval"
    every_epochs: Optional[int]
    enabled: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "every_epochs": self.every_epochs, "enabled": bool(self.enabled)}


# 键名刻意拆写，避免仓库内 grep 旧符号时与本拒绝表误报为「仍在用」
_LEGACY_FULL_BLEU_KEYS: FrozenSet[str] = frozenset(
    (
        f"{'full_eval'}_{'every_epochs'}",
        f"{'full_eval'}_{'phased'}",
        f"{'full_bleu_eval'}_{'every_epochs'}",
    )
)


def _reject_legacy_full_bleu_keys(row: Mapping[str, Any]) -> None:
    bad = sorted(k for k in _LEGACY_FULL_BLEU_KEYS if k in row)
    if bad:
        raise ValueError(
            "training_row 含已废弃字段 "
            f"{bad}；请删除并改用唯一块 full_bleu_eval: "
            "{{ mode: off | interval, every_epochs: <int>（仅 interval 且 >0） }}。"
        )


def parse_full_bleu_eval_block(block: Any, *, ctx: str = "full_bleu_eval") -> FullBleuEvalResolved:
    if not isinstance(block, dict):
        raise TypeError(f"{ctx} 须为 dict，当前为 {type(block).__name__}")
    unk = set(block.keys()) - {"mode", "every_epochs"}
    if unk:
        raise ValueError(f"{ctx} 含有未知字段 {sorted(unk)}")
    raw_mode = block.get("mode", "")
    # PyYAML 1.1 会把 off/on 解析成 bool；此处与字符串 mode 等价处理
    if isinstance(raw_mode, bool):
        if raw_mode is False:
            mode = "off"
        else:
            raise ValueError(f"{ctx}.mode 为布尔真值，非法；请使用字符串 off 或 interval")
    else:
        mode = str(raw_mode).strip().lower()
    if mode not in ("off", "interval"):
        raise ValueError(f"{ctx}.mode 须为 off 或 interval，当前为 {block.get('mode')!r}")
    if mode == "off":
        if "every_epochs" in block and block["every_epochs"] is not None:
            raise ValueError(f"{ctx}: mode=off 时不应设置 every_epochs")
        return FullBleuEvalResolved(mode="off", every_epochs=None, enabled=False)
    raw_ee = block.get("every_epochs", None)
    if raw_ee is None:
        raise ValueError(f"{ctx}: mode=interval 时必须提供 every_epochs（正整数）")
    try:
        ee = int(raw_ee)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{ctx}.every_epochs 须为整数，当前为 {raw_ee!r}") from e
    if ee <= 0:
        raise ValueError(f"{ctx}.every_epochs 须 > 0，当前为 {ee}")
    return FullBleuEvalResolved(mode="interval", every_epochs=ee, enabled=True)


def resolve_full_bleu_eval_from_training_row(row: Mapping[str, Any]) -> FullBleuEvalResolved:
    """仅从 training_row 读取 full_bleu_eval；遇旧字段或缺失块则 fail fast。"""
    _reject_legacy_full_bleu_keys(row)
    if "full_bleu_eval" not in row:
        raise ValueError(
            "training_row 缺少必填块 full_bleu_eval，例如:\n"
            "  full_bleu_eval:\n"
            "    mode: interval\n"
            "    every_epochs: 2\n"
            "或 smoke: { mode: off }"
        )
    return parse_full_bleu_eval_block(row["full_bleu_eval"], ctx="training_row.full_bleu_eval")


def should_run_full_bleu_eval_epoch(
    epoch_1_based: int,
    schedule: FullBleuEvalResolved,
) -> bool:
    """epoch_1_based 为从 1 开始的 epoch 序号。"""
    if not schedule.enabled or schedule.mode != "interval":
        return False
    ee = schedule.every_epochs
    assert ee is not None and ee > 0
    return epoch_1_based % ee == 0


def format_full_bleu_eval_resolved_log_line(schedule: FullBleuEvalResolved) -> str:
    return (
        f"[full_bleu_eval] mode={schedule.mode} every_epochs={schedule.every_epochs} "
        f"enabled={1 if schedule.enabled else 0}"
    )


def format_full_bleu_eval_epoch_decision_log_line(epoch_1_based: int, should_run: bool) -> str:
    return f"[full_bleu_eval] epoch={epoch_1_based} should_run={1 if should_run else 0}"


def parse_full_bleu_decode_strategy(v: Any, *, ctx: str = "full_bleu_decode_strategy") -> str:
    """训练期 full BLEU 监控用的解码策略：greedy=监控独立 greedy；inherit=与主 decode_strategy 一致。"""
    s = str(v).strip().lower()
    if s in ("greedy", "inherit"):
        return s
    raise ValueError(f"{ctx} 必须为 greedy 或 inherit，当前为 {v!r}")


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
# hardware 解析链的代码默认上限（无 MAX_PARALLEL_CPU 且无 D4C_HARDWARE_PRESET 覆盖时与历史默认 12 一致）
_HARDWARE_BASE_MAX_PARALLEL_CPU = 12


def _get_num_cpu() -> int:
    return int(effective_cpu_count() or 8)


def _resolve_max_parallel_cpu_cli(max_parallel_cli: Optional[int] = None) -> int:
    """hardware_base → hardware_preset → MAX_PARALLEL_CPU → 可选 CLI。"""
    v = max(1, int(_HARDWARE_BASE_MAX_PARALLEL_CPU))
    rp = _active_hardware_preset_slice()
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
# eval 全局 batch：torchrun 子进程由 build_resolved_training_config 从 effective payload 读取；get_eval_batch_size() 仅服务
# 未走 d4c torchrun 的辅脚本（CLI > EVAL_BATCH_SIZE > BASE）。
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
# 命名训练预设（TRAINING_PRESETS / presets/training/*.yaml）仅由 `python code/d4c.py --preset` 与
# load_resolved_config 合并链消费；**不再**通过 D4C_TRAIN_PRESET / D4C_PRESET_TASK_ID 环境变量二次选型。
#
# 下列 get_train_batch_size / get_epochs 仅供**无 d4c CLI** 的极薄辅助脚本（如离线工具），
# 仅使用 BASE_TRAINING_DEFAULTS 与 D4C_TRAIN_BATCH_SIZE / D4C_EPOCHS；与 TRAINING_PRESETS 解耦。
# ---------------------------------------------------------------------------

_TRAINING_PRESET_ALLOWED_KEYS: FrozenSet[str] = frozenset(
    {
        "train_batch_size",
        "train_label_max_length",
        "epochs",
        "full_bleu_eval",
        "min_lr_ratio",
        "adv",
        "lr",
        "coef",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
        "train_dynamic_padding",
        "loss_weight_repeat_ul",
        "loss_weight_terminal_clean",
        "terminal_clean_span",
        "full_bleu_decode_strategy",
    }
)

_TRAINING_PRESET_INT_KEYS: FrozenSet[str] = frozenset(
    {
        "train_batch_size",
        "train_label_max_length",
        "epochs",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
        "terminal_clean_span",
    }
)

_TRAINING_PRESET_FLOAT_KEYS: FrozenSet[str] = frozenset(
    {
        "min_lr_ratio",
        "adv",
        "lr",
        "coef",
        "loss_weight_repeat_ul",
        "loss_weight_terminal_clean",
    }
)

_TRAINING_PRESET_BOOL_KEYS: FrozenSet[str] = frozenset({"train_dynamic_padding"})


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
            if k == "full_bleu_eval":
                parse_full_bleu_eval_block(v, ctx=f"{name} {ctx} full_bleu_eval")
                continue
            if k == "full_bleu_decode_strategy":
                parse_full_bleu_decode_strategy(v, ctx=f"{name} {ctx} full_bleu_decode_strategy")
                continue
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
            elif k in _TRAINING_PRESET_BOOL_KEYS:
                if not isinstance(v, bool):
                    raise TypeError(f"{name} {ctx} 字段 {k!r} 应为 bool，当前为 {type(v).__name__}")

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
# train_batch_size/epochs/full_bleu_eval/min_lr_ratio/adv 各任务相同；lr/coef 与对应 task_configs 对齐。
_FBE_INTERVAL_2 = {"mode": "interval", "every_epochs": 2}
_TRAINING_PRESET_STEP3_TABLE: Dict[int, Any] = {
    1: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 5e-4,
        "coef": 1,
    },
    2: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.1,
    },
    3: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 5e-4,
        "coef": 0.5,
    },
    4: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.5,
    },
    5: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.5,
    },
    6: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-3,
        "coef": 0.5,
    },
    7: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 1e-4,
        "coef": 0.5,
    },
    8: {
        "train_batch_size": 1024,
        "train_label_max_length": 64,
        "epochs": 30,
        "full_bleu_eval": dict(_FBE_INTERVAL_2),
        "min_lr_ratio": 0.1,
        "adv": 0.005,
        "lr": 5e-4,
        "coef": 1,
    },
}

_TRAINING_PRESETS_BUILTIN: Dict[str, Any] = {
    "step3": _TRAINING_PRESET_STEP3_TABLE,
    "step5": {
        tid: {**dict(row), "full_bleu_eval": {"mode": "interval", "every_epochs": 3}}
        for tid, row in _TRAINING_PRESET_STEP3_TABLE.items()
    },
}

TRAINING_PRESETS: Dict[str, Any] = _TRAINING_PRESETS_BUILTIN
_LOADED_TRAINING_YAML = _try_load_training_presets_from_yaml()
if _LOADED_TRAINING_YAML is not None:
    TRAINING_PRESETS = _LOADED_TRAINING_YAML

_validate_training_presets(TRAINING_PRESETS)

# ---------------------------------------------------------------------------
# 命名 hardware 预设（可选；与 presets/hardware/*.yaml 及内置 HARDWARE_PRESETS 对齐）
#
# 主线：父进程 ``load_resolved_config`` 选定 stem，序列化硬件切片写入 ``D4C_HARDWARE_PROFILE_JSON``，并注入
# ``D4C_HARDWARE_PRESET``（stem 字符串）。torchrun 子进程 **只消费** 上述注入结果，不再把父 shell 的零散 ENV
# 当作二次选型入口。
#
# 裸调 / 无 JSON 注入时的回退优先级：D4C_HARDWARE_PRESET 命名表 → MAX_PARALLEL_CPU 等 ENV → CLI（仅辅路径）。
# ---------------------------------------------------------------------------

_HARDWARE_PRESET_ALLOWED_KEYS: FrozenSet[str] = frozenset(
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
        "omp_num_threads",
        "mkl_num_threads",
        "tokenizers_parallelism",
        # launcher-only：仅出现在 YAML/命名表校验；不入 D4C_HARDWARE_PROFILE_JSON 归一化结果
        "cuda_visible_devices",
    }
)

_HARDWARE_PRESET_INT_KEYS: FrozenSet[str] = _HARDWARE_PRESET_ALLOWED_KEYS - frozenset(
    {"tokenizers_parallelism", "cuda_visible_devices"}
)


def _normalize_hardware_profile_mapping(obj: Mapping[str, Any]) -> Dict[str, Any]:
    """将 D4C_HARDWARE_PROFILE_JSON 或同类 mapping 规范为 build_resolved 可用的 hardware 切片。"""
    out: Dict[str, Any] = {}
    if not isinstance(obj, Mapping):
        return out
    for k, v in obj.items():
        sk = str(k)
        if sk not in _HARDWARE_PRESET_ALLOWED_KEYS:
            continue
        if sk == "cuda_visible_devices":
            continue
        if sk == "tokenizers_parallelism":
            if isinstance(v, bool):
                out[sk] = v
            else:
                out[sk] = str(v).strip().lower() in ("true", "1", "yes", "on")
            continue
        try:
            iv = int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        out[sk] = iv
    return out


def _validate_hardware_presets(presets: Dict[str, Any], *, name: str = "HARDWARE_PRESETS") -> None:
    for preset_name, blob in presets.items():
        if not isinstance(blob, dict):
            raise TypeError(f'{name} 中预设 {preset_name!r} 应为 dict，当前为 {type(blob).__name__}')
        unknown = set(blob.keys()) - _HARDWARE_PRESET_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{name} 预设 {preset_name!r} 含有未知字段 {sorted(unknown)}；"
                f"允许: {sorted(_HARDWARE_PRESET_ALLOWED_KEYS)}"
            )
        for k, v in blob.items():
            if k == "cuda_visible_devices":
                if not isinstance(v, str) or not str(v).strip():
                    raise TypeError(
                        f"{name} 预设 {preset_name!r} 字段 cuda_visible_devices 须为非空 str，当前为 {type(v).__name__}"
                    )
                continue
            if k == "tokenizers_parallelism":
                if not isinstance(v, bool):
                    raise TypeError(
                        f"{name} 预设 {preset_name!r} 字段 tokenizers_parallelism 须为 bool，当前为 {type(v).__name__}"
                    )
                continue
            if k in _HARDWARE_PRESET_INT_KEYS:
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
                if k in ("omp_num_threads", "mkl_num_threads") and iv < 1:
                    raise ValueError(f"{name} 预设 {preset_name!r} 字段 {k!r} 须 >= 1，当前为 {iv}")


_HARDWARE_PRESETS_BUILTIN: Dict[str, Dict[str, Any]] = {
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

HARDWARE_PRESETS: Dict[str, Dict[str, Any]] = _HARDWARE_PRESETS_BUILTIN
_LOADED_HARDWARE_YAML = _try_load_hardware_presets_from_yaml()
if _LOADED_HARDWARE_YAML is not None:
    HARDWARE_PRESETS = _LOADED_HARDWARE_YAML

_validate_hardware_presets(HARDWARE_PRESETS)

_unknown_hardware_preset_warned: Set[str] = set()


def _named_hardware_preset_blob() -> Optional[Dict[str, Any]]:
    name = (os.environ.get("D4C_HARDWARE_PRESET") or "").strip()
    if not name:
        return None
    raw = HARDWARE_PRESETS.get(name)
    if raw is None:
        import warnings

        if name not in _unknown_hardware_preset_warned:
            _unknown_hardware_preset_warned.add(name)
            warnings.warn(
                f"未知 D4C_HARDWARE_PRESET={name!r}，忽略 hardware 预设（按未设置 preset 继续）。",
                UserWarning,
                stacklevel=2,
            )
        return None
    return raw if isinstance(raw, dict) else None


def _active_hardware_preset_slice() -> Optional[Dict[str, Any]]:
    """当前激活的 hardware 切片：优先 D4C_HARDWARE_PROFILE_JSON（d4c 注入子进程），否则命名预设表。"""
    rawj = (os.environ.get("D4C_HARDWARE_PROFILE_JSON") or "").strip()
    if rawj:
        try:
            loaded = json.loads(rawj)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            norm = _normalize_hardware_profile_mapping(loaded)
            if norm:
                return norm
    return _named_hardware_preset_blob()


def training_preset_is_per_task() -> bool:
    """历史 API：环境变量不再驱动训练预设切片；恒为 False。"""
    return False


def get_task_config(task_idx: int) -> Optional[TaskConfig]:
    """仅返回 TASK_DEFAULTS 表项（不含预设合并）；训练时以 build_resolved_training_config 为准。"""
    return TASK_DEFAULTS.get(int(task_idx))


def resolve_train_batch_layout(
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


# 历史名：与 resolve_train_batch_layout 同义（单一实现，避免两套公式漂移）
resolve_ddp_train_microbatch_layout = resolve_train_batch_layout


def resolve_eval_batch_layout(eval_batch_size: int, ddp_world_size: int) -> Tuple[int, int]:
    """
    全局评测 batch 按 DDP world size 切分；返回 (global_eval_batch_size, eval_per_gpu_batch_size)。
    与 step5_engine / adv_train_core 中 eval strict 校验一致；step4 反事实推理使用同一合同。
    """
    E = int(eval_batch_size)
    W = int(ddp_world_size)
    if E < 1:
        raise ValueError(f"eval_batch_size 须 >= 1，当前为 {E}")
    if W < 1:
        raise ValueError(f"ddp_world_size 须 >= 1，当前为 {W}")
    if E % W != 0:
        raise ValueError(
            f"eval_batch_size={E} 与 world_size={W} 不整除。"
            "请修改 presets/eval_profiles/*.yaml 的 eval_batch_size，或调整 presets/hardware/*.yaml 的 ddp_world_size。"
        )
    return E, E // W


def resolve_train_batch_from_training_row(
    row: Mapping[str, Any],
    world_size: int,
) -> Tuple[int, int, int, int]:
    """
    与 ``build_resolved_training_config`` 中 G/P/A 解析一致；返回
    ``(train_batch_size_G, per_device_train_batch_size, gradient_accumulation_steps, effective_global_batch_size)``。
    """
    G = int(BASE_TRAINING_DEFAULTS.train_batch_size)
    if "train_batch_size" in row:
        G = _preset_int_min(row["train_batch_size"], 1)
    p_opt: Optional[int] = None
    if "per_device_train_batch_size" in row:
        p_opt = max(1, int(row["per_device_train_batch_size"]))
    A0 = int(BASE_TRAINING_DEFAULTS.gradient_accumulation_steps)
    if "gradient_accumulation_steps" in row:
        A0 = _preset_int_min(row["gradient_accumulation_steps"], 1)
    a_cli: Optional[int] = None
    if p_opt is not None:
        accum_for_layout: Optional[int] = a_cli
    else:
        accum_for_layout = A0
    G, P, A = resolve_train_batch_layout(
        G,
        world_size,
        per_device_batch_size=p_opt,
        gradient_accumulation_steps=accum_for_layout,
    )
    eff = P * int(world_size) * A
    return G, P, A, eff


def get_embed_batch_size():
    """返回 embedding 计算的 batch_size，供 run_preprocess_and_embed / compute_embeddings 使用"""
    return embed_batch_size


def get_eval_batch_size(cli: Optional[int] = None) -> int:
    """辅脚本用 eval batch；优先级：cli > EVAL_BATCH_SIZE > BASE。d4c torchrun 子进程请用 effective payload。"""
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
    """无 hardware preset / 无对应 ENV 时的 DataLoader workers 推导（与历史逻辑一致）。"""
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
    """hardware 自动推导 → hardware_preset → ENV → 可选 CLI。"""
    mp = _resolve_max_parallel_cpu_cli(None)
    nw = _auto_derive_dataloader_num_workers(split, mp)
    rp = _active_hardware_preset_slice()
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
    """derived(min(cpu,max_parallel)) → hardware_preset → D4C_NUM_PROC → CLI。"""
    mp = _resolve_max_parallel_cpu_cli(None)
    v = min(_get_num_cpu(), mp)
    rp = _active_hardware_preset_slice()
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
    rp = _active_hardware_preset_slice()
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
    """datasets.map（Tokenize）并行进程数；与 DataLoader num_workers 独立。hardware_preset / D4C_NUM_PROC 可覆盖。"""
    return _resolve_num_proc_cli(None)


def get_max_parallel_cpu() -> int:
    """并行 CPU 上限；hardware_preset → MAX_PARALLEL_CPU。"""
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
    若给定 split 且 hardware preset（或 D4C_PREFETCH_*）有值则用之；否则与历史一致为 prefetch=4。
    """
    if num_workers <= 0:
        return None
    rp = _active_hardware_preset_slice()
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


def _hardware_cli_val(
    hardware_overrides: Optional[Dict[str, Any]],
    args: Any,
    key: str,
) -> Any:
    """优先使用入口收集的 hardware_overrides（与 args 同源、非 None 字段），等价于显式 override dict。"""
    if hardware_overrides is not None and key in hardware_overrides:
        return hardware_overrides[key]
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
    hardware_preset_name: Optional[str]
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

    full_bleu_eval_resolved: FullBleuEvalResolved

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
    # 训练期 full BLEU 监控解码：greedy / inherit（inherit=与 decode_strategy 一致）；非正式 eval 口径
    full_bleu_decode_strategy: str = "inherit"

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
    train_label_max_length: int = 64
    train_dynamic_padding: bool = True
    train_padding_strategy: str = "dynamic_batch"
    decode_strategy: str = "greedy"
    decode_seed: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    min_len: Optional[int] = None
    # --- decode controller v2（手写 generate；与 presets/decode 对齐）---
    soft_max_len: Optional[int] = None
    hard_max_len: Optional[int] = None
    eos_boost_start: int = 9999
    eos_boost_value: float = 0.0
    tail_temperature: float = -1.0
    tail_top_p: float = -1.0
    forbid_eos_after_open_quote: bool = True
    forbid_eos_after_open_bracket: bool = True
    forbid_bad_terminal_tokens: bool = True
    bad_terminal_token_ids: Optional[Tuple[int, ...]] = None
    decode_token_repeat_window: int = 4
    decode_token_repeat_max: int = 2
    candidate_family: str = "balanced"
    candidate_mixed_include_diverse: bool = True
    # --- explanation 支路轻量质量正则 ---
    loss_weight_repeat_ul: float = 0.0
    loss_weight_terminal_clean: float = 0.0
    terminal_clean_span: int = 3
    nuser: int = 0
    nitem: int = 0

    device: int = 0
    device_ids: Tuple[int, ...] = ()
    save_file: str = ""
    # Step5：save_file = best_mainline.pth；last 仅训练结束写入 last_checkpoint_path
    last_checkpoint_path: str = ""
    log_file: Optional[str] = None
    ddp_world_size: int = 1
    ddp_find_unused_parameters: bool = False
    rank0_only_logging: bool = True
    run_id: str = ""
    ddp_fast_backends: bool = False
    # 由 d4c runners 注入 D4C_EVAL_PROFILE_NAME（仅评测侧记录）
    eval_profile_name: Optional[str] = None

    logger: Any = None
    valid_dataset: Any = None

    def to_log_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("logger", None)
        d.pop("valid_dataset", None)
        d["sources"] = dict(self.sources)
        return d


def build_mainline_alignment_monitor_override(cfg: FinalTrainingConfig) -> Dict[str, Any]:
    """
    主路径监控 / best_mainline 选模 / 默认外评：与 greedy alignment 一致，禁止尾部加噪 schedule。
    与训练 decode 预设的长度等对齐，但强制 strategy=greedy、tail_temperature/tail_top_p 关闭。
    """
    hm = int(getattr(cfg, "hard_max_len", None) or cfg.max_explanation_length)
    sm = getattr(cfg, "soft_max_len", None)
    soft = int(sm) if sm is not None and int(sm) > 0 else max(1, hm - 8)
    _nr = cfg.no_repeat_ngram_size
    nrs = max(3, int(_nr)) if _nr is not None and int(_nr) > 0 else 3
    _mn = cfg.min_len
    min_l = max(4, int(_mn)) if _mn is not None and int(_mn) > 0 else 4
    return {
        "strategy": "greedy",
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": float(cfg.repetition_penalty),
        "no_repeat_ngram_size": nrs,
        "min_len": min_l,
        "soft_max_len": soft,
        "hard_max_len": max(1, hm),
        "eos_boost_start": int(getattr(cfg, "eos_boost_start", 9999)),
        "eos_boost_value": float(getattr(cfg, "eos_boost_value", 0.0)),
        "tail_temperature": -1.0,
        "tail_top_p": -1.0,
        "forbid_eos_after_open_quote": bool(getattr(cfg, "forbid_eos_after_open_quote", True)),
        "forbid_eos_after_open_bracket": bool(getattr(cfg, "forbid_eos_after_open_bracket", True)),
        "forbid_bad_terminal_tokens": bool(getattr(cfg, "forbid_bad_terminal_tokens", True)),
        "token_repeat_window": int(getattr(cfg, "decode_token_repeat_window", 4)),
        "token_repeat_max": int(getattr(cfg, "decode_token_repeat_max", 2)),
        "decode_seed": cfg.decode_seed,
    }


def build_full_bleu_monitor_cfg_override(cfg: FinalTrainingConfig) -> Dict[str, Any]:
    """训练期 full monitor 与 best_mainline 选模统一走 Mainline Alignment（忽略 inherit/nucleus 主 decode）。"""
    _ = str(cfg.full_bleu_decode_strategy).strip().lower()  # 仍写入 manifest，监控侧不再分支
    return build_mainline_alignment_monitor_override(cfg)


def format_full_bleu_monitor_log_line(cfg: FinalTrainingConfig) -> str:
    return (
        "[mainline_monitor] decode=greedy deterministic_alignment "
        "tail_noise=off no_repeat_ngram>=3 min_len>=4 (build_mainline_alignment_monitor_override)"
    )


def build_resolved_training_config(
    args: Any,
    *,
    task_idx: int,
    world_size: int,
    hardware_overrides: Optional[Dict[str, Any]] = None,
) -> FinalTrainingConfig:
    """
    **torchrun 子进程**构造 ``FinalTrainingConfig`` 的唯一入口。

    - 训练语义：必须存在 ``D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON``（父进程 ``load_resolved_config`` 写入的
      schema_version>=2 payload）；**禁止**再用 ``D4C_TRAIN_PRESET`` / ``TRAINING_PRESETS`` / ``TRAIN_*`` 重算训练切片。
    - 硬件语义：须由父进程注入 ``D4C_HARDWARE_PROFILE_JSON``（与 ``--hardware-preset`` / eval_profile 选定结果一致）；
      已注入时 **仅信任 JSON**，不再用 ``MAX_PARALLEL_CPU`` / ``D4C_NUM_PROC`` 等父 shell 残留覆盖。
    - ``hardware_overrides``：来自 CLI 的显式覆盖 dict（通常由 ``collect_training_hardware_overrides_from_args`` 收集），
      与训练 payload 正交；不写回 ``os.environ``。
    """
    ro = hardware_overrides if hardware_overrides is not None else collect_training_hardware_overrides_from_args(args)
    src: Dict[str, str] = {}

    def rv(key: str) -> Any:
        return _hardware_cli_val(ro, args, key)

    tid = int(task_idx)
    tc = TASK_DEFAULTS.get(tid)
    if tc is None:
        raise ValueError(f"无效 task_idx={tid}，TASK_DEFAULTS 中无此任务")

    _eff_raw = (os.environ.get("D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()
    if not _eff_raw:
        raise RuntimeError(
            "缺少 D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON：本进程须由 `python code/d4c.py …` 经 torchrun 启动。\n"
            "禁止裸调 executors/step5_entry、step3_entry 而未注入父进程 effective training payload。"
        )
    try:
        _payload = json.loads(_eff_raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON 非合法 JSON: {e}") from e
    if int(_payload.get("task_id", -1)) != tid:
        raise ValueError(
            f"effective payload task_id={_payload.get('task_id')!r} 与当前解析 task_idx={tid} 不一致"
        )
    row = _payload.get("training_row")
    if not isinstance(row, dict):
        raise TypeError("effective payload 缺少 training_row dict")
    preset_nm = str(_payload.get("preset_name") or "").strip() or None

    auxiliary = str(_payload.get("auxiliary") or "").strip()
    target = str(_payload.get("target") or "").strip()
    if not auxiliary or not target:
        raise RuntimeError(
            "D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON 缺少 auxiliary/target；"
            "须由新版 python code/d4c.py 注入（schema_version>=2）。"
        )

    _hardware_json_injected = bool((os.environ.get("D4C_HARDWARE_PROFILE_JSON") or "").strip())

    # ----- global train batch G / P / A（仅父进程下发的 training_row；与 resolve_train_batch_from_training_row 共用）-----
    src["train_batch_size"] = "base"
    if "train_batch_size" in row:
        src["train_batch_size"] = "effective_payload"
    src["per_device_train_batch_size"] = "base"
    if "per_device_train_batch_size" in row:
        src["per_device_train_batch_size"] = "effective_payload"
    src["gradient_accumulation_steps"] = "base"
    if "gradient_accumulation_steps" in row:
        src["gradient_accumulation_steps"] = "effective_payload"

    G, P, A, eff = resolve_train_batch_from_training_row(row, world_size)

    # ----- epochs -----
    ep = int(BASE_TRAINING_DEFAULTS.epochs)
    src["epochs"] = "base"
    if "epochs" in row:
        ep = _preset_int_min(row["epochs"], 1)
        src["epochs"] = "effective_payload"

    # ----- learning rate（task 表底 + training_row）-----
    initial_f = float(tc["lr"])
    src["learning_rate"] = "task"
    if "lr" in row:
        initial_f = float(_coerce_task_param_numeric(row["lr"]))
        src["learning_rate"] = "effective_payload"

    # ----- min_lr_ratio -----
    min_lr = float(BASE_TRAINING_DEFAULTS.min_lr_ratio)
    src["min_lr_ratio"] = "base"
    if "min_lr_ratio" in row:
        min_lr = float(row["min_lr_ratio"])
        src["min_lr_ratio"] = "effective_payload"

    # ----- lr_scheduler -----
    lr_sched = str(BASE_TRAINING_DEFAULTS.lr_scheduler)
    src["lr_scheduler"] = "base"
    if "lr_scheduler" in row and str(row["lr_scheduler"]).strip():
        v = str(row["lr_scheduler"]).strip().lower()
        if v in ("none", "off", "disabled"):
            lr_sched = "none"
        elif v in ("warmup_cosine", "warmup-cosine", "cosine"):
            lr_sched = "warmup_cosine"
        else:
            lr_sched = "none"
        src["lr_scheduler"] = "effective_payload"

    # ----- warmup_epochs -----
    wu_ep = float(BASE_TRAINING_DEFAULTS.warmup_epochs)
    src["warmup_epochs"] = "base"
    if "warmup_epochs" in row:
        wu_ep = float(row["warmup_epochs"])
        src["warmup_epochs"] = "effective_payload"

    # ----- warmup steps / ratio -----
    wsteps: Optional[int] = None
    src["d4c_warmup_steps"] = "base"
    if "warmup_steps" in row:
        v = int(row["warmup_steps"])
        wsteps = v if v > 0 else None
        src["d4c_warmup_steps"] = "effective_payload"

    wratio: Optional[float] = None
    src["d4c_warmup_ratio"] = "base"
    if "warmup_ratio" in row:
        wratio = float(row["warmup_ratio"])
        src["d4c_warmup_ratio"] = "effective_payload"

    # ----- eval batch -----
    eval_bs = int(BASE_TRAINING_DEFAULTS.eval_batch_size)
    src["eval_batch_size"] = "base"
    if "eval_batch_size" in row:
        eval_bs = max(1, int(row["eval_batch_size"]))
        src["eval_batch_size"] = "effective_payload"

    # ----- early stop -----
    min_ep = max(1, int(BASE_TRAINING_DEFAULTS.train_min_epochs))
    src["min_epochs"] = "base"
    if "min_epochs" in row:
        min_ep = max(1, int(row["min_epochs"]))
        src["min_epochs"] = "effective_payload"

    esp = max(1, int(BASE_TRAINING_DEFAULTS.train_early_stop_patience))
    src["early_stop_patience"] = "base"
    if "early_stop_patience" in row:
        esp = max(1, int(row["early_stop_patience"]))
        src["early_stop_patience"] = "effective_payload"

    if "early_stop_patience_full" in row:
        esp_full = max(1, int(row["early_stop_patience_full"]))
        src["early_stop_patience_full"] = "effective_payload"
    else:
        esp_full = esp
        src["early_stop_patience_full"] = src["early_stop_patience"]

    if "early_stop_patience_loss" in row:
        esp_loss = max(1, int(row["early_stop_patience_loss"]))
        src["early_stop_patience_loss"] = "effective_payload"
    else:
        esp_loss = esp
        src["early_stop_patience_loss"] = src["early_stop_patience"]

    # ----- BLEU samples -----
    b4 = max(64, int(BASE_TRAINING_DEFAULTS.train_bleu4_max_samples))
    src["bleu4_max_samples"] = "base"
    if "bleu4_max_samples" in row:
        b4 = max(64, int(row["bleu4_max_samples"]))
        src["bleu4_max_samples"] = "effective_payload"

    qeval = b4
    src["quick_eval_max_samples"] = "base"
    if "quick_eval_max_samples" in row:
        qeval = max(64, int(row["quick_eval_max_samples"]))
        src["quick_eval_max_samples"] = "effective_payload"

    # ----- full BLEU eval schedule（仅 training_row.full_bleu_eval；旧字段在 resolve 内直接报错）-----
    fe_sched = resolve_full_bleu_eval_from_training_row(row)
    src["full_bleu_eval"] = "effective_payload"

    fb_ds = "inherit"
    src["full_bleu_decode_strategy"] = "base"
    if "full_bleu_decode_strategy" in row:
        fb_ds = parse_full_bleu_decode_strategy(row["full_bleu_decode_strategy"])
        src["full_bleu_decode_strategy"] = "effective_payload"

    ckpt_metric = str(getattr(args, "checkpoint_metric", "mainline_composite"))
    dual = ckpt_metric in ("bleu4", "mainline_composite") and fe_sched.enabled

    # ----- coef / adversarial_coef -----
    coef_f = float(_coerce_task_param_numeric(tc["coef"]))
    src["coef"] = "task"
    if "coef" in row:
        coef_f = float(_coerce_task_param_numeric(row["coef"]))
        src["coef"] = "effective_payload"

    adv_f = float(_coerce_task_param_numeric(tc["adv"]))
    src["adversarial_coef"] = "task"
    if "adv" in row:
        adv_f = float(_coerce_task_param_numeric(row["adv"]))
        src["adversarial_coef"] = "effective_payload"

    # ----- eta（父进程 payload）-----
    eta_f = float(_payload.get("eta", 1e-3))
    src["eta"] = "effective_payload"

    # ----- adversarial schedule（仅 training_row）-----
    start = row.get("adversarial_start_epoch")
    src["adversarial_schedule"] = "base"
    if start is None:
        adv_sched_on = False
        adv_skip = 0
        adv_warm = 0
        adv_target = adv_f
        src["adversarial_schedule_enabled"] = "base"
    else:
        adv_sched_on = True
        adv_skip = max(0, int(start))
        w = row.get("adversarial_warmup_epochs")
        adv_warm = 0 if w is None else max(0, int(w))
        t = row.get("adversarial_coef_target")
        adv_target = adv_f if t is None else float(t)
        src["adversarial_schedule_enabled"] = "effective_payload"

    # ----- hardware preset 元数据（D4C_HARDWARE_PRESET 由 runners 注入的 stem；未知 stem 时 blob 为 None）-----
    hardware_preset_nm = (os.environ.get("D4C_HARDWARE_PRESET") or "").strip() or None

    # ----- max_parallel_cpu / num_proc：已注入 D4C_HARDWARE_PROFILE_JSON 时仅信任 JSON（单一真相源）-----
    max_par_v = max(1, int(_HARDWARE_BASE_MAX_PARALLEL_CPU))
    src["max_parallel_cpu"] = "base"
    rp_rt = _active_hardware_preset_slice()
    if _hardware_json_injected and rp_rt:
        if "max_parallel_cpu" in rp_rt:
            max_par_v = max(1, int(rp_rt["max_parallel_cpu"]))
            src["max_parallel_cpu"] = "hardware_profile_json"
        num_proc_v = min(_get_num_cpu(), max_par_v)
        src["num_proc"] = "derived"
        if "num_proc" in rp_rt:
            num_proc_v = max(1, int(rp_rt["num_proc"]))
            num_proc_v = min(num_proc_v, _get_num_cpu())
            src["num_proc"] = "hardware_profile_json"
    elif _hardware_json_injected:
        raise RuntimeError(
            "D4C_HARDWARE_PROFILE_JSON 已设置但无法解析为有效 hardware 切片；"
            "请检查父进程 hardware 预设导出的 JSON。"
        )
    else:
        if rp_rt and "max_parallel_cpu" in rp_rt:
            max_par_v = max(1, int(rp_rt["max_parallel_cpu"]))
            src["max_parallel_cpu"] = "hardware_preset"
        if not _hardware_json_injected and "MAX_PARALLEL_CPU" in os.environ:
            max_par_v = max(1, int(os.environ["MAX_PARALLEL_CPU"]))
            src["max_parallel_cpu"] = "env"
        num_proc_v = min(_get_num_cpu(), max_par_v)
        src["num_proc"] = "derived"
        if rp_rt and "num_proc" in rp_rt:
            num_proc_v = max(1, int(rp_rt["num_proc"]))
            num_proc_v = min(num_proc_v, _get_num_cpu())
            src["num_proc"] = "hardware_preset"
        if not _hardware_json_injected and "D4C_NUM_PROC" in os.environ:
            num_proc_v = max(1, int(os.environ["D4C_NUM_PROC"]))
            num_proc_v = min(num_proc_v, _get_num_cpu())
            src["num_proc"] = "env"
        if not _hardware_json_injected and rv("num_proc") is not None:
            num_proc_v = max(1, int(rv("num_proc")))
            num_proc_v = min(num_proc_v, _get_num_cpu())
            src["num_proc"] = "cli"

    ws = max(int(world_size), 1)
    nw_train = _resolve_ddp_train_num_workers_per_rank_cli(ws, None)
    if rp_rt and (
        "dataloader_num_workers_train" in rp_rt or "dataloader_workers_train_per_rank_cap" in rp_rt
    ):
        src["dataloader_num_workers_train"] = "hardware_preset"
    elif (
        not _hardware_json_injected
        and (
            "D4C_DATALOADER_WORKERS_TRAIN" in os.environ or "D4C_DATALOADER_TRAIN_PER_RANK_CAP" in os.environ
        )
    ):
        src["dataloader_num_workers_train"] = "env"
    else:
        src["dataloader_num_workers_train"] = "derived"

    dl_valid_base = _resolve_dataloader_num_workers_for_split("valid", None)
    nw_valid = max(1, min(dl_valid_base, nw_train))
    if rp_rt and "dataloader_num_workers_valid" in rp_rt:
        src["dataloader_num_workers_valid"] = "hardware_preset"
    elif not _hardware_json_injected and "D4C_DATALOADER_WORKERS_VALID" in os.environ:
        src["dataloader_num_workers_valid"] = "env"
    else:
        src["dataloader_num_workers_valid"] = "derived"

    nw_test = _resolve_dataloader_num_workers_for_split("test", None)
    if rp_rt and "dataloader_num_workers_test" in rp_rt:
        src["dataloader_num_workers_test"] = "hardware_preset"
    elif not _hardware_json_injected and "D4C_DATALOADER_WORKERS_TEST" in os.environ:
        src["dataloader_num_workers_test"] = "env"
    else:
        src["dataloader_num_workers_test"] = "derived"

    pf_t = get_dataloader_prefetch_factor(nw_train, split="train")
    pf_v = get_dataloader_prefetch_factor(nw_valid, split="valid")
    pf_test = get_dataloader_prefetch_factor(nw_test, split="test")
    src["dataloader_prefetch_factor_train"] = (
        "hardware_preset"
        if rp_rt and "dataloader_prefetch_factor_train" in rp_rt
        else ("env" if not _hardware_json_injected and "D4C_PREFETCH_TRAIN" in os.environ else "derived")
    )
    src["dataloader_prefetch_factor_valid"] = (
        "hardware_preset"
        if rp_rt and "dataloader_prefetch_factor_valid" in rp_rt
        else ("env" if not _hardware_json_injected and "D4C_PREFETCH_VALID" in os.environ else "derived")
    )
    src["dataloader_prefetch_factor_test"] = (
        "hardware_preset"
        if rp_rt and "dataloader_prefetch_factor_test" in rp_rt
        else ("env" if not _hardware_json_injected and "D4C_PREFETCH_TEST" in os.environ else "derived")
    )

    _tlm_base = int(max(8, min(512, int(BASE_TRAINING_DEFAULTS.train_label_max_length))))
    src["train_label_max_length"] = "base"
    train_label_max_length_v = _tlm_base
    if "train_label_max_length" in row:
        train_label_max_length_v = int(max(8, min(512, int(row["train_label_max_length"]))))
        src["train_label_max_length"] = "effective_payload"

    train_dynamic_padding_v = True
    src["train_dynamic_padding"] = "base"
    if "train_dynamic_padding" in row:
        train_dynamic_padding_v = bool(row["train_dynamic_padding"])
        src["train_dynamic_padding"] = "effective_payload"

    loss_weight_repeat_ul_v = 0.0
    src["loss_weight_repeat_ul"] = "base"
    if "loss_weight_repeat_ul" in row:
        loss_weight_repeat_ul_v = float(row["loss_weight_repeat_ul"])
        src["loss_weight_repeat_ul"] = "effective_payload"

    loss_weight_terminal_clean_v = 0.0
    src["loss_weight_terminal_clean"] = "base"
    if "loss_weight_terminal_clean" in row:
        loss_weight_terminal_clean_v = float(row["loss_weight_terminal_clean"])
        src["loss_weight_terminal_clean"] = "effective_payload"

    terminal_clean_span_v = 3
    src["terminal_clean_span"] = "base"
    if "terminal_clean_span" in row:
        terminal_clean_span_v = max(1, int(row["terminal_clean_span"]))
        src["terminal_clean_span"] = "effective_payload"

    ddp_find_unused_v = False
    src["ddp_find_unused_parameters"] = "base"
    if "ddp_find_unused_parameters" in row:
        ddp_find_unused_v = bool(row["ddp_find_unused_parameters"])
        src["ddp_find_unused_parameters"] = "effective_payload"

    sources_tuple = tuple(sorted(src.items()))

    _eval_b = (os.environ.get("D4C_EVAL_PROFILE_NAME") or "").strip() or None

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
        hardware_preset_name=hardware_preset_nm,
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
        full_bleu_eval_resolved=fe_sched,
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
        full_bleu_decode_strategy=fb_ds,
        eval_profile_name=_eval_b,
        train_label_max_length=int(train_label_max_length_v),
        train_dynamic_padding=bool(train_dynamic_padding_v),
        train_padding_strategy=("dynamic_batch" if bool(train_dynamic_padding_v) else "fixed_max_length"),
        loss_weight_repeat_ul=float(loss_weight_repeat_ul_v),
        loss_weight_terminal_clean=float(loss_weight_terminal_clean_v),
        terminal_clean_span=int(terminal_clean_span_v),
        ddp_find_unused_parameters=ddp_find_unused_v,
    )


def get_train_batch_size(task_idx: Optional[int] = None) -> int:
    """
    全局训练 batch G；供**无 d4c CLI** 的极薄辅助场景。
    顺序：BASE → ``D4C_TRAIN_BATCH_SIZE`` / ``D4C_OPT_BATCH_SIZE``。
    ``task_idx`` 保留以兼容旧调用方，**不**再参与解析。
    """
    _ = task_idx
    G = int(BASE_TRAINING_DEFAULTS.train_batch_size)
    if "D4C_TRAIN_BATCH_SIZE" in os.environ:
        G = _preset_int_min(os.environ["D4C_TRAIN_BATCH_SIZE"], 1)
    elif "D4C_OPT_BATCH_SIZE" in os.environ:
        G = _preset_int_min(os.environ["D4C_OPT_BATCH_SIZE"], 1)
    return G


def get_epochs(task_idx: Optional[int] = None) -> int:
    """
    训练轮数；供**无 d4c CLI** 的极薄辅助场景。
    顺序：BASE → ``D4C_EPOCHS``。
    ``task_idx`` 保留以兼容旧调用方，**不**再参与解析。
    """
    _ = task_idx
    ep = int(BASE_TRAINING_DEFAULTS.epochs)
    if "D4C_EPOCHS" in os.environ:
        ep = _preset_int_min(os.environ["D4C_EPOCHS"], 1)
    return ep


def __getattr__(name: str) -> Any:
    """惰性解析：``from config import num_proc`` / ``eval_batch_size`` 等。"""
    if name == "num_proc":
        return get_num_proc()
    if name == "eval_batch_size":
        return get_eval_batch_size()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
