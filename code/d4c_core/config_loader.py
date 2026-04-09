"""
MAINLINE 支撑模块 — 供 ``d4c.py`` 解析 ResolvedConfig；不单独作为用户入口。

单一真相源：父进程在此完成分层合并，注入 ``D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON``、
``D4C_HARDWARE_PROFILE_JSON``（hardware **语义**切片：不含 ``omp_num_threads`` / ``mkl_num_threads`` /
``tokenizers_parallelism`` / ``cuda_visible_devices``；线程与 CUDA 由 ``runners`` 写子进程 **env**）、
decode/rerank JSON。子进程 **只消费** 上述注入结果，
不再读取平行 task/preset ENV 链。

真正参与 merge 的配置层：task → training → hardware → decode（若该命令消费）→ rerank（仅 eval-rerank）
→ **CLI 标量最后覆盖**。

``eval_profile`` **不是** merge 主链上的一层；它是**编排层 / selector**：先解析 profile 得到引用的
hardware/decode/rerank preset id 与少量 bundle-owned 字段（``eval_batch_size``、``num_return_sequences``），
再按上式加载各 YAML 并合并。

merge 顺序（命令）：
  step3: task → training → hardware → CLI
  step4: **须** resolve eval_profile（``--eval-profile``）→ task → training → hardware → CLI（推理 batch 仅来自 profile）
  step5: 可选 resolve eval_profile（若 ``--eval-profile``）→ task → training → hardware → decode（仅非 --train-only）→ CLI
  eval*: 先 resolve eval_profile（若提供）→ task → training（上下文）→ hardware → decode
        → bundle_owned 字段 → CLI
  eval-rerank*: 同上 → rerank → CLI
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from config import (
    BASE_TRAINING_DEFAULTS,
    _normalize_cuda_visible_devices_yaml,
    parse_full_bleu_decode_strategy,
    resolve_eval_batch_layout,
    resolve_full_bleu_eval_from_training_row,
    resolve_train_batch_from_training_row,
)

from d4c_core import path_layout, run_naming
from d4c_core.training_diagnostics import runtime_diagnostics_fingerprint_source

_CODE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CODE_DIR.parent
_PRESETS = _REPO_ROOT / "presets"

# step3/step4 不加载 decode YAML 时，ResolvedConfig 中 decode 相关标量的占位（不进入 generation_semantic_fingerprint）
_DECODE_PLACEHOLDER: Dict[str, Any] = {
    "label_smoothing": 0.1,
    "repetition_penalty": 1.15,
    "generate_temperature": 0.8,
    "generate_top_p": 0.9,
    "decode_strategy": "greedy",
    "decode_seed": None,
    "max_explanation_length": 25,
    "no_repeat_ngram_size": None,
    "min_len": None,
}

# rerank YAML 顶层「编排」字段：不进 D4C_RERANK_PROFILE_JSON（仅 rule_v3 权重等进子进程）
_RERANK_YAML_ORCHESTRATION_KEYS = frozenset(
    {"num_return_sequences", "export_examples_mode", "rerank_method"}
)

# 使用 eval_profile 编排的高层 eval 入口（profile 本身不参与 YAML 平铺合并）
_EVAL_PROFILE_COMMANDS = frozenset({"eval", "eval-rerank", "eval-matrix", "eval-rerank-matrix"})

# presets/eval_profiles/*.yaml 仅允许这些顶层键（须通过 preset 引用硬件/解码/rerank，禁止复制整段参数块）
_EVAL_PROFILE_ALLOWED_KEYS = frozenset(
    {"hardware_preset", "decode_preset", "rerank_preset", "eval_batch_size", "num_return_sequences"}
)

# 须向子进程下发 D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON 的命令（torchrun + build_resolved_training_config）
_EFFECTIVE_TRAINING_PAYLOAD_COMMANDS = frozenset({"step3", "step5", "eval", "eval-rerank"})
_DECODE_TRAINING_ONLY_FORBIDDEN_KEYS = frozenset(
    {"loss_weight_repeat_ul", "loss_weight_terminal_clean", "terminal_clean_span"}
)
_TRAINING_FORBIDDEN_KEYS = frozenset({"eval_batch_size", "num_return_sequences"})
_HARDWARE_FORBIDDEN_KEYS = frozenset(
    {
        "train_batch_size",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "eval_batch_size",
        "num_return_sequences",
    }
)

# 自 hardware YAML 进入 _export_rt，但不写入 D4C_HARDWARE_PROFILE_JSON、不参与 training_semantic_fingerprint
_RUNTIME_LAUNCHER_HARDWARE_KEYS = frozenset(
    {
        "omp_num_threads",
        "mkl_num_threads",
        "tokenizers_parallelism",
        "cuda_visible_devices",
    }
)


def _nonempty_str(v: Any) -> bool:
    return v is not None and str(v).strip() != ""


def _shell_positive_int(name: str) -> Optional[int]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _shell_tokenizers_parallelism() -> Optional[bool]:
    raw = (os.environ.get("TOKENIZERS_PARALLELISM") or "").strip()
    if not raw:
        return None
    return raw.lower() in ("1", "true", "yes", "on")


def _assert_visible_gpus_for_ddp(ddp_world_size: int) -> None:
    """可见 GPU 数不足时直接失败（不再自动夹断）。"""
    if ddp_world_size < 1:
        raise ValueError("ddp_world_size 须 >= 1")
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    n = int(torch.cuda.device_count())
    if ddp_world_size > n:
        raise ValueError(
            f"ddp_world_size={ddp_world_size} 超过当前可见 GPU 数 ({n})。"
            f"请减小 DDP 进程数、调整 CUDA_VISIBLE_DEVICES，或换用匹配 GPU 数的 hardware / eval_profile 预设。"
        )


def _build_merged_training_row(trow: Mapping[str, Any], ttrain: Mapping[str, Any]) -> Dict[str, Any]:
    """与 load_resolved_config 中 lr/coef/adv 合并规则一致，供子进程 payload。"""
    out = dict(ttrain)
    if "lr" in trow:
        out["lr"] = trow["lr"]
    if "coef" in trow:
        out["coef"] = trow["coef"]
    if "adv" in trow:
        out["adv"] = trow["adv"]
    return out


def build_effective_training_payload_dict(
    *,
    task_id: int,
    preset_name: str,
    training_row: Mapping[str, Any],
    eta: float,
    auxiliary: str,
    target: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "task_id": int(task_id),
        "preset_name": str(preset_name),
        "training_row": dict(training_row),
        "eta": float(eta),
        "auxiliary": str(auxiliary),
        "target": str(target),
    }


def compute_config_fingerprint(cfg_like: Mapping[str, Any]) -> str:
    """稳定指纹：用于 manifest / metrics 对齐。"""
    raw = json.dumps(cfg_like, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# training_semantic_fingerprint：task/training/hardware 等与训练主实验身份相关，不含 decode。
# generation_semantic_fingerprint：decode、评测 batch、rerank、num_return_sequences 等生成/评测语义。
# runtime_diagnostics_fingerprint 输入：仅 runtime_diagnostics_fingerprint_source()。


def _needs_decode_layer(command: str, *, step5_train_only: bool = False) -> bool:
    if command in ("eval", "eval-rerank", "eval-matrix", "eval-rerank-matrix"):
        return True
    if command == "step5":
        return not step5_train_only
    return False


def _needs_rerank_layer(command: str) -> bool:
    return command in ("eval-rerank", "eval-rerank-matrix")


def _matrix_context_from_env() -> Dict[str, Any]:
    raw = (os.environ.get("D4C_MATRIX_CONTEXT_JSON") or "").strip()
    if not raw:
        return {}
    try:
        o = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return o if isinstance(o, dict) else {}


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("d4c 需要 PyYAML：pip install pyyaml") from e
    if not path.is_file():
        raise FileNotFoundError(f"预设文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw


def _merge_task_tables() -> Dict[int, Dict[str, Any]]:
    root = _PRESETS / "tasks"
    if not root.is_dir():
        raise FileNotFoundError(f"缺少 presets/tasks 目录: {root}")
    files = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not files:
        raise FileNotFoundError(f"presets/tasks 下无 yaml: {root}")
    merged: Dict[int, Dict[str, Any]] = {}
    for path in files:
        raw = _load_yaml(path)
        if not isinstance(raw, dict):
            raise TypeError(f"{path.name} 根须为 mapping")
        for k, v in raw.items():
            tid = int(k)
            if tid < 1 or tid > 8:
                raise ValueError(f"任务号须 1..8: {k!r}")
            if not isinstance(v, dict):
                raise TypeError(f"{path.name} task {tid} 须为 dict")
            merged[tid] = {**merged.get(tid, {}), **v}
    need = set(range(1, 9))
    if set(merged.keys()) != need:
        raise ValueError(f"合并后须恰好含任务 1..8，当前: {sorted(merged.keys())}")
    return merged


def _training_row(training_blob: Mapping[Any, Any], task_id: int) -> Dict[str, Any]:
    if task_id in training_blob:
        row = training_blob[task_id]
        if isinstance(row, dict):
            return dict(row)
    # YAML 可能把键解析为 str
    if str(task_id) in training_blob:
        row = training_blob[str(task_id)]
        if isinstance(row, dict):
            return dict(row)
    raise KeyError(f"训练预设中无 task {task_id} 条目")


def _coerce_float(x: Any) -> float:
    return float(x)


def _coerce_int(x: Any) -> int:
    return int(x)


def _merge_decode_yaml(decode_preset: str) -> tuple[Dict[str, Any], str]:
    """返回 (合并后的 decode 字段 dict, decode_preset_id 用于 manifest)。"""
    base_path = _PRESETS / "decode" / "default.yaml"
    base_raw = _load_yaml(base_path)
    if not isinstance(base_raw, dict):
        raise TypeError("decode/default.yaml 根须为 mapping")
    name = (decode_preset or "").strip() or "default"
    if name == "default":
        return dict(base_raw), "default"
    overlay_path = _PRESETS / "decode" / f"{name}.yaml"
    if not overlay_path.is_file():
        raise FileNotFoundError(f"decode 预设不存在: {overlay_path}（可用名见 presets/decode/*.yaml）")
    overlay = _load_yaml(overlay_path)
    if not isinstance(overlay, dict):
        raise TypeError(f"{overlay_path.name} 根须为 mapping")
    forbidden = sorted(k for k in _DECODE_TRAINING_ONLY_FORBIDDEN_KEYS if k in overlay)
    if forbidden:
        raise ValueError(
            f"{overlay_path.name} 含 training-only 字段 {forbidden}；"
            "decode preset 禁止携带训练损失字段，请移至 presets/training/*.yaml。"
        )
    merged: Dict[str, Any] = {**base_raw, **overlay}
    return merged, name


def _parse_decode_seed(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if raw in ("", "null", "None"):
        return None
    return _coerce_int(raw)


def _parse_decode_opt_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if raw in ("", "null", "None"):
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("null", "none"):
        return None
    return _coerce_int(raw)


def _merge_rerank_yaml(rerank_preset: str) -> Dict[str, Any]:
    base_path = _PRESETS / "rerank" / "default.yaml"
    if not base_path.is_file():
        return {}
    base_raw = _load_yaml(base_path)
    if not isinstance(base_raw, dict):
        raise TypeError("rerank/default.yaml 根须为 mapping")
    name = (rerank_preset or "").strip() or "default"
    if "num_return_sequences" in base_raw:
        raise ValueError("presets/rerank/default.yaml 禁止包含 num_return_sequences；该字段只能出现在 eval_profile。")
    if name == "default":
        return dict(base_raw)
    overlay_path = _PRESETS / "rerank" / f"{name}.yaml"
    if not overlay_path.is_file():
        raise FileNotFoundError(f"rerank 预设不存在: {overlay_path}")
    overlay = _load_yaml(overlay_path)
    if not isinstance(overlay, dict):
        raise TypeError(f"{overlay_path.name} 根须为 mapping")
    if "num_return_sequences" in overlay:
        raise ValueError(
            f"rerank preset {name!r} 禁止包含 num_return_sequences；该字段只能出现在 presets/eval_profiles/*.yaml。"
        )
    return {**base_raw, **overlay}


def _eval_profile_rerank_active(raw: Any) -> bool:
    """bundle 中 rerank_preset: null / 省略 视为无 rerank。"""
    if raw is None:
        return False
    if isinstance(raw, bool) and not raw:
        return False
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none", "~", "false", "0"):
        return False
    return True


def _load_eval_profile_dict(name: str) -> Dict[str, Any]:
    stem = (name or "").strip()
    if not stem:
        raise ValueError("eval profile 名不能为空")
    path = _PRESETS / "eval_profiles" / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"eval profile 不存在: {path}（见 presets/eval_profiles/、docs/PRESETS.md）"
        )
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise TypeError(f"{path.name} 根须为 mapping")
    return dict(raw)


@dataclass(frozen=True)
class EvalProfileResolved:
    """eval_profile 编排层解析结果：引用底层 preset + profile-owned 字段，不平铺为完整配置层。"""

    stem: str
    hardware_preset: Optional[str]
    decode_preset: Optional[str]
    rerank_preset: Optional[str]
    eval_batch_size: Optional[int]
    num_return_sequences: Optional[int]


def resolve_eval_profile(stem: str) -> EvalProfileResolved:
    """
    读取 ``presets/eval_profiles/<stem>.yaml``，校验仅含允许键，返回结构化 selector 结果。
    不向主配置 dict 平铺大段字段。
    """
    raw = _load_eval_profile_dict(stem)
    extra = set(raw.keys()) - _EVAL_PROFILE_ALLOWED_KEYS
    if extra:
        raise ValueError(
            f"eval_profile {stem!r} 含非法键 {sorted(extra)!r}；仅允许: {sorted(_EVAL_PROFILE_ALLOWED_KEYS)}"
        )

    def _opt_str(k: str) -> Optional[str]:
        v = raw.get(k)
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in ("null", "none", "~"):
            return None
        return s

    def _opt_int(k: str) -> Optional[int]:
        v = raw.get(k)
        if v is None:
            return None
        return _coerce_int(v)

    return EvalProfileResolved(
        stem=(stem or "").strip(),
        hardware_preset=_opt_str("hardware_preset"),
        decode_preset=_opt_str("decode_preset"),
        rerank_preset=_opt_str("rerank_preset"),
        eval_batch_size=_opt_int("eval_batch_size"),
        num_return_sequences=_opt_int("num_return_sequences"),
    )


@dataclass(frozen=True)
class ResolvedConfig:
    command: str
    repo_root: Path
    code_dir: Path

    task_id: int
    auxiliary: str
    target: str

    preset_name: str
    run_name: Optional[str]
    from_run: Optional[str]
    step5_run: Optional[str]
    # Step4 产物目录名（…/train/step4/<step4_run>/）；step5 下由 step5_run 反推（manifest 与摘要）
    step4_run: Optional[str]
    # Step4 加载 Step3 权重的目录（仅 step4 命令下非 None）
    step3_checkpoint_dir: Optional[str]

    train_csv: Optional[str]
    model_path: Optional[str]

    learning_rate: float
    coef: float
    adv: float
    eta: float

    train_batch_size: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int
    epochs: int
    num_proc: int
    ddp_world_size: int
    seed: int

    checkpoint_dir: str
    log_dir: str
    # runs/task{T}/vN/ 根；非 metrics.json 目录（指标在 eval_run_dir/metrics.json 或 train run 内产物）
    iteration_root_dir: str
    iteration_id: str
    manifest_dir: str
    eval_run_dir: Optional[str]

    label_smoothing: float
    repetition_penalty: float
    generate_temperature: float
    generate_top_p: float

    decode_strategy: str
    decode_seed: Optional[int]
    max_explanation_length: int
    train_label_max_length: int
    no_repeat_ngram_size: Optional[int]
    min_len: Optional[int]

    # Step3：full=train+收尾 eval；train_only=仅 train；eval_only=仅 eval
    step3_mode: str
    # Step5 train：True 时向 step5 runner 传入 --train-only（与 sh 对齐）
    step5_train_only: bool

    # 实际加载的预设 stem（写入 manifest / 复现）
    hardware_preset_id: str
    decode_preset_id: str

    # eval-rerank（仅 command=eval-rerank；其它 command 下为默认值）
    num_return_sequences: int
    rerank_method: str
    rerank_top_k: int
    rerank_weight_logprob: float
    rerank_weight_length: float
    rerank_weight_repeat: float
    rerank_weight_dirty: float
    rerank_target_len_ratio: float
    export_examples_mode: str
    export_full_rerank_examples: bool
    rerank_malformed_tail_penalty: float
    rerank_malformed_token_penalty: float

    # presets/decode/*.yaml 全量 JSON（子进程 D4C_DECODE_PROFILE_JSON）
    decode_profile_json: str
    # presets/rerank/*.yaml（eval-rerank；D4C_RERANK_PROFILE_JSON）
    rerank_profile_json: str
    rerank_preset_id: str

    # torchrun 子进程：D4C_HARDWARE_PROFILE_JSON（不含 launcher-only 线程/CUDA 字段）+ 线程/CUDA 由 runners 注入 env
    hardware_profile_json: str
    omp_num_threads: int
    mkl_num_threads: int
    tokenizers_parallelism: bool
    # 编排层 runtime env：preset+CLI 为 requested；CLI > shell > preset > default 为 effective（JSON 对象）
    thread_env_requested_json: str
    thread_env_effective_json: str
    launcher_env_requested_json: str
    launcher_env_effective_json: str
    # 训练预设中的 train_batch_size（eval 命令下与 eval batch 区分，供 manifest）
    training_preset_train_batch_size: int
    # eval / eval-rerank 的全局评测 batch（与 train batch 分离）
    global_eval_batch_size: Optional[int]
    eval_per_gpu_batch_size: Optional[int]
    # presets/eval_profiles/<name>.yaml stem；未使用时为空串
    eval_profile_id: str

    # manifest / 追踪：仅记录真实加载过的层
    consumed_presets_json: str
    # CLI 覆盖前的语义快照（JSON）
    config_before_cli_json: str
    matrix_session_id: Optional[str]
    matrix_cell_id: Optional[str]
    invoked_command: str
    resolved_command_kind: str
    cell_command: Optional[str]

    # 父进程 → 子进程：训练切片唯一载体（JSON）；step4 等为空串
    effective_training_payload_json: str
    # 训练主实验身份（不含 decode / 生成语义）
    training_semantic_fingerprint: str
    # 生成/评测语义（decode、eval batch、rerank 等）；未消费生成路径时为空串
    generation_semantic_fingerprint: str
    # 运行诊断/观测开关指纹（finite_check、grad_topk 等），与训练语义指纹分离
    runtime_diagnostics_fingerprint: str
    config_field_sources_json: str
    # eval_profile selector 声明（JSON）；未使用 profile 时为 "{}" 对象字面值
    eval_profile_resolution_json: str
    # eval / eval-rerank：无 --model-path 时由 checkpoint_kind 解析 best_mainline vs last
    checkpoint_kind: str = "best_mainline"
    # 与 build_resolved_training_config 一致的 full BLEU 调度（仅含 effective payload 的命令）
    full_bleu_eval_resolved: Optional[Dict[str, Any]] = None
    # 训练期 full BLEU 监控解码策略（来自 training preset 行；默认 inherit）
    full_bleu_decode_strategy: str = "inherit"


def load_resolved_config(args: Any, command: str) -> ResolvedConfig:
    """
    从 argparse Namespace 与预设分层构造 ResolvedConfig。

    不在此读取 ``TRAIN_*`` 或 ``D4C_HARDWARE_PRESET`` 选择 YAML；矩阵追踪字段由
    ``D4C_MATRIX_CONTEXT_JSON`` 可选注入。
    """
    task_id = int(args.task)
    preset_name = str(args.preset)

    mx = _matrix_context_from_env()
    if mx:
        invoked_command = str(mx.get("invoked_command") or command)
        cell_command = mx.get("cell_command")
        if cell_command is not None:
            cell_command = str(cell_command)
        matrix_session_id = mx.get("matrix_session_id")
        if matrix_session_id is not None:
            matrix_session_id = str(matrix_session_id)
        matrix_cell_id = mx.get("matrix_cell_id")
        if matrix_cell_id is not None:
            matrix_cell_id = str(matrix_cell_id)
        resolved_command_kind = str(mx.get("resolved_command_kind") or command)
    else:
        invoked_command = command
        cell_command = None
        matrix_session_id = None
        matrix_cell_id = None
        resolved_command_kind = command

    task_table = _merge_task_tables()
    if task_id not in task_table:
        raise KeyError(f"无效 task_id={task_id}")
    trow = task_table[task_id]
    auxiliary = str(trow["auxiliary"])
    target = str(trow["target"])

    step5_train_only = command == "step5" and bool(getattr(args, "train_only", False))

    training_path = _PRESETS / "training" / f"{preset_name}.yaml"
    training_raw = _load_yaml(training_path)
    if not isinstance(training_raw, dict):
        raise TypeError(f"{training_path.name} 根须为 mapping")
    ttrain = _training_row(training_raw, task_id)
    bad_training = sorted(k for k in _TRAINING_FORBIDDEN_KEYS if k in ttrain)
    if bad_training:
        raise ValueError(
            f"training preset {preset_name!r} 含非法字段 {bad_training}；"
            "training 仅允许训练语义，eval_batch_size/num_return_sequences 必须写入 presets/eval_profiles/*.yaml。"
        )

    eval_profile_id = ""
    eb_res: Optional[EvalProfileResolved] = None
    eb_cli = getattr(args, "eval_profile", None)
    if command in _EVAL_PROFILE_COMMANDS:
        if _nonempty_str(eb_cli):
            eval_profile_id = str(eb_cli).strip()
            eb_res = resolve_eval_profile(eval_profile_id)
            if command in ("eval", "eval-matrix") and eb_res.num_return_sequences is not None:
                raise ValueError(
                    f"eval_profile {eval_profile_id!r} 含 num_return_sequences，仅 eval-rerank* 可用；"
                    "请改用 eval-rerank 或从 profile 中移除该字段。"
                )
            if command == "eval-matrix" and _eval_profile_rerank_active(eb_res.rerank_preset):
                raise ValueError(
                    f"eval profile {eval_profile_id!r} 含 rerank_preset，不能与 eval-matrix 联用；"
                    "请改用 eval-rerank-matrix，或换用不含 rerank 的 profile。"
                )
            if command == "eval" and _eval_profile_rerank_active(eb_res.rerank_preset):
                raise ValueError(
                    f"eval profile {eval_profile_id!r} 含 rerank_preset，须使用子命令 eval-rerank：\n"
                    f"  python code/d4c.py eval-rerank ... --eval-profile {eval_profile_id}"
                )
        else:
            if not _nonempty_str(getattr(args, "hardware_preset", None)):
                raise ValueError(
                    "eval / eval-matrix / eval-rerank* 须指定 --eval-profile（推荐），"
                    "或同时显式传入 --hardware-preset 与 --decode-preset。"
                )
            if not _nonempty_str(getattr(args, "decode_preset", None)):
                raise ValueError(
                    "未使用 --eval-profile 时必须同时指定 --hardware-preset 与 --decode-preset。"
                )
    elif command == "step5" and _nonempty_str(eb_cli):
        eval_profile_id = str(eb_cli).strip()
        eb_res = resolve_eval_profile(eval_profile_id)
        if eb_res.num_return_sequences is not None:
            raise ValueError(
                f"eval_profile {eval_profile_id!r} 含 num_return_sequences，仅 eval-rerank* 可用；"
                "step5 请换用不含该字段的 profile。"
            )
        if _eval_profile_rerank_active(eb_res.rerank_preset):
            raise ValueError(
                f"eval_profile {eval_profile_id!r} 含 rerank_preset，step5 不支持；请使用 eval-rerank 子命令。"
            )
    elif command == "step4":
        if not _nonempty_str(eb_cli):
            raise ValueError(
                "step4 已并入 eval 语义侧（反事实推理），必须显式提供 --eval-profile <stem>。\n"
                "推理全局 batch 仅来自 presets/eval_profiles/<stem>.yaml 的 eval_batch_size；\n"
                "禁止回退到 training preset 的 train_batch_size。\n"
                "示例: python code/d4c.py step4 --task N --preset step3 --iter v1 --from-run <run> --eval-profile eval_fast_single_gpu"
            )
        eval_profile_id = str(eb_cli).strip()
        eb_res = resolve_eval_profile(eval_profile_id)
        if eb_res.eval_batch_size is None:
            raise ValueError(
                f"step4 的 eval_profile={eval_profile_id!r} 必须声明 eval_batch_size。\n"
                "请编辑 presets/eval_profiles/<name>.yaml，为该 profile 设置 eval_batch_size。"
            )
        if eb_res.num_return_sequences is not None or _eval_profile_rerank_active(eb_res.rerank_preset):
            raise ValueError(
                f"step4 的 eval_profile={eval_profile_id!r} 仅允许 eval_batch_size / hardware_preset / decode_preset；"
                "不能包含 rerank_preset 或 num_return_sequences。"
            )

    hw_cli = getattr(args, "hardware_preset", None)
    hardware_stem = ""
    if _nonempty_str(hw_cli):
        hardware_stem = str(hw_cli).strip()
    elif eb_res and eb_res.hardware_preset:
        hardware_stem = str(eb_res.hardware_preset).strip()
    else:
        hardware_stem = "default"

    hardware_path = _PRESETS / "hardware" / f"{hardware_stem}.yaml"
    if not hardware_path.is_file():
        raise FileNotFoundError(f"hardware 预设不存在: {hardware_path}（见 presets/hardware/*.yaml）")
    runtime_raw = _load_yaml(hardware_path)
    if not isinstance(runtime_raw, dict):
        raise TypeError(f"{hardware_path.name} 根须为 mapping")
    bad_hardware = sorted(k for k in _HARDWARE_FORBIDDEN_KEYS if k in runtime_raw)
    if bad_hardware:
        raise ValueError(
            f"hardware preset {hardware_stem!r} 含非法字段 {bad_hardware}；"
            "hardware 仅允许资源并行语义，不允许 train/eval batch 或 num_return_sequences。"
        )

    consumed_presets: Dict[str, Any] = {
        "training_preset": preset_name,
        "hardware_preset": hardware_path.stem,
    }
    if eval_profile_id:
        consumed_presets["eval_profile"] = eval_profile_id

    decode_preset_id = ""
    if _needs_decode_layer(command, step5_train_only=step5_train_only):
        decode_stem = ""
        if _nonempty_str(getattr(args, "decode_preset", None)):
            decode_stem = str(args.decode_preset).strip()
        elif eb_res and eb_res.decode_preset:
            decode_stem = str(eb_res.decode_preset).strip()
        elif command == "step5":
            decode_stem = "decode_mainline_alignment_v2"
        else:
            decode_stem = str(getattr(args, "decode_preset", None) or "").strip()
        decode_raw, decode_preset_id = _merge_decode_yaml(decode_stem)
        consumed_presets["decode_preset"] = decode_preset_id
    else:
        decode_raw = dict(_DECODE_PLACEHOLDER)
        consumed_presets["decode_preset"] = None

    # --- 按链合并标量（后者覆盖前者）；CLI 覆盖在 omp 线程块之后 ---
    lr = _coerce_float(trow["lr"])
    coef = _coerce_float(trow["coef"])
    adv = _coerce_float(trow["adv"])

    if "lr" in ttrain:
        lr = _coerce_float(ttrain["lr"])
    if "coef" in ttrain:
        coef = _coerce_float(ttrain["coef"])
    if "adv" in ttrain:
        adv = _coerce_float(ttrain["adv"])

    training_preset_train_batch_size = _coerce_int(ttrain["train_batch_size"])
    train_batch_size = training_preset_train_batch_size
    epochs = _coerce_int(ttrain["epochs"])

    num_proc = _coerce_int(runtime_raw["num_proc"])
    ddp_world_size = _coerce_int(runtime_raw.get("ddp_world_size", 2))

    label_smoothing = _coerce_float(decode_raw["label_smoothing"])
    repetition_penalty = _coerce_float(decode_raw["repetition_penalty"])
    generate_temperature = _coerce_float(decode_raw["generate_temperature"])
    generate_top_p = _coerce_float(decode_raw["generate_top_p"])

    decode_strategy = str(decode_raw.get("decode_strategy", "greedy")).strip().lower()
    if decode_strategy not in ("greedy", "nucleus"):
        raise ValueError(f"decode_strategy 须为 greedy 或 nucleus，当前: {decode_strategy!r}")
    decode_seed = _parse_decode_seed(decode_raw.get("decode_seed"))
    max_explanation_length = _coerce_int(decode_raw.get("max_explanation_length", 25))
    no_repeat_ngram_size = _parse_decode_opt_int(decode_raw.get("no_repeat_ngram_size"))
    min_len = _parse_decode_opt_int(decode_raw.get("min_len"))

    train_label_max_length = int(
        max(8, min(512, int(BASE_TRAINING_DEFAULTS.train_label_max_length)))
    )
    _row_tl = ttrain.get("train_label_max_length")
    if _row_tl is not None:
        train_label_max_length = int(max(8, min(512, int(_row_tl))))

    seed = 3407
    if getattr(args, "seed", None) is not None:
        seed = _coerce_int(args.seed)

    global_eval_batch_size: Optional[int] = None
    if command in _EVAL_PROFILE_COMMANDS:
        if not eb_res or eb_res.eval_batch_size is None:
            raise ValueError(
                f"{command} 必须通过 eval_profile 提供 eval_batch_size；"
                "请在 presets/eval_profiles/<name>.yaml 中设置 eval_batch_size。"
            )
        global_eval_batch_size = int(eb_res.eval_batch_size)
    elif command == "step5":
        if (not step5_train_only) and (not eb_res or eb_res.eval_batch_size is None):
            raise ValueError(
                "step5 非 --train-only 模式必须提供 --eval-profile，且该 eval_profile 必须声明 eval_batch_size。"
            )
        if eb_res and eb_res.eval_batch_size is not None:
            global_eval_batch_size = int(eb_res.eval_batch_size)
    elif command == "step4":
        if eb_res is None or eb_res.eval_batch_size is None:
            raise RuntimeError("内部错误: step4 应在前面分支已解析 eval_profile 且含 eval_batch_size。")
        global_eval_batch_size = int(eb_res.eval_batch_size)

    def _truthy_parallelism(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("true", "1", "yes", "on")

    omp_threads = max(1, int(runtime_raw.get("omp_num_threads", 1)))
    mkl_threads = max(1, int(runtime_raw.get("mkl_num_threads", 1)))
    tok_par = _truthy_parallelism(runtime_raw.get("tokenizers_parallelism", False))

    _preset_phase = {
        "train_batch_size": train_batch_size,
        "epochs": epochs,
        "num_proc": num_proc,
        "ddp_world_size": ddp_world_size,
        "global_eval_batch_size": global_eval_batch_size,
        "seed": seed,
        "decode_preset_id": decode_preset_id or None,
        "hardware_preset_id": hardware_path.stem,
    }
    config_before_cli_json = json.dumps(_preset_phase, ensure_ascii=False, sort_keys=True, default=str)

    if getattr(args, "epochs", None) is not None and command in ("step3", "step4", "step5"):
        epochs = _coerce_int(args.epochs)
    if getattr(args, "num_proc", None) is not None:
        num_proc = _coerce_int(args.num_proc)
    if getattr(args, "ddp_world_size", None) is not None:
        ddp_world_size = _coerce_int(args.ddp_world_size)

    omp_cli_p = getattr(args, "omp_num_threads", None) is not None
    if omp_cli_p:
        omp_threads = max(1, _coerce_int(args.omp_num_threads))
    mkl_cli_p = getattr(args, "mkl_num_threads", None) is not None
    if mkl_cli_p:
        mkl_threads = max(1, _coerce_int(args.mkl_num_threads))
    tok_cli_p = getattr(args, "tokenizers_parallelism", None) is not None
    if tok_cli_p:
        tok_par = str(args.tokenizers_parallelism).strip().lower() == "true"

    step3_mode = "full"
    if command == "step3":
        ev = bool(getattr(args, "eval_only", False))
        tr = bool(getattr(args, "train_only", False))
        if ev and tr:
            raise ValueError("step3 不能同时指定 --eval-only 与 --train-only")
        if ev:
            step3_mode = "eval_only"
        elif tr:
            step3_mode = "train_only"

    # Step5 反事实权重：与历史 shell 一致，默认取合并后的 adv；可由 CLI --eta 覆盖（若存在）
    eta = adv
    if getattr(args, "eta", None) is not None:
        eta = _coerce_float(args.eta)

    root = _CODE_DIR.parent

    iter_raw = getattr(args, "iteration_id", None) or "v1"
    iteration_id = run_naming.normalize_iteration_id(str(iter_raw))
    run_id_cli = getattr(args, "run_id", None)
    run_id_req = (
        None
        if run_id_cli is None or str(run_id_cli).strip().lower() in ("", "auto")
        else str(run_id_cli).strip()
    )
    run_name: Optional[str] = getattr(args, "run_name", None)
    from_run: Optional[str] = getattr(args, "from_run", None)
    step5_run: Optional[str] = getattr(args, "step5_run", None)
    train_csv_arg: Optional[str] = getattr(args, "train_csv", None)
    model_path_arg: Optional[str] = getattr(args, "model_path", None)
    if model_path_arg:
        model_path_arg = str(Path(model_path_arg).expanduser().resolve())
    checkpoint_kind = str(getattr(args, "checkpoint_kind", "best_mainline") or "best_mainline")
    if checkpoint_kind not in ("best_mainline", "last"):
        checkpoint_kind = "best_mainline"

    iteration_root = path_layout.get_iteration_root(root, task_id, iteration_id)
    meta_dir = path_layout.get_task_meta_dir(root, task_id, iteration_id)
    meta_dir.mkdir(parents=True, exist_ok=True)
    _iter_json = meta_dir / "iteration.json"
    if not _iter_json.is_file():
        _iter_json.write_text(
            json.dumps(
                {
                    "iteration_id": iteration_id,
                    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    eval_run_dir_out: Optional[str] = None
    manifest_dir: str
    ck: Path
    lg: Path
    step4_run_out: Optional[str] = None
    step3_checkpoint_dir_out: Optional[str] = None

    if command == "step3":
        t3_parent = iteration_root / "train" / "step3"
        t3_parent.mkdir(parents=True, exist_ok=True)
        req = run_id_req if run_id_req is not None else (str(run_name).strip() if run_name else None)
        step3_rid = run_naming.allocate_child_dir(t3_parent, requested=req, kind="run")
        ck = path_layout.get_train_step3_run_root(root, task_id, iteration_id, step3_rid)
        ck.mkdir(parents=True, exist_ok=True)
        lg = path_layout.logs_dir(ck)
        lg.mkdir(parents=True, exist_ok=True)
        run_name = step3_rid
        from_run = None
        step5_run = None
        manifest_dir = str(ck.resolve())
    elif command == "step4":
        if not from_run:
            raise ValueError(
                "step4 须指定 --from-run（Step3 的 run 目录名，与 runs/.../train/step3/<run>/ 一致）。\n"
                "说明见 docs/D4C_Scripts_and_Runtime_Guide.md。"
            )
        step3_rid = run_naming.parse_run_id(from_run)
        step3_ck = path_layout.get_train_step3_run_root(root, task_id, iteration_id, step3_rid)
        t4_parent = iteration_root / "train" / "step4"
        t4_parent.mkdir(parents=True, exist_ok=True)
        s4_cli = getattr(args, "step4_run", None)
        s4_req = (
            None
            if s4_cli is None or str(s4_cli).strip().lower() in ("", "auto")
            else str(s4_cli).strip()
        )
        if not s4_req:
            step4_rid = run_naming.allocate_step4_run_id(t4_parent, step3_rid)
        else:
            step4_rid = run_naming.parse_run_id(s4_req)
            t4_target = t4_parent / step4_rid
            if t4_target.exists():
                raise FileExistsError(f"已存在目录（禁止覆盖）: {t4_target}")
        ck = path_layout.get_train_step4_run_root(root, task_id, iteration_id, step4_rid)
        ck.mkdir(parents=True, exist_ok=True)
        lg = path_layout.logs_dir(ck)
        lg.mkdir(parents=True, exist_ok=True)
        manifest_dir = str(ck.resolve())
        step4_run_out = step4_rid
        step3_checkpoint_dir_out = str(step3_ck.resolve())
    elif command == "step5":
        if not from_run:
            raise ValueError("step5 须指定 --from-run（Step3 的 run 目录名）。")
        step3_rid = run_naming.parse_run_id(from_run)
        s4_cli = getattr(args, "step4_run", None)
        step4_for_allocate: Optional[str] = None
        if s4_cli is not None and str(s4_cli).strip() != "":
            step4_for_allocate = run_naming.parse_run_id(str(s4_cli).strip())
        t5_parent = iteration_root / "train" / "step5"
        t5_parent.mkdir(parents=True, exist_ok=True)
        s5_raw = step5_run
        s5_req = (
            None
            if s5_raw is None or str(s5_raw).strip().lower() in ("", "auto")
            else str(s5_raw).strip()
        )
        if not s5_req:
            if step4_for_allocate is not None:
                step5_rid = run_naming.allocate_step5_run_id(t5_parent, step4_for_allocate)
            else:
                raise ValueError(
                    "step5 使用 --step5-run auto 时必须指定 --step4-run（用于分配 train/step5/{step4}_n）。\n"
                    "若已确定 Step5 目录名，请显式传入 --step5-run（如 2_1_1）；"
                    "训练 CSV 固定为 train/step4/{去掉末段}/factuals_counterfactuals.csv。"
                )
        else:
            step5_rid = run_naming.allocate_child_dir(t5_parent, requested=s5_req, kind="run")
        step5_run = step5_rid
        if s4_cli is not None and str(s4_cli).strip() != "":
            explicit_s4 = run_naming.parse_run_id(str(s4_cli).strip())
            inferred_s4 = run_naming.step4_slug_from_step5_slug(step5_rid)
            if explicit_s4 != inferred_s4:
                raise ValueError(
                    "step5 的 --step4-run 与由 --step5-run 反推的 Step4 slug 不一致。\n"
                    f"  你指定 --step4-run={explicit_s4!r}\n"
                    f"  由 step5_run={step5_rid!r} 反推应为 train/step4/{inferred_s4}/（规则见 "
                    "code/d4c_core/run_naming.py:step4_slug_from_step5_slug）\n"
                    "请修正参数：要么省略 --step4-run（在 --step5-run auto 时仅用其分配），"
                    "要么使二者一致。"
                )
        step4_run_out = run_naming.step4_slug_from_step5_slug(step5_rid)
        ck = path_layout.get_train_step5_run_root(root, task_id, iteration_id, step5_rid)
        ck.mkdir(parents=True, exist_ok=True)
        lg = path_layout.logs_dir(ck)
        lg.mkdir(parents=True, exist_ok=True)
        manifest_dir = str(ck.resolve())
        # 训练后内置 valid 评估需要导出 metrics/predictions；与独立 eval 命令一致写入 D4C_EVAL_RUN_DIR。
        if not step5_train_only:
            _pe = ck / "post_train_eval"
            _pe.mkdir(parents=True, exist_ok=True)
            eval_run_dir_out = str(_pe.resolve())
    elif command in ("eval", "eval-rerank"):
        if model_path_arg:
            mp = Path(model_path_arg)
            if not mp.is_file():
                raise FileNotFoundError(f"评测权重不存在: {mp}")
            if mp.parent.name != "model":
                raise ValueError(
                    "新版布局要求权重位于 .../train/step5/<run>/model/best_mainline.pth 或 last.pth；"
                    "或使用 --from-run + --step5-run + --iter 解析标准路径。"
                )
            ck = mp.parent.parent.resolve()
            if ck.parent.name == "step5":
                try:
                    s5 = run_naming.parse_run_id(ck.name)
                    step5_run = s5
                    step4_run_out = run_naming.step4_slug_from_step5_slug(s5)
                except ValueError:
                    pass
            lg_base_parent = iteration_root / ("rerank" if command == "eval-rerank" else "eval")
            lg_base_parent.mkdir(parents=True, exist_ok=True)
            eval_rid = run_naming.allocate_child_dir(lg_base_parent, requested=run_id_req, kind="run")
            if command == "eval-rerank":
                ead = path_layout.get_rerank_run_root(root, task_id, iteration_id, eval_rid)
            else:
                ead = path_layout.get_eval_run_root(root, task_id, iteration_id, eval_rid)
            ead.mkdir(parents=True, exist_ok=True)
            lg = path_layout.logs_dir(ead)
            lg.mkdir(parents=True, exist_ok=True)
            eval_run_dir_out = str(ead.resolve())
            manifest_dir = eval_run_dir_out
        else:
            if not from_run or not step5_run:
                raise ValueError(
                    "eval / eval-rerank 须指定 --model-path，或同时指定 --from-run 与 --step5-run（均为 run 目录名）。"
                )
            step3_rid = run_naming.parse_run_id(from_run)
            step5_rid = run_naming.parse_run_id(step5_run)
            step4_run_out = run_naming.step4_slug_from_step5_slug(step5_rid)
            ck = path_layout.get_train_step5_run_root(root, task_id, iteration_id, step5_rid)
            lg_base_parent = iteration_root / ("rerank" if command == "eval-rerank" else "eval")
            lg_base_parent.mkdir(parents=True, exist_ok=True)
            eval_rid = run_naming.allocate_child_dir(lg_base_parent, requested=run_id_req, kind="run")
            if command == "eval-rerank":
                ead = path_layout.get_rerank_run_root(root, task_id, iteration_id, eval_rid)
            else:
                ead = path_layout.get_eval_run_root(root, task_id, iteration_id, eval_rid)
            ead.mkdir(parents=True, exist_ok=True)
            lg = path_layout.logs_dir(ead)
            lg.mkdir(parents=True, exist_ok=True)
            eval_run_dir_out = str(ead.resolve())
            manifest_dir = eval_run_dir_out
    else:
        raise ValueError(f"未知 command: {command}")

    metrics = iteration_root

    _assert_visible_gpus_for_ddp(ddp_world_size)

    rerank_profile_json = "{}"
    rerank_preset_id = ""
    rmerged_early: Dict[str, Any] = {}
    if _needs_rerank_layer(command):
        rp_stem = getattr(args, "rerank_preset", None)
        if rp_stem is not None and str(rp_stem).strip():
            rp_name = str(rp_stem).strip()
        elif eb_res and _eval_profile_rerank_active(eb_res.rerank_preset):
            rp_name = str(eb_res.rerank_preset).strip()
        else:
            rp_name = "rerank_v3_default"
        rerank_preset_id = rp_name
        rmerged_early = _merge_rerank_yaml(rp_name)
        consumed_presets["rerank_preset"] = rerank_preset_id
        _rj_subproc = {k: v for k, v in rmerged_early.items() if k not in _RERANK_YAML_ORCHESTRATION_KEYS}
        rerank_profile_json = json.dumps(_rj_subproc, ensure_ascii=False, sort_keys=True, default=str)

    num_return_sequences = 3
    rerank_method = "rule_v3"
    rerank_top_k = 1
    rerank_weight_logprob = 0.35
    rerank_weight_length = 0.10
    rerank_weight_repeat = 0.16
    rerank_weight_dirty = 0.20
    rerank_target_len_ratio = 1.10
    export_examples_mode = "head50"
    export_full_rerank_examples = False
    rerank_malformed_tail_penalty = 0.15
    rerank_malformed_token_penalty = 0.18
    if command == "eval-rerank":
        ov = rmerged_early
        if "num_return_sequences" in ov:
            num_return_sequences = max(1, _coerce_int(ov["num_return_sequences"]))
        if "rerank_method" in ov and str(ov["rerank_method"]).strip():
            rerank_method = str(ov["rerank_method"]).strip()
        if "export_examples_mode" in ov and str(ov["export_examples_mode"]).strip():
            export_examples_mode = str(ov["export_examples_mode"]).strip().lower()
        if "rerank_top_k" in ov:
            rerank_top_k = max(1, _coerce_int(ov["rerank_top_k"]))
        if "rerank_weight_logprob" in ov:
            rerank_weight_logprob = _coerce_float(ov["rerank_weight_logprob"])
        if "rerank_weight_length" in ov:
            rerank_weight_length = _coerce_float(ov["rerank_weight_length"])
        if "rerank_weight_repeat" in ov:
            rerank_weight_repeat = _coerce_float(ov["rerank_weight_repeat"])
        if "rerank_weight_dirty" in ov:
            rerank_weight_dirty = _coerce_float(ov["rerank_weight_dirty"])
        if "rerank_target_len_ratio" in ov:
            rerank_target_len_ratio = _coerce_float(ov["rerank_target_len_ratio"])
        if "rerank_malformed_tail_penalty" in ov:
            rerank_malformed_tail_penalty = _coerce_float(ov["rerank_malformed_tail_penalty"])
        if "rerank_malformed_token_penalty" in ov:
            rerank_malformed_token_penalty = _coerce_float(ov["rerank_malformed_token_penalty"])
        if export_examples_mode == "full":
            export_full_rerank_examples = True
        if eb_res and eb_res.num_return_sequences is not None:
            num_return_sequences = max(1, _coerce_int(eb_res.num_return_sequences))

    if _needs_decode_layer(command, step5_train_only=step5_train_only):
        decode_profile_json = json.dumps(decode_raw, ensure_ascii=False, sort_keys=True, default=str)
    else:
        decode_profile_json = json.dumps(dict(_DECODE_PLACEHOLDER), ensure_ascii=False, sort_keys=True, default=str)

    consumed_presets_json = json.dumps(consumed_presets, ensure_ascii=False, sort_keys=True, default=str)

    eval_profile_resolution_json = "{}"
    if eb_res:
        _owned: Dict[str, Any] = {}
        if eb_res.eval_batch_size is not None:
            _owned["eval_batch_size"] = eb_res.eval_batch_size
        if eb_res.num_return_sequences is not None:
            _owned["num_return_sequences"] = eb_res.num_return_sequences
        eval_profile_resolution_json = json.dumps(
            {
                "eval_profile": eb_res.stem,
                "selector_hardware_preset": eb_res.hardware_preset,
                "selector_decode_preset": eb_res.decode_preset,
                "selector_rerank_preset": eb_res.rerank_preset,
                "bundle_owned": _owned,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    cuda_preset = _normalize_cuda_visible_devices_yaml(runtime_raw.get("cuda_visible_devices"))
    cuda_cli_p = _nonempty_str(getattr(args, "cuda_visible_devices", None))
    cuda_req = (
        _normalize_cuda_visible_devices_yaml(str(getattr(args, "cuda_visible_devices")).strip())
        if cuda_cli_p
        else cuda_preset
    )
    shell_cuda = _normalize_cuda_visible_devices_yaml(os.environ.get("CUDA_VISIBLE_DEVICES"))
    cuda_eff = (
        _normalize_cuda_visible_devices_yaml(str(getattr(args, "cuda_visible_devices")).strip())
        if cuda_cli_p
        else (shell_cuda or cuda_preset)
    )

    omp_effective = omp_threads if omp_cli_p else (_shell_positive_int("OMP_NUM_THREADS") or omp_threads)
    mkl_effective = mkl_threads if mkl_cli_p else (_shell_positive_int("MKL_NUM_THREADS") or mkl_threads)
    _tok_shell = _shell_tokenizers_parallelism()
    tok_effective = tok_par if tok_cli_p else (_tok_shell if _tok_shell is not None else tok_par)

    thread_env_requested = {
        "OMP_NUM_THREADS": str(int(omp_threads)),
        "MKL_NUM_THREADS": str(int(mkl_threads)),
        "TOKENIZERS_PARALLELISM": "true" if tok_par else "false",
    }
    thread_env_effective = {
        "OMP_NUM_THREADS": str(int(omp_effective)),
        "MKL_NUM_THREADS": str(int(mkl_effective)),
        "TOKENIZERS_PARALLELISM": "true" if tok_effective else "false",
    }
    launcher_env_requested: Dict[str, str] = {}
    if cuda_req:
        launcher_env_requested["CUDA_VISIBLE_DEVICES"] = cuda_req
    launcher_env_effective: Dict[str, str] = {}
    if cuda_eff:
        launcher_env_effective["CUDA_VISIBLE_DEVICES"] = cuda_eff

    thread_env_requested_json = json.dumps(thread_env_requested, ensure_ascii=False, sort_keys=True)
    thread_env_effective_json = json.dumps(thread_env_effective, ensure_ascii=False, sort_keys=True)
    launcher_env_requested_json = json.dumps(launcher_env_requested, ensure_ascii=False, sort_keys=True)
    launcher_env_effective_json = json.dumps(launcher_env_effective, ensure_ascii=False, sort_keys=True)

    _export_rt: Dict[str, Any] = {str(k): v for k, v in runtime_raw.items()}
    _export_rt["num_proc"] = int(num_proc)
    _export_rt["ddp_world_size"] = int(ddp_world_size)
    _semantic_hw = {
        k: v for k, v in _export_rt.items() if k not in _RUNTIME_LAUNCHER_HARDWARE_KEYS
    }
    _hardware_profile_json = json.dumps(_semantic_hw, ensure_ascii=False, sort_keys=True, default=str)

    row_for_payload = _build_merged_training_row(trow, ttrain)
    if global_eval_batch_size is not None and command in ("eval", "eval-rerank"):
        row_for_payload = {**row_for_payload, "eval_batch_size": int(global_eval_batch_size)}
    elif global_eval_batch_size is not None and command == "step5" and not step5_train_only:
        row_for_payload = {**row_for_payload, "eval_batch_size": int(global_eval_batch_size)}

    row_tid: Dict[str, Any] = {k: v for k, v in row_for_payload.items() if k != "eval_batch_size"}
    G_res, P_res, A_res, eff_res = resolve_train_batch_from_training_row(row_tid, ddp_world_size)
    eval_per_gpu_res: Optional[int] = None
    if global_eval_batch_size is not None:
        _, eval_per_gpu_res = resolve_eval_batch_layout(global_eval_batch_size, ddp_world_size)

    full_bleu_decode_strategy_resolved = "inherit"
    if "full_bleu_decode_strategy" in row_for_payload:
        full_bleu_decode_strategy_resolved = parse_full_bleu_decode_strategy(
            row_for_payload["full_bleu_decode_strategy"],
            ctx="training_preset.full_bleu_decode_strategy",
        )

    effective_training_payload_json = ""
    training_semantic_fingerprint = ""
    generation_semantic_fingerprint = ""
    runtime_diagnostics_fingerprint = compute_config_fingerprint(runtime_diagnostics_fingerprint_source())
    config_field_sources_json = "{}"
    full_bleu_eval_resolved: Optional[Dict[str, Any]] = None
    if command in _EFFECTIVE_TRAINING_PAYLOAD_COMMANDS:
        _ep_obj = build_effective_training_payload_dict(
            task_id=task_id,
            preset_name=preset_name,
            training_row=row_for_payload,
            eta=float(eta),
            auxiliary=auxiliary,
            target=target,
        )
        effective_training_payload_json = json.dumps(_ep_obj, ensure_ascii=False, sort_keys=True, default=str)
        full_bleu_eval_resolved = resolve_full_bleu_eval_from_training_row(row_for_payload).as_dict()
        row_train_identity = {k: v for k, v in row_for_payload.items() if k != "eval_batch_size"}
        _ep_train_identity = build_effective_training_payload_dict(
            task_id=task_id,
            preset_name=preset_name,
            training_row=row_train_identity,
            eta=float(eta),
            auxiliary=auxiliary,
            target=target,
        )
        _src_map: Dict[str, Any] = {
            "training_preset": preset_name,
            "hardware_preset": hardware_path.stem,
            "ddp_world_size": f"hardware:{hardware_path.stem}",
            "num_proc": f"hardware:{hardware_path.stem}",
            "train_batch_size": f"training_preset:{preset_name}",
            "epochs": f"training_preset:{preset_name}",
            "train_label_max_length": f"training_preset:{preset_name}",
            "train_dynamic_padding": f"training_preset:{preset_name}",
            "loss_weight_repeat_ul": f"training_preset:{preset_name}",
            "loss_weight_terminal_clean": f"training_preset:{preset_name}",
            "terminal_clean_span": f"training_preset:{preset_name}",
            "full_bleu_eval": f"training_preset:{preset_name}",
            "full_bleu_decode_strategy": f"training_preset:{preset_name}",
        }
        if eval_profile_id:
            _src_map["eval_profile_orchestrator"] = eval_profile_id
        if decode_preset_id:
            if _nonempty_str(getattr(args, "decode_preset", None)):
                _src_map["decode_preset"] = "cli"
            elif eb_res and eb_res.decode_preset:
                _src_map["decode_preset"] = f"eval_profile_selector:{eval_profile_id}"
            else:
                _src_map["decode_preset"] = f"decode_preset_file:{decode_preset_id}"
        if command == "eval-rerank":
            if _nonempty_str(getattr(args, "rerank_preset", None)):
                _src_map["rerank_preset"] = "cli"
            elif eb_res and _eval_profile_rerank_active(eb_res.rerank_preset):
                _src_map["rerank_preset"] = f"eval_profile_selector:{eval_profile_id}"
            else:
                _src_map["rerank_preset"] = f"rerank_preset_file:{rerank_preset_id}"
        if global_eval_batch_size is not None:
            if getattr(args, "eval_batch_size", None) is not None:
                _src_map["eval_batch_size"] = "cli"
            elif eb_res and eb_res.eval_batch_size is not None:
                _src_map["eval_batch_size"] = f"eval_profile_selector:{eval_profile_id}"
            else:
                _src_map["eval_batch_size"] = f"training_preset:{preset_name}"
        config_field_sources_json = json.dumps(_src_map, ensure_ascii=False, sort_keys=True)
        _fp_train: Dict[str, Any] = {
            "training_payload": _ep_train_identity,
            "hardware_profile": dict(_semantic_hw),
            "ddp_world_size": int(ddp_world_size),
            "train_label_max_length": int(train_label_max_length),
        }
        training_semantic_fingerprint = compute_config_fingerprint(_fp_train)

        _fp_gen: Dict[str, Any] = {}
        if decode_preset_id and _needs_decode_layer(command, step5_train_only=step5_train_only):
            _fp_gen["decode_sha1"] = hashlib.sha1(decode_profile_json.encode("utf-8")).hexdigest()[:24]
        if command in _EVAL_PROFILE_COMMANDS and global_eval_batch_size is not None:
            _fp_gen["eval_batch_size"] = int(global_eval_batch_size)
        elif command == "step5" and (not step5_train_only) and global_eval_batch_size is not None:
            _fp_gen["eval_batch_size"] = int(global_eval_batch_size)
        elif command == "step4" and global_eval_batch_size is not None:
            _fp_gen["eval_batch_size"] = int(global_eval_batch_size)
        if _needs_rerank_layer(command) and rerank_preset_id:
            _fp_gen["rerank_sha1"] = hashlib.sha1(rerank_profile_json.encode("utf-8")).hexdigest()[:24]
        if command == "eval-rerank":
            _fp_gen["num_return_sequences"] = int(num_return_sequences)
        if _fp_gen:
            generation_semantic_fingerprint = compute_config_fingerprint(_fp_gen)
        runtime_diagnostics_fingerprint = compute_config_fingerprint(
            runtime_diagnostics_fingerprint_source()
        )

    return ResolvedConfig(
        command=command,
        repo_root=root,
        code_dir=_CODE_DIR,
        task_id=task_id,
        auxiliary=auxiliary,
        target=target,
        preset_name=preset_name,
        run_name=run_name,
        from_run=from_run,
        step5_run=step5_run,
        step4_run=step4_run_out,
        step3_checkpoint_dir=step3_checkpoint_dir_out,
        train_csv=train_csv_arg,
        model_path=model_path_arg,
        learning_rate=lr,
        coef=coef,
        adv=adv,
        eta=eta,
        train_batch_size=G_res,
        per_device_train_batch_size=P_res,
        gradient_accumulation_steps=A_res,
        effective_global_batch_size=eff_res,
        epochs=epochs,
        num_proc=num_proc,
        ddp_world_size=ddp_world_size,
        seed=seed,
        checkpoint_dir=str(ck.resolve()),
        log_dir=str(lg.resolve()),
        iteration_root_dir=str(metrics.resolve()),
        iteration_id=iteration_id,
        manifest_dir=manifest_dir,
        eval_run_dir=eval_run_dir_out,
        label_smoothing=label_smoothing,
        repetition_penalty=repetition_penalty,
        generate_temperature=generate_temperature,
        generate_top_p=generate_top_p,
        decode_strategy=decode_strategy,
        decode_seed=decode_seed,
        max_explanation_length=max_explanation_length,
        train_label_max_length=train_label_max_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_len=min_len,
        step3_mode=step3_mode,
        step5_train_only=step5_train_only,
        hardware_preset_id=hardware_path.stem,
        decode_preset_id=decode_preset_id,
        num_return_sequences=num_return_sequences,
        rerank_method=rerank_method,
        rerank_top_k=rerank_top_k,
        rerank_weight_logprob=rerank_weight_logprob,
        rerank_weight_length=rerank_weight_length,
        rerank_weight_repeat=rerank_weight_repeat,
        rerank_weight_dirty=rerank_weight_dirty,
        rerank_target_len_ratio=rerank_target_len_ratio,
        export_examples_mode=export_examples_mode,
        export_full_rerank_examples=export_full_rerank_examples,
        rerank_malformed_tail_penalty=rerank_malformed_tail_penalty,
        rerank_malformed_token_penalty=rerank_malformed_token_penalty,
        decode_profile_json=decode_profile_json,
        rerank_profile_json=rerank_profile_json,
        rerank_preset_id=rerank_preset_id,
        hardware_profile_json=_hardware_profile_json,
        omp_num_threads=int(omp_effective),
        mkl_num_threads=int(mkl_effective),
        tokenizers_parallelism=bool(tok_effective),
        thread_env_requested_json=thread_env_requested_json,
        thread_env_effective_json=thread_env_effective_json,
        launcher_env_requested_json=launcher_env_requested_json,
        launcher_env_effective_json=launcher_env_effective_json,
        training_preset_train_batch_size=int(training_preset_train_batch_size),
        global_eval_batch_size=global_eval_batch_size,
        eval_per_gpu_batch_size=eval_per_gpu_res,
        eval_profile_id=eval_profile_id,
        consumed_presets_json=consumed_presets_json,
        config_before_cli_json=config_before_cli_json,
        matrix_session_id=matrix_session_id,
        matrix_cell_id=matrix_cell_id,
        invoked_command=invoked_command,
        resolved_command_kind=resolved_command_kind,
        cell_command=cell_command,
        effective_training_payload_json=effective_training_payload_json,
        training_semantic_fingerprint=training_semantic_fingerprint,
        generation_semantic_fingerprint=generation_semantic_fingerprint,
        runtime_diagnostics_fingerprint=runtime_diagnostics_fingerprint,
        config_field_sources_json=config_field_sources_json,
        eval_profile_resolution_json=eval_profile_resolution_json,
        checkpoint_kind=checkpoint_kind,
        full_bleu_eval_resolved=full_bleu_eval_resolved,
        full_bleu_decode_strategy=full_bleu_decode_strategy_resolved,
    )
