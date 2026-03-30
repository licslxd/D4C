"""
MAINLINE 支撑模块 — 供 ``d4c.py`` 解析 ResolvedConfig；不单独作为用户入口。

配置解析：固定顺序合并，不读取用户环境变量。
task preset → training preset → runtime preset → decode（default.yaml 与 --decode-preset 叠加）→ CLI override
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from d4c_core import paths as d4c_paths

_CODE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CODE_DIR.parent
_PRESETS = _REPO_ROOT / "presets"


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
    merged: Dict[str, Any] = {**base_raw, **overlay}
    return merged, name


def _parse_decode_seed(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if raw in ("", "null", "None"):
        return None
    return _coerce_int(raw)


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

    train_csv: Optional[str]
    model_path: Optional[str]

    learning_rate: float
    coef: float
    adv: float
    eta: float

    batch_size: int
    epochs: int
    num_proc: int
    ddp_world_size: int
    seed: int

    checkpoint_dir: str
    log_dir: str
    metrics_dir: str

    label_smoothing: float
    repetition_penalty: float
    generate_temperature: float
    generate_top_p: float

    decode_strategy: str
    decode_seed: Optional[int]
    max_explanation_length: int

    # Step3：full=train+收尾 eval；train_only=仅 train；eval_only=仅 eval（与 sh/run_step3_optimized 对齐）
    step3_mode: str
    # Step5 train：True 时向 step5 runner 传入 --train-only（与 sh 对齐）
    step5_train_only: bool

    # 实际加载的预设文件 stem（写入 manifest / 复现）
    runtime_preset_id: str
    decode_preset_id: str


def _default_step3_run_name() -> str:
    return f"step3_opt_{datetime.now().strftime('%Y%m%d_%H%M')}"


def _default_step5_run_name() -> str:
    return f"step5_opt_{datetime.now().strftime('%Y%m%d_%H%M')}"


def load_resolved_config(args: Any, command: str) -> ResolvedConfig:
    """
    从 argparse Namespace 与固定预设文件构造 ResolvedConfig。
    合并链不读取 TRAIN_* 等训练侧环境变量；可选读取 D4C_RUNTIME_PRESET 以选择 presets/runtime/<name>.yaml。
    """
    task_id = int(args.task)
    preset_name = str(args.preset)

    task_table = _merge_task_tables()
    if task_id not in task_table:
        raise KeyError(f"无效 task_id={task_id}")
    trow = task_table[task_id]
    auxiliary = str(trow["auxiliary"])
    target = str(trow["target"])

    training_path = _PRESETS / "training" / f"{preset_name}.yaml"
    training_raw = _load_yaml(training_path)
    if not isinstance(training_raw, dict):
        raise TypeError(f"{training_path.name} 根须为 mapping")
    ttrain = _training_row(training_raw, task_id)

    runtime_name = (os.environ.get("D4C_RUNTIME_PRESET") or "").strip()
    runtime_path = _PRESETS / "runtime" / "default.yaml"
    if runtime_name:
        cand = _PRESETS / "runtime" / f"{runtime_name}.yaml"
        if cand.is_file():
            runtime_path = cand
    runtime_raw = _load_yaml(runtime_path)
    if not isinstance(runtime_raw, dict):
        raise TypeError(f"{runtime_path.name} 根须为 mapping")

    decode_preset_cli = getattr(args, "decode_preset", "default")
    decode_raw, decode_preset_id = _merge_decode_yaml(str(decode_preset_cli))

    # --- 按链合并标量（后者覆盖前者）---
    lr = _coerce_float(trow["lr"])
    coef = _coerce_float(trow["coef"])
    adv = _coerce_float(trow["adv"])

    if "lr" in ttrain:
        lr = _coerce_float(ttrain["lr"])
    if "coef" in ttrain:
        coef = _coerce_float(ttrain["coef"])
    if "adv" in ttrain:
        adv = _coerce_float(ttrain["adv"])

    batch_size = _coerce_int(ttrain["train_batch_size"])
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

    seed = 3407

    # CLI override（仅非 None）
    if getattr(args, "batch_size", None) is not None:
        batch_size = _coerce_int(args.batch_size)
    if getattr(args, "epochs", None) is not None:
        epochs = _coerce_int(args.epochs)
    if getattr(args, "num_proc", None) is not None:
        num_proc = _coerce_int(args.num_proc)
    if getattr(args, "seed", None) is not None:
        seed = _coerce_int(args.seed)
    if getattr(args, "ddp_world_size", None) is not None:
        ddp_world_size = _coerce_int(args.ddp_world_size)

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

    step5_train_only = command == "step5" and bool(getattr(args, "train_only", False))

    # Step5 反事实权重：与历史 shell 一致，默认取合并后的 adv；可由 CLI --eta 覆盖（若存在）
    eta = adv
    if getattr(args, "eta", None) is not None:
        eta = _coerce_float(args.eta)

    root = d4c_paths.repo_root_from_code_dir(_CODE_DIR)

    run_name: Optional[str] = getattr(args, "run_name", None)
    from_run: Optional[str] = getattr(args, "from_run", None)
    step5_run: Optional[str] = getattr(args, "step5_run", None)
    train_csv_arg: Optional[str] = getattr(args, "train_csv", None)
    model_path_arg: Optional[str] = getattr(args, "model_path", None)
    if model_path_arg:
        model_path_arg = str(Path(model_path_arg).expanduser().resolve())

    if command == "step3":
        eff_run = run_name or _default_step3_run_name()
        ck = d4c_paths.resolve_step3_dir(root, task_id, eff_run)
        lg = d4c_paths.resolve_step3_log_dir(root, task_id, eff_run)
        run_name = eff_run
        from_run = None
        step5_run = None
    elif command == "step4":
        if not from_run:
            raise ValueError(
                "step4 须指定 --from-run（与 Step3 产出的 checkpoint 子目录名一致，例如 step3_opt_YYYYMMDD_HHMM）。\n"
                "下一步: 查 Step3 日志或 checkpoints/<task>/step3_optimized/ 下目录名；"
                "说明见 docs/D4C_Scripts_and_Runtime_Guide.md §1.2；配置合并见 docs/PRESETS.md。"
            )
        ck = d4c_paths.resolve_step3_dir(root, task_id, from_run)
        lg = d4c_paths.resolve_step4_log_dir(root, task_id, from_run)
    elif command == "step5":
        if not from_run:
            raise ValueError(
                "step5 须指定 --from-run（Step3 的 run 目录名）。\n"
                "另须 --step5-run（可省略则自动生成）。详见 README 快速开始与 docs/PRESETS.md。"
            )
        s5 = step5_run or _default_step5_run_name()
        step5_run = s5
        ck = d4c_paths.resolve_step5_dir(root, task_id, from_run, s5)
        lg = d4c_paths.resolve_step5_log_dir(root, task_id, s5)
    elif command == "eval":
        if model_path_arg:
            mp = Path(model_path_arg).expanduser().resolve()
            ck = mp.parent
            lg = root / "log" / str(task_id) / "eval" / mp.stem
        else:
            if not from_run:
                raise ValueError(
                    "eval 须指定 --model-path，或同时指定 --from-run 与 --step5-run。\n"
                    "说明: README「Eval」；路径约定见 docs/D4C_Scripts_and_Runtime_Guide.md。"
                )
            if not step5_run:
                raise ValueError(
                    "eval 在不用 --model-path 时必须带 --step5-run。\n"
                    "若只有权重文件路径，请改用 --model-path <path/to/model.pth>。"
                )
            s5 = step5_run
            ck = d4c_paths.resolve_step5_dir(root, task_id, from_run, s5)
            lg = d4c_paths.resolve_step5_log_dir(root, task_id, s5)
    else:
        raise ValueError(f"未知 command: {command}")

    metrics = d4c_paths.resolve_metrics_dir(ck)

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
        train_csv=train_csv_arg,
        model_path=model_path_arg,
        learning_rate=lr,
        coef=coef,
        adv=adv,
        eta=eta,
        batch_size=batch_size,
        epochs=epochs,
        num_proc=num_proc,
        ddp_world_size=ddp_world_size,
        seed=seed,
        checkpoint_dir=str(ck.resolve()),
        log_dir=str(lg.resolve()),
        metrics_dir=str(metrics.resolve()),
        label_smoothing=label_smoothing,
        repetition_penalty=repetition_penalty,
        generate_temperature=generate_temperature,
        generate_top_p=generate_top_p,
        decode_strategy=decode_strategy,
        decode_seed=decode_seed,
        max_explanation_length=max_explanation_length,
        step3_mode=step3_mode,
        step5_train_only=step5_train_only,
        runtime_preset_id=runtime_path.stem,
        decode_preset_id=decode_preset_id,
    )
