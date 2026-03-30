"""显式路径解析：替代 D4C_CHECKPOINT_* 隐式约定（由 d4c 入口注入）。"""
from __future__ import annotations

from pathlib import Path


def repo_root_from_code_dir(code_dir: Path) -> Path:
    return code_dir.resolve().parent


def resolve_step3_dir(root: Path, task: int, run_name: str) -> Path:
    return root / "checkpoints" / str(task) / "step3_optimized" / run_name


def resolve_step5_dir(root: Path, task: int, step3_run: str, step5_run: str) -> Path:
    return root / "checkpoints" / str(task) / "step3_optimized" / step3_run / "step5" / step5_run


def resolve_step3_log_dir(root: Path, task: int, run_name: str) -> Path:
    return root / "log" / str(task) / "step3_optimized" / run_name


def resolve_step5_log_dir(root: Path, task: int, step5_run: str) -> Path:
    return root / "log" / str(task) / "step5_optimized" / step5_run


def resolve_step4_log_dir(root: Path, task: int, run_name: str) -> Path:
    """Step 4 与 Step 3 同任务、同 step3 run 命名空间下记录日志。"""
    return root / "log" / str(task) / "step4_optimized" / run_name


def resolve_train_csv(root: Path, task: int, step3_run: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return resolve_step3_dir(root, task, step3_run) / "factuals_counterfactuals.csv"


def resolve_model_path(
    root: Path,
    task: int,
    step3_run: str | None,
    step5_run: str | None,
    explicit: str | None,
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if step3_run and step5_run:
        return resolve_step5_dir(root, task, step3_run, step5_run) / "model.pth"
    if step3_run:
        return resolve_step3_dir(root, task, step3_run) / "model.pth"
    raise ValueError("model path cannot be resolved: need --model-path or step3/step5 run ids")


def resolve_metrics_dir(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "eval_runs"
