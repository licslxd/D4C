"""产物路径：训练 CSV、模型、manifest 相关。"""
from __future__ import annotations

import os
from pathlib import Path

from d4c_core import path_layout, run_naming
from d4c_core.config_loader import ResolvedConfig
from d4c_core.paths import (
    repo_root_from_code_dir,
    resolve_iteration_root_dir,
    resolve_metrics_dir,
    resolve_step3_dir,
    resolve_step5_dir,
)


def train_csv_path(cfg: ResolvedConfig) -> Path:
    if cfg.train_csv:
        return Path(cfg.train_csv).expanduser().resolve()
    if cfg.command == "step4":
        return Path(cfg.checkpoint_dir) / "factuals_counterfactuals.csv"
    if cfg.step5_run:
        rid4 = run_naming.step4_slug_from_step5_slug(cfg.step5_run)
        return (
            path_layout.get_train_step4_run_root(cfg.repo_root, cfg.task_id, cfg.iteration_id, rid4)
            / "factuals_counterfactuals.csv"
        )
    if cfg.step4_run:
        rid4 = run_naming.parse_run_id(cfg.step4_run)
        return (
            path_layout.get_train_step4_run_root(cfg.repo_root, cfg.task_id, cfg.iteration_id, rid4)
            / "factuals_counterfactuals.csv"
        )
    raise ValueError(
        "无法解析训练 CSV：请指定 --train-csv，或 --step5-run（如 2_1_1 → train/step4/2_1/），"
        "或在 step4 命令下使用 checkpoint 目录内 CSV。"
    )


def model_path_default(cfg: ResolvedConfig) -> Path:
    return path_layout.model_file_path(Path(cfg.checkpoint_dir))


def ensure_step5_csv_symlink(cfg: ResolvedConfig) -> None:
    assert cfg.from_run is not None and cfg.step5_run is not None
    dest = Path(cfg.checkpoint_dir) / "factuals_counterfactuals.csv"
    src = train_csv_path(cfg)
    if not src.is_file():
        raise FileNotFoundError(f"缺少 Step4 产物 CSV: {src}")
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == src.resolve():
            return
        raise FileExistsError(f"已存在且非预期软链: {dest}")
    rel = os.path.relpath(src, dest.parent)
    os.symlink(rel, dest)


__all__ = [
    "train_csv_path",
    "model_path_default",
    "ensure_step5_csv_symlink",
    "repo_root_from_code_dir",
    "resolve_step3_dir",
    "resolve_step5_dir",
    "resolve_iteration_root_dir",
    "resolve_metrics_dir",
]
