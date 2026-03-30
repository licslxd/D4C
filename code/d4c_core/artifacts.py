"""产物路径解析：checkpoint / log / CSV / model / metrics 归口（实现委托 paths）。"""
from __future__ import annotations

import os
from pathlib import Path

from d4c_core import paths as d4c_paths
from d4c_core.config_loader import ResolvedConfig


def train_csv_path(cfg: ResolvedConfig) -> Path:
    assert cfg.from_run is not None
    return d4c_paths.resolve_train_csv(cfg.repo_root, cfg.task_id, cfg.from_run, cfg.train_csv)


def model_path_default(cfg: ResolvedConfig) -> Path:
    return d4c_paths.resolve_model_path(
        cfg.repo_root,
        cfg.task_id,
        cfg.from_run,
        cfg.step5_run,
        cfg.model_path,
    )


def ensure_step5_csv_symlink(cfg: ResolvedConfig) -> None:
    """Step5 训练前：在 checkpoint 目录下挂接 Step4 CSV（与 runners 历史行为一致）。"""
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


# 显式 re-export 常用 paths API，供文档与调用方单一入口
repo_root_from_code_dir = d4c_paths.repo_root_from_code_dir
resolve_step3_dir = d4c_paths.resolve_step3_dir
resolve_step5_dir = d4c_paths.resolve_step5_dir
resolve_step3_log_dir = d4c_paths.resolve_step3_log_dir
resolve_step4_log_dir = d4c_paths.resolve_step4_log_dir
resolve_step5_log_dir = d4c_paths.resolve_step5_log_dir
resolve_metrics_dir = d4c_paths.resolve_metrics_dir
