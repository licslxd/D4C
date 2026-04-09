# -*- coding: utf-8 -*-
"""
D4C 离线运行路径配置（新版）

- 项目根：D4C_ROOT（默认 code 上级）。
- 当前阶段产物根目录：须设置 **D4C_STAGE_RUN_DIR**（由 ``python code/d4c.py`` 在 torchrun 前注入），
  对应 ``runs/task{T}/vN/train/step3|step4|step5/<run>/`` 等。
- Step4 另可选 **D4C_STEP3_RUN_DIR**：仅 step4 runner 加载 Step3 权重时指向 ``train/step3/<from-run>/``（CSV 与 partial 仍写入 ``D4C_STAGE_RUN_DIR``）。
- HF datasets 缓存根：须设置 **D4C_HF_CACHE_ROOT**（通常为 ``<repo>/cache/task{T}/hf``）。

路径常量请使用本模块显式 ``get_*()``；不再提供模块级惰性属性或 ``__getattr__`` 动态导出。
"""
from __future__ import annotations

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_d4c_root() -> str:
    """项目根目录；运行时读取 D4C_ROOT，默认 code 的上一级。"""
    env = os.environ.get("D4C_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def get_models_dir() -> str:
    return os.path.join(get_d4c_root(), "pretrained_models")


def get_t5_small_dir() -> str:
    return os.path.join(get_models_dir(), "t5-small")


def get_mpnet_dir() -> str:
    return os.path.join(get_models_dir(), "sentence-transformers_all-mpnet-base-v2")


def get_meteor_cache_dir() -> str:
    return os.path.join(get_models_dir(), "evaluate_meteor")


def get_data_dir() -> str:
    return os.path.join(get_d4c_root(), "data")


def get_merged_data_dir() -> str:
    return os.path.join(get_d4c_root(), "Merged_data")


def get_stage_run_dir(_task_idx: int | None = None) -> str:
    """当前 **stage run 根目录**（环境变量 ``D4C_STAGE_RUN_DIR``）。

    权重与当次 CSV 等产物与此目录对齐。``_task_idx`` 仅为与旧调用形态对齐，不参与解析。
    """
    _ = _task_idx
    stage = os.environ.get("D4C_STAGE_RUN_DIR", "").strip()
    if not stage:
        raise RuntimeError(
            "D4C_STAGE_RUN_DIR 未设置。请使用仓库根目录的: python code/d4c.py <子命令> … 启动训练/评测。"
        )
    return os.path.abspath(stage)


def get_hf_cache_root(task_idx: int) -> str:
    """HF datasets 缓存根目录；须设置 D4C_HF_CACHE_ROOT（通常为 runs 外的 cache/taskN/hf）。"""
    v = os.environ.get("D4C_HF_CACHE_ROOT", "").strip()
    if not v:
        raise RuntimeError("D4C_HF_CACHE_ROOT 未设置。请通过 d4c 主入口启动。")
    return os.path.abspath(v)


CODE_DIR = _SCRIPT_DIR
DEFAULT_MIRROR_LOG = os.path.join(CODE_DIR, "log.out")
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"  # 需下载到 HF_HOME


def _mirror_enabled():
    v = os.environ.get("D4C_MIRROR_LOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def append_log_dual(primary_log_file, text, mirror=None):
    """写入 primary_log_file（若给定）。mirror=True 或环境变量 D4C_MIRROR_LOG=1 时再写入 code/log.out。"""
    if mirror is None:
        mirror = _mirror_enabled()
    paths = []
    if primary_log_file:
        paths.append(os.path.abspath(os.path.expanduser(primary_log_file)))
    if mirror:
        paths.append(os.path.abspath(DEFAULT_MIRROR_LOG))
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


def get_data_path(dataset):
    """获取数据集路径"""
    return os.path.join(get_data_dir(), dataset)


def get_merged_path(task_idx):
    """获取 Merged_data 中任务目录"""
    return os.path.join(get_merged_data_dir(), str(task_idx))


def get_t5_tokenizer_path():
    """T5 tokenizer 本地路径"""
    return get_t5_small_dir()


def get_mpnet_path():
    """MPNet 模型本地路径"""
    return get_mpnet_dir()
