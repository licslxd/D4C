# -*- coding: utf-8 -*-
"""
D4C 离线运行路径配置
通过环境变量 D4C_ROOT 指定项目根目录，默认为 code 的上级目录。
可选 D4C_CHECKPOINT_SUBDIR、D4C_CHECKPOINT_GROUP：仅 SUBDIR 时为 checkpoints/<task>/<subdir>/；GROUP+SUBDIR 均设时为 checkpoints/<task>/<group>/<subdir>/。
日志目录见 get_log_task_dir。**仅影响日志**且与 checkpoint 解耦时，可设 D4C_LOG_GROUP / D4C_LOG_SUBDIR（语义与 checkpoint 两变量对称）或单独设 D4C_LOG_STEP（等价于单层 log/<task>/<STEP>/）。
未使用上述 LOG_* 变量时，仍按 checkpoint 环境变量解析（与 checkpoint 不完全对称——当 GROUP+SUBDIR 均设时，日志固定在 log/<task>/<group>/），
runs/<时间戳>/train.log 等由 shell 传入 --log_file；eval 汇总由 train_logging 写入同级的 eval/ 子目录（eval_runs.*）；权重仍在 …/<group>/<subdir>/。

D4C_ROOT、D4C_CHECKPOINT_* 在运行时通过 get_d4c_root() / get_checkpoint_subdir() / get_checkpoint_group() 读取，
不在 import 时冻结，避免训练子进程或后续 export 与已 import 模块不一致。
"""
from __future__ import annotations

import os
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_d4c_root() -> str:
    """项目根目录；运行时读取 D4C_ROOT，默认 code 的上一级。"""
    env = os.environ.get("D4C_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def get_checkpoint_subdir() -> str:
    return os.environ.get("D4C_CHECKPOINT_SUBDIR", "").strip()


def get_checkpoint_group() -> str:
    return os.environ.get("D4C_CHECKPOINT_GROUP", "").strip()


def get_checkpoint_task_dir(task_idx):
    """单任务目录：checkpoints/<task>/ 或 checkpoints/<task>/<subdir>/ 或 checkpoints/<task>/<group>/<subdir>/"""
    t = str(task_idx)
    base = os.path.join(get_d4c_root(), "checkpoints", t)
    _ckpt_sub = get_checkpoint_subdir()
    _ckpt_group = get_checkpoint_group()
    if _ckpt_sub:
        if _ckpt_group:
            return os.path.join(base, _ckpt_group, _ckpt_sub)
        return os.path.join(base, _ckpt_sub)
    if _ckpt_group:
        return os.path.join(base, _ckpt_group)
    return base


def get_log_task_dir(task_idx):
    """单任务日志根目录：其下可有 runs/<时间戳>/train.log（由 run_step*_optimized.sh 约定）；eval 汇总在子目录 eval/。

    优先级（从高到低）：
    1. **仅日志** — 若 ``D4C_LOG_SUBDIR`` 或 ``D4C_LOG_GROUP`` 任一非空：布局与 checkpoint 两变量对称
       （二者均非空时目录为 ``log/<task>/<D4C_LOG_GROUP>/``，不按 LOG_SUBDIR 再分层）。
    2. **仅日志** — 否则若 ``D4C_LOG_STEP`` 非空：``log/<task>/<D4C_LOG_STEP>/``（与 checkpoint 无关）。
    3. **沿用 checkpoint 环境变量**（与 get_checkpoint_task_dir 使用同一组 D4C_CHECKPOINT_*）：
       - 无 SUBDIR：log/<task>/ 或（仅有 GROUP 时）log/<task>/<group>/
       - 仅有 SUBDIR：log/<task>/<subdir>/
       - GROUP 与 SUBDIR 均设：checkpoint 为 …/<group>/<subdir>/，日志统一在 log/<task>/<group>/（不按 SUBDIR 再分子目录）
    """
    t = str(task_idx)
    base = os.path.join(get_d4c_root(), "log", t)
    log_sub = os.environ.get("D4C_LOG_SUBDIR", "").strip()
    log_group = os.environ.get("D4C_LOG_GROUP", "").strip()
    log_step = os.environ.get("D4C_LOG_STEP", "").strip()
    _ckpt_sub = get_checkpoint_subdir()
    _ckpt_group = get_checkpoint_group()
    if log_sub or log_group:
        if log_sub:
            if log_group:
                return os.path.join(base, log_group)
            return os.path.join(base, log_sub)
        return os.path.join(base, log_group)
    if log_step:
        return os.path.join(base, log_step)
    if _ckpt_sub:
        if _ckpt_group:
            return os.path.join(base, _ckpt_group)
        return os.path.join(base, _ckpt_sub)
    if _ckpt_group:
        return os.path.join(base, _ckpt_group)
    return base


CODE_DIR = _SCRIPT_DIR
DEFAULT_MIRROR_LOG = os.path.join(CODE_DIR, "log.out")
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"  # 需下载到 HF_HOME


def __getattr__(name: str) -> Any:
    """惰性解析依赖 D4C_ROOT 的路径，兼容 ``from paths_config import DATA_DIR`` 等旧写法。"""
    root = get_d4c_root()
    if name == "D4C_ROOT":
        return root
    if name == "MODELS_DIR":
        return os.path.join(root, "pretrained_models")
    if name == "T5_SMALL_DIR":
        return os.path.join(root, "pretrained_models", "t5-small")
    if name == "MPNET_DIR":
        return os.path.join(root, "pretrained_models", "sentence-transformers_all-mpnet-base-v2")
    if name == "MPNET_PATH":
        return os.path.join(root, "pretrained_models", "sentence-transformers_all-mpnet-base-v2")
    if name == "METEOR_CACHE":
        return os.path.join(root, "pretrained_models", "evaluate_meteor")
    if name == "DATA_DIR":
        return os.path.join(root, "data")
    if name == "MERGED_DATA_DIR":
        return os.path.join(root, "Merged_data")
    if name == "CHECKPOINT_DIR":
        return os.path.join(root, "checkpoints")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    return os.path.join(get_d4c_root(), "data", dataset)


def get_merged_path(task_idx):
    """获取 Merged_data 中任务目录"""
    return os.path.join(get_d4c_root(), "Merged_data", str(task_idx))


def get_t5_tokenizer_path():
    """T5 tokenizer 本地路径"""
    return os.path.join(get_d4c_root(), "pretrained_models", "t5-small")


def get_mpnet_path():
    """MPNet 模型本地路径"""
    return os.path.join(get_d4c_root(), "pretrained_models", "sentence-transformers_all-mpnet-base-v2")
