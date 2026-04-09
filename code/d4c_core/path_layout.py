"""新版唯一正式产物布局：runs/task{T}/vN/... 与 runs/global/vN/...（禁止默认写入 checkpoints/、log/）。

目录边界（须与文档、Shell 注释一致；无旧路径 fallback）：

- ``runs/global/vN/meta/`` **仅**放跨任务、无法归属单一 task 的产物，例如：
  ``eval_registry_all.*``、多任务编排器的 shell 汇总日志、跨任务批处理摘要、其它 truly global 元数据。

- 凡可明确归属 ``task{T}`` 的产物 **禁止** 写入 global，必须落在
  ``runs/task{T}/vN/meta/``，例如：单任务 ``eval_registry.*``、单任务 eval/rerank 侧写注册、
  analysis pack 引用源、bad cases、**multi_seed**（``meta/multi_seed/<run>/``）、单任务 shell_logs。
  另：``runs/task{T}/vN/baselines/`` 存放正式基线登记与 metrics 快照（见 ``d4c_core.baseline_registry``）。

详见 README 与 ``docs/D4C_Scripts_and_Runtime_Guide.md`` §2。
"""
from __future__ import annotations

from pathlib import Path

from d4c_core import run_naming


def runs_root(repo_root: Path) -> Path:
    return (repo_root / "runs").resolve()


def get_global_iteration_root(repo_root: Path, iteration_id: str) -> Path:
    """``runs/global/vN/``（仅迭代段；子树内只应含跨任务元数据）。"""
    it = run_naming.normalize_iteration_id(iteration_id)
    return runs_root(repo_root) / "global" / it


def get_global_meta_dir(repo_root: Path, iteration_id: str) -> Path:
    """跨任务全局 meta：如 ``eval_registry_all.*``、多任务 ``shell_logs``。"""
    return get_global_iteration_root(repo_root, iteration_id) / "meta"


def get_task_root(repo_root: Path, task_id: int) -> Path:
    return runs_root(repo_root) / f"task{int(task_id)}"


def get_iteration_root(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    it = run_naming.normalize_iteration_id(iteration_id)
    return get_task_root(repo_root, task_id) / it


def get_task_meta_dir(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    """单任务迭代 meta：``eval_registry.*``、``shell_logs``、``multi_seed/`` 等。"""
    return get_iteration_root(repo_root, task_id, iteration_id) / "meta"


def get_multiseed_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    """``runs/task{T}/vN/meta/multi_seed/<run>/``（与 ``run_naming`` 分配的目录名一致，含 shell tee 日志）。"""
    rid = run_naming.parse_run_id(run_id)
    return get_task_meta_dir(repo_root, task_id, iteration_id) / "multi_seed" / rid


def get_train_step3_run_root(
    repo_root: Path, task_id: int, iteration_id: str, run_id: str
) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "train" / "step3" / run_id


def get_train_step5_run_root(
    repo_root: Path, task_id: int, iteration_id: str, run_id: str
) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "train" / "step5" / run_id


def get_train_step4_run_root(
    repo_root: Path, task_id: int, iteration_id: str, run_id: str
) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "train" / "step4" / run_id


def get_eval_run_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "eval" / run_id


def get_rerank_run_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "rerank" / run_id


def get_matrix_run_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "matrix" / run_id


def get_baselines_root(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    """``runs/task{T}/vN/baselines/``：注册基线、默认基线索引与 metrics 快照（不修改原 eval 目录）。"""
    return get_iteration_root(repo_root, task_id, iteration_id) / "baselines"


def get_analysis_root(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "analysis"


def get_analysis_pack_root(
    repo_root: Path, task_id: int, iteration_id: str, pack_id: str
) -> Path:
    return get_analysis_root(repo_root, task_id, iteration_id) / pack_id


def get_stage_run_root(
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    stage_name: str,
    run_id: str,
) -> Path:
    """
    stage_name: train_step3 | train_step5 | eval | rerank | matrix
    """
    if stage_name == "train_step3":
        return get_train_step3_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "train_step4":
        return get_train_step4_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "train_step5":
        return get_train_step5_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "eval":
        return get_eval_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "rerank":
        return get_rerank_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "matrix":
        return get_matrix_run_root(repo_root, task_id, iteration_id, run_id)
    raise ValueError(f"未知 stage_name: {stage_name!r}")


def best_mainline_model_path(stage_run_root: Path) -> Path:
    """Step5 默认评测与 post_train_eval 加载的 best 权重。"""
    return (stage_run_root / "model" / "best_mainline.pth").resolve()


def last_model_path(stage_run_root: Path) -> Path:
    """训练收尾 epoch 权重（不覆盖 best_mainline）。"""
    return (stage_run_root / "model" / "last.pth").resolve()


def model_file_path(stage_run_root: Path) -> Path:
    """与 best_mainline 对齐（Step5 单一默认真相）。"""
    return best_mainline_model_path(stage_run_root)


def logs_dir(stage_run_root: Path) -> Path:
    return (stage_run_root / "logs").resolve()


def hf_cache_root(repo_root: Path, task_id: int) -> Path:
    return (repo_root / "cache" / f"task{int(task_id)}" / "hf").resolve()
