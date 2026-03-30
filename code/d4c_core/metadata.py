"""运行期元信息：轻量摘要（完整主线块见 logging_meta.print_pre_run_banner）。"""
from __future__ import annotations

from d4c_core.config_loader import ResolvedConfig


def print_resolved_summary(cfg: ResolvedConfig) -> None:
    """兼容旧调用点；与 ``logging_meta.print_pre_run_banner`` 相比为精简版。"""
    print("[D4C Mainline] resolved (short):", flush=True)
    print(f"  task={cfg.task_id} preset={cfg.preset_name!r} command={cfg.command}", flush=True)
    print(f"  auxiliary={cfg.auxiliary!r} target={cfg.target!r}", flush=True)
    print(f"  checkpoint_dir={cfg.checkpoint_dir}", flush=True)
    print(f"  log_dir={cfg.log_dir}", flush=True)
    print(f"  metrics_dir={cfg.metrics_dir}", flush=True)
    if cfg.run_name:
        print(f"  run_name={cfg.run_name!r}", flush=True)
    if cfg.from_run:
        print(f"  from_run={cfg.from_run!r}", flush=True)
    if cfg.step5_run:
        print(f"  step5_run={cfg.step5_run!r}", flush=True)
    print(
        f"  batch_size={cfg.batch_size} epochs={cfg.epochs} num_proc={cfg.num_proc} "
        f"ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
        flush=True,
    )
