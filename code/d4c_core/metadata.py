"""运行期元信息：轻量摘要（完整主线块见 logging_meta.print_pre_run_banner）。

路径归属以 path_layout 为准：任务级 meta 为 runs/task{T}/vN/meta/，跨任务全局为 runs/global/vN/meta/。
"""
from __future__ import annotations

from d4c_core.config_loader import ResolvedConfig


def print_resolved_summary(cfg: ResolvedConfig) -> None:
    """精简版解析摘要；完整块见 ``logging_meta.print_pre_run_banner``。"""
    print("[D4C Mainline] resolved (short):", flush=True)
    print(f"  task={cfg.task_id} iter={cfg.iteration_id} preset={cfg.preset_name!r} command={cfg.command}", flush=True)
    print(f"  auxiliary={cfg.auxiliary!r} target={cfg.target!r}", flush=True)
    print(f"  checkpoint_dir={cfg.checkpoint_dir}", flush=True)
    print(f"  manifest_dir={cfg.manifest_dir}", flush=True)
    _tfp = getattr(cfg, "training_semantic_fingerprint", "") or ""
    _gfp = getattr(cfg, "generation_semantic_fingerprint", "") or ""
    if _tfp.strip():
        print(f"  training_semantic_fingerprint={_tfp}", flush=True)
    if _gfp.strip():
        print(f"  generation_semantic_fingerprint={_gfp}", flush=True)
    _rd = getattr(cfg, "runtime_diagnostics_fingerprint", "") or ""
    if _rd.strip():
        print(f"  runtime_diagnostics_fingerprint={_rd}", flush=True)
    if cfg.eval_run_dir:
        print(f"  eval_run_dir={cfg.eval_run_dir}", flush=True)
    print(f"  log_dir={cfg.log_dir}", flush=True)
    print(f"  iteration_root_dir={cfg.iteration_root_dir}", flush=True)
    if cfg.run_name:
        print(f"  run_name={cfg.run_name!r}", flush=True)
    if cfg.from_run:
        print(f"  from_run={cfg.from_run!r}", flush=True)
    if cfg.step5_run:
        print(f"  step5_run={cfg.step5_run!r}", flush=True)
    if cfg.step4_run:
        print(f"  step4_run={cfg.step4_run!r}", flush=True)
    if cfg.command == "step4" and (getattr(cfg, "eval_profile_id", "") or "").strip():
        print(f"  eval_profile_name={cfg.eval_profile_id!r}  # step4 推理 batch 仅来自该 profile", flush=True)
    eval_bs = cfg.global_eval_batch_size if cfg.global_eval_batch_size is not None else "-"
    eval_pg = (
        cfg.eval_per_gpu_batch_size
        if cfg.eval_per_gpu_batch_size is not None
        else "-"
    )
    print(
        f"  train_global_batch_size={cfg.train_batch_size} eval_global_batch_size={eval_bs} "
        f"eval_per_gpu_batch_size={eval_pg} "
        f"epochs={cfg.epochs} num_proc={cfg.num_proc} ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
        flush=True,
    )
