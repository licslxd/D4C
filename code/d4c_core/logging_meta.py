"""MAINLINE 统一日志块：阶段、预设、解析摘要、分发说明（不重复散落在 executors）。"""
from __future__ import annotations

import json
from pathlib import Path

from d4c_core.config_loader import ResolvedConfig
from d4c_core.dispatch import print_dispatch_routing, print_dispatch_script_detail
from d4c_core.manifests import MANIFEST_FILENAME, build_run_manifest, manifest_json_path, should_write_manifest_json


def _stage_label(command: str) -> str:
    return {
        "step3": "step3（域对抗）",
        "step4": "step4（反事实推理，eval 语义 / eval_profile）",
        "step5": "step5（主模型训练）",
        "eval": "eval（Step5 评测）",
        "eval-rerank": "eval-rerank（Step5 多候选 rerank 评测）",
        "eval-rerank-matrix": "eval-rerank-matrix（多 decode preset × rerank）",
        "rerank-summary": "rerank-summary（Phase2 汇总表）",
        "pipeline": "pipeline（step3→step4→step5）",
        "smoke-ddp": "smoke-ddp（DDP 冒烟）",
    }.get(command, command)


def print_pre_run_banner(
    command: str,
    cfg: ResolvedConfig,
    *,
    cli_invocation: str | None = None,
) -> None:
    """运行子进程前的主线摘要（JSON 在 ``runners`` torchrun 前落盘，见 ``[Manifest] wrote``）。"""
    man = build_run_manifest(cfg, cli_invocation=cli_invocation)
    print(f"[D4C Mainline] command={command}", flush=True)
    _tfp = getattr(cfg, "training_semantic_fingerprint", "") or ""
    _gfp = getattr(cfg, "generation_semantic_fingerprint", "") or ""
    _rd = getattr(cfg, "runtime_diagnostics_fingerprint", "") or ""
    if _tfp.strip():
        print(f"[Semantic] training_semantic_fingerprint={_tfp}", flush=True)
    if _gfp.strip():
        print(f"[Semantic] generation_semantic_fingerprint={_gfp}", flush=True)
    if _rd.strip():
        print(f"[Diagnostics] runtime_diagnostics_fingerprint={_rd}", flush=True)
    print(f"[Stage] {_stage_label(command)}", flush=True)
    print(
        f"[Preset] training={cfg.preset_name!r} hardware={cfg.hardware_preset_id!r} "
        f"decode_preset={cfg.decode_preset_id!r}",
        flush=True,
    )
    if getattr(cfg, "eval_profile_id", "") and cfg.command in ("eval", "eval-rerank", "step4"):
        _rp = cfg.rerank_preset_id if cfg.command == "eval-rerank" else ""
        print(
            f"[Eval profile orchestrator] name={cfg.eval_profile_id!r} hardware={cfg.hardware_preset_id!r} "
            f"decode_preset={cfg.decode_preset_id!r} rerank_preset={_rp!r} "
            f"global_eval_batch_size={cfg.global_eval_batch_size} eval_per_gpu_batch_size={cfg.eval_per_gpu_batch_size} "
            f"ddp_world_size={cfg.ddp_world_size}",
            flush=True,
        )
    print("[Resolved Inputs]", flush=True)
    print(f"  task={cfg.task_id} auxiliary={cfg.auxiliary!r} target={cfg.target!r}", flush=True)
    if cfg.train_csv:
        print(f"  train_csv (CLI)={cfg.train_csv}", flush=True)
    ri = man.get("resolved_inputs") or {}
    if ri.get("train_csv_resolved"):
        print(f"  train_csv (resolved)={ri['train_csv_resolved']}", flush=True)
    if cfg.run_name:
        print(f"  run_name={cfg.run_name!r}", flush=True)
    if cfg.from_run:
        print(f"  from_run={cfg.from_run!r}", flush=True)
    if cfg.step5_run:
        print(f"  step5_run={cfg.step5_run!r}", flush=True)
    if cfg.step4_run:
        print(f"  step4_run={cfg.step4_run!r}", flush=True)
    if cfg.model_path:
        print(f"  model_path={cfg.model_path}", flush=True)
    elif ri.get("model_weights_resolved"):
        print(f"  model_weights (resolved)={ri['model_weights_resolved']}", flush=True)
    print("[Resolved Outputs]", flush=True)
    print(f"  stage_run_dir={cfg.checkpoint_dir}", flush=True)
    print(f"  log_dir={cfg.log_dir}", flush=True)
    print(
        f"  iteration_root_dir={cfg.iteration_root_dir}  # vN 根，非 metrics.json 目录",
        flush=True,
    )
    print(f"  iteration_id={cfg.iteration_id}", flush=True)
    print(f"  manifest_dir={cfg.manifest_dir}", flush=True)
    if cfg.eval_run_dir:
        _er = Path(cfg.eval_run_dir)
        print(f"  eval_run_dir={cfg.eval_run_dir}", flush=True)
        if cfg.command == "eval-rerank":
            print(f"  rerank_run_dir={cfg.eval_run_dir}", flush=True)
        print(f"  metrics_json_path={_er / 'metrics.json'}", flush=True)
    if command == "step3":
        print(f"  step3_mode={cfg.step3_mode}", flush=True)
    if command == "step5":
        print(f"  step5_train_only={cfg.step5_train_only}", flush=True)
    hp = man.get("hyperparameters") or {}
    _re = man.get("runtime_env") or {}
    _te = _re.get("thread_env_effective") if isinstance(_re, dict) else {}
    _le = _re.get("launcher_env_effective") if isinstance(_re, dict) else {}
    if not isinstance(_te, dict):
        _te = {}
    if not isinstance(_le, dict):
        _le = {}
    print(
        "  runtime_env: "
        f"thread_env_effective={json.dumps(_te, ensure_ascii=False)} "
        f"launcher_env_effective={json.dumps(_le, ensure_ascii=False)}",
        flush=True,
    )
    if cfg.command == "step4" and cfg.global_eval_batch_size is not None:
        _epg = cfg.eval_per_gpu_batch_size
        _epid = (getattr(cfg, "eval_profile_id", "") or "").strip()
        print(
            f"  step4_eval_inference: eval_profile_name={_epid!r} "
            f"global_eval_batch_size={cfg.global_eval_batch_size} "
            f"eval_per_gpu_batch_size={_epg} num_proc={cfg.num_proc} "
            f"ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
            flush=True,
        )
    elif cfg.global_eval_batch_size is not None and cfg.command in ("eval", "eval-rerank", "step5"):
        _epg = cfg.eval_per_gpu_batch_size
        print(
            f"  eval_parallelism: global_eval_batch_size={cfg.global_eval_batch_size} "
            f"eval_per_gpu_batch_size={_epg} train_global_batch_size={cfg.train_batch_size} "
            f"train_per_device_batch_size={cfg.per_device_train_batch_size} gradient_accumulation_steps={cfg.gradient_accumulation_steps} "
            f"effective_global_batch_size={cfg.effective_global_batch_size} num_proc={cfg.num_proc} "
            f"ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
            flush=True,
        )
    else:
        print(
            f"  train_parallelism: train_global_batch_size={cfg.train_batch_size} "
            f"train_per_device_batch_size={cfg.per_device_train_batch_size} gradient_accumulation_steps={cfg.gradient_accumulation_steps} "
            f"effective_global_batch_size={cfg.effective_global_batch_size} epochs={cfg.epochs} "
            f"num_proc={cfg.num_proc} ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
            flush=True,
        )
    print(
        f"  train_objective: lr={hp.get('learning_rate')} coef={hp.get('coef')} "
        f"adv={hp.get('adv')} eta={hp.get('eta')}",
        flush=True,
    )
    dr = man.get("generation_semantic_resolved") or {}
    print(
        f"  decode_preset={dr.get('decode_preset')!r} decode_strategy={dr.get('decode_strategy')!r} "
        f"decode_seed={dr.get('decode_seed')!r} max_explanation_length={dr.get('max_explanation_length')}",
        flush=True,
    )
    print(
        f"  decode (generation): label_smoothing={dr.get('label_smoothing')} "
        f"repetition_penalty={dr.get('repetition_penalty')} "
        f"temperature={dr.get('generate_temperature')} top_p={dr.get('generate_top_p')}",
        flush=True,
    )
    print("[Dispatch Summary]", flush=True)
    print_dispatch_routing(command)
    print_dispatch_script_detail(command)
    mp = manifest_json_path(cfg)
    if should_write_manifest_json():
        print(
            f"[Manifest] torchrun 前将写入 {mp}（文件名固定为 {MANIFEST_FILENAME}；"
            f"关闭: export D4C_WRITE_RUN_MANIFEST=0）。复现字段说明: docs/PRESETS.md、README。",
            flush=True,
        )
    else:
        print(
            "[Manifest] 已关闭 JSON 落盘（D4C_WRITE_RUN_MANIFEST=0）。"
            "排障仍可看本页 stdout 与 train.log；完整字段说明见 README。",
            flush=True,
        )


def print_pipeline_opening(*, step3_preset: str) -> None:
    print("[D4C Mainline] command=pipeline", flush=True)
    print("[Stage] pipeline（step3→step4→step5）", flush=True)
    print(
        f"[Preset] Step3/Step4 使用 CLI --preset={step3_preset!r}；Step5 将强制 preset='step5'；"
        "Step4 须本命令的 --eval-profile（推理 batch 仅来自该 profile 的 eval_batch_size）。",
        flush=True,
    )
    print_dispatch_routing("pipeline")
    print_dispatch_script_detail("pipeline")
    print(
        f"[Manifest] 各段 torchrun 前在各自 manifest_dir 写入 {MANIFEST_FILENAME} "
        "（默认开启；D4C_WRITE_RUN_MANIFEST=0 可关闭整段）。",
        flush=True,
    )


def print_smoke_ddp_preamble() -> None:
    print("[D4C Mainline] command=smoke-ddp", flush=True)
    print("[Stage] smoke-ddp（DDP 冒烟）", flush=True)
    print_dispatch_routing("smoke-ddp")
    print_dispatch_script_detail("smoke-ddp")
    print(
        "[Manifest] smoke-ddp 使用与主线相同 manifest.json；产物见 runs/task1/v0/train/…。",
        flush=True,
    )
