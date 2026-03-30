"""MAINLINE 统一日志块：阶段、预设、解析摘要、分发说明（不重复散落在 executors）。"""
from __future__ import annotations

from d4c_core.config_loader import ResolvedConfig
from d4c_core.dispatch import print_dispatch_routing, print_dispatch_script_detail
from d4c_core.manifests import MANIFEST_FILENAME, build_run_manifest, manifest_json_path, should_write_manifest_json


def _stage_label(command: str) -> str:
    return {
        "step3": "step3（域对抗）",
        "step4": "step4（反事实生成）",
        "step5": "step5（主模型训练）",
        "eval": "eval（Step5 评测）",
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
    print(f"[Stage] {_stage_label(command)}", flush=True)
    print(
        f"[Preset] training={cfg.preset_name!r} runtime={cfg.runtime_preset_id!r} "
        f"decode_preset={cfg.decode_preset_id!r}",
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
    if cfg.model_path:
        print(f"  model_path={cfg.model_path}", flush=True)
    elif ri.get("model_weights_resolved"):
        print(f"  model_weights (resolved)={ri['model_weights_resolved']}", flush=True)
    print("[Resolved Outputs]", flush=True)
    print(f"  checkpoint_dir={cfg.checkpoint_dir}", flush=True)
    print(f"  log_dir={cfg.log_dir}", flush=True)
    print(f"  metrics_dir={cfg.metrics_dir}", flush=True)
    if command == "step3":
        print(f"  step3_mode={cfg.step3_mode}", flush=True)
    if command == "step5":
        print(f"  step5_train_only={cfg.step5_train_only}", flush=True)
    hp = man.get("hyperparameters") or {}
    print(
        f"  train_runtime: batch_size={cfg.batch_size} epochs={cfg.epochs} "
        f"num_proc={cfg.num_proc} ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
        flush=True,
    )
    print(
        f"  train_objective: lr={hp.get('learning_rate')} coef={hp.get('coef')} "
        f"adv={hp.get('adv')} eta={hp.get('eta')}",
        flush=True,
    )
    dr = man.get("decode_resolved") or {}
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
        f"[Preset] Step3/Step4 使用 CLI --preset={step3_preset!r}；Step5 将强制 preset='step5'。",
        flush=True,
    )
    print_dispatch_routing("pipeline")
    print_dispatch_script_detail("pipeline")
    print(
        f"[Manifest] 各段 torchrun 前在各自 log_dir 写入 {MANIFEST_FILENAME} "
        "（默认开启；D4C_WRITE_RUN_MANIFEST=0 可关闭整段）。",
        flush=True,
    )


def print_smoke_ddp_preamble() -> None:
    print("[D4C Mainline] command=smoke-ddp", flush=True)
    print("[Stage] smoke-ddp（DDP 冒烟）", flush=True)
    print_dispatch_routing("smoke-ddp")
    print_dispatch_script_detail("smoke-ddp")
    print(
        "[Manifest] smoke-ddp 不使用标准 manifest；产物见 checkpoints/1/smoke_ddp/ 与 log/1/smoke_ddp/。",
        flush=True,
    )
