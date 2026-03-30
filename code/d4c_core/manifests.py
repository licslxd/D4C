"""单次运行的复现/排障清单：结构化字段 + 稳定 JSON 路径。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from d4c_core.config_loader import ResolvedConfig
from d4c_core import paths as d4c_paths

MANIFEST_SCHEMA_VERSION = "1.1"
MANIFEST_FILENAME = "d4c_run_manifest.json"


def _stage_label(command: str) -> str:
    return {
        "step3": "step3_domain_adversarial",
        "step4": "step4_counterfactual_generate",
        "step5": "step5_main_train",
        "eval": "eval_step5_valid",
    }.get(command, command)


def _resolved_train_csv(cfg: ResolvedConfig) -> str | None:
    if cfg.command not in ("step5", "step4") or not cfg.from_run:
        return None
    p = d4c_paths.resolve_train_csv(cfg.repo_root, cfg.task_id, cfg.from_run, cfg.train_csv)
    return str(p.resolve())


def _resolved_model_weights(cfg: ResolvedConfig) -> str | None:
    if cfg.model_path:
        return str(Path(cfg.model_path).resolve())
    if cfg.command in ("step5", "eval") and cfg.from_run and cfg.step5_run:
        return str((Path(cfg.checkpoint_dir) / "model.pth").resolve())
    return None


def build_run_manifest(cfg: ResolvedConfig, *, cli_invocation: str | None = None) -> dict[str, Any]:
    """
    供 stdout 摘要、JSON 落盘与外部工具解析。
    字段以「复现与排障可读」优先；部分键与旧版兼容（如 command / preset）。
    """
    if not (cli_invocation or "").strip():
        cli_invocation = (os.environ.get("D4C_MANIFEST_CLI_INVOCATION") or "").strip() or None

    train_csv_res = _resolved_train_csv(cfg)
    model_res = _resolved_model_weights(cfg)

    m: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(cfg.repo_root.resolve()),
        "mainline_command": cfg.command,
        "stage": _stage_label(cfg.command),
        "task_id": cfg.task_id,
        "training_preset": cfg.preset_name,
        "runtime_preset": cfg.runtime_preset_id,
        "decode_preset": cfg.decode_preset_id,
        "decode_resolved": {
            "decode_preset": cfg.decode_preset_id,
            "decode_strategy": cfg.decode_strategy,
            "decode_seed": cfg.decode_seed,
            "max_explanation_length": cfg.max_explanation_length,
            "label_smoothing": cfg.label_smoothing,
            "repetition_penalty": cfg.repetition_penalty,
            "generate_temperature": cfg.generate_temperature,
            "generate_top_p": cfg.generate_top_p,
        },
        "domain_auxiliary": cfg.auxiliary,
        "domain_target": cfg.target,
        "paths": {
            "checkpoint_dir": cfg.checkpoint_dir,
            "log_dir": cfg.log_dir,
            "metrics_dir": cfg.metrics_dir,
        },
        "hyperparameters": {
            "learning_rate": cfg.learning_rate,
            "coef": cfg.coef,
            "adv": cfg.adv,
            "eta": cfg.eta,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "num_proc": cfg.num_proc,
            "ddp_world_size": cfg.ddp_world_size,
            "seed": cfg.seed,
            "label_smoothing": cfg.label_smoothing,
            "repetition_penalty": cfg.repetition_penalty,
            "generate_temperature": cfg.generate_temperature,
            "generate_top_p": cfg.generate_top_p,
        },
        "step_modes": {
            "step3_mode": cfg.step3_mode,
            "step5_train_only": cfg.step5_train_only,
        },
    }
    if cli_invocation:
        m["cli_invocation"] = cli_invocation

    ids: dict[str, Any] = {}
    if cfg.run_name is not None:
        ids["run_name"] = cfg.run_name
    if cfg.from_run is not None:
        ids["from_run"] = cfg.from_run
    if cfg.step5_run is not None:
        ids["step5_run"] = cfg.step5_run
    if ids:
        m["run_identifiers"] = ids

    ri: dict[str, Any] = {}
    if cfg.train_csv:
        ri["train_csv_cli"] = cfg.train_csv
    if train_csv_res:
        ri["train_csv_resolved"] = train_csv_res
    if model_res:
        ri["model_weights_resolved"] = model_res
    if ri:
        m["resolved_inputs"] = ri

    # --- 扁平兼容键（便于 jq / 旧脚本）---
    m["command"] = cfg.command
    m["preset"] = cfg.preset_name
    m["task"] = cfg.task_id
    m["auxiliary"] = cfg.auxiliary
    m["target"] = cfg.target
    m["checkpoint_dir"] = cfg.checkpoint_dir
    m["log_dir"] = cfg.log_dir
    m["metrics_dir"] = cfg.metrics_dir
    m["batch_size"] = cfg.batch_size
    m["epochs"] = cfg.epochs
    m["num_proc"] = cfg.num_proc
    m["ddp_world_size"] = cfg.ddp_world_size
    m["seed"] = cfg.seed
    m["step3_mode"] = cfg.step3_mode
    m["step5_train_only"] = cfg.step5_train_only
    if cfg.run_name is not None:
        m["run_name"] = cfg.run_name
    if cfg.from_run is not None:
        m["from_run"] = cfg.from_run
    if cfg.step5_run is not None:
        m["step5_run"] = cfg.step5_run
    if cfg.train_csv:
        m["train_csv"] = cfg.train_csv
    if cfg.model_path:
        m["model_path"] = cfg.model_path
    elif model_res:
        m["model_path"] = model_res

    return m


def manifest_json_path(cfg: ResolvedConfig) -> Path:
    """稳定路径：`<log_dir>/d4c_run_manifest.json`（与 MANIFEST_FILENAME 一致）。"""
    return Path(cfg.log_dir) / MANIFEST_FILENAME


def write_run_manifest_json(cfg: ResolvedConfig, manifest: Mapping[str, Any] | None = None) -> Path:
    data = dict(manifest) if manifest is not None else build_run_manifest(cfg)
    out = manifest_json_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def should_write_manifest_json() -> bool:
    """
    默认写入 JSON（便于复现与排障；文件在 log/ 下，通常已被 .gitignore）。

    显式关闭：D4C_WRITE_RUN_MANIFEST=0 / false / no / off
    """
    v = (os.environ.get("D4C_WRITE_RUN_MANIFEST") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True
