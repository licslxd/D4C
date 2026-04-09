"""单次运行的复现/排障清单：结构化字段 + 稳定 JSON 路径。"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from d4c_core.artifacts import train_csv_path
from d4c_core.config_loader import ResolvedConfig
from d4c_core import path_layout
from d4c_core.training_diagnostics import training_diagnostics_snapshot
from d4c_core.generation_semantics import compute_generation_semantic_family_tag

MANIFEST_SCHEMA_VERSION = "4.3"
MANIFEST_FILENAME = "manifest.json"


def _stage_label(command: str) -> str:
    return {
        "step3": "step3_domain_adversarial",
        "step4": "step4_counterfactual_eval_inference",
        "step5": "step5_main_train",
        "eval": "eval_step5_valid",
        "eval-rerank": "eval_step5_valid_rerank",
    }.get(command, command)


def _resolved_train_csv(cfg: ResolvedConfig) -> str | None:
    if cfg.command == "step4":
        if not cfg.from_run:
            return None
        return str(train_csv_path(cfg).resolve())
    if cfg.command == "step5" and cfg.from_run and cfg.step5_run:
        return str(train_csv_path(cfg).resolve())
    if cfg.command in ("eval", "eval-rerank") and cfg.step5_run:
        return str(train_csv_path(cfg).resolve())
    return None


def _resolved_model_weights(cfg: ResolvedConfig) -> str | None:
    if cfg.model_path:
        return str(Path(cfg.model_path).resolve())
    if cfg.command in ("step5", "eval", "eval-rerank"):
        ck = Path(cfg.checkpoint_dir)
        if str(getattr(cfg, "checkpoint_kind", "best_mainline") or "best_mainline") == "last":
            return str(path_layout.last_model_path(ck))
        return str(path_layout.best_mainline_model_path(ck))
    return None


def _run_lineage(cfg: ResolvedConfig) -> dict[str, Any]:
    """task/iter 与各 stage slug，供实验组脚本单点读取。"""
    out: dict[str, Any] = {
        "task_id": cfg.task_id,
        "iteration_id": cfg.iteration_id,
    }
    if cfg.run_name is not None:
        out["step3_run"] = cfg.run_name
    if cfg.from_run is not None:
        out["step3_run"] = cfg.from_run
    if cfg.step4_run:
        out["step4_run"] = cfg.step4_run
    if cfg.step5_run:
        out["step5_run"] = cfg.step5_run
    if cfg.eval_run_dir:
        er = Path(cfg.eval_run_dir)
        out["eval_run"] = er.name
        out["eval_run_dir"] = str(er.resolve())
        out["metrics_json_path"] = str((er / "metrics.json").resolve())
    if cfg.command == "eval-rerank" and cfg.eval_run_dir:
        out["rerank_run"] = Path(cfg.eval_run_dir).name
        out["rerank_run_dir"] = str(Path(cfg.eval_run_dir).resolve())
    dr = getattr(cfg, "decode_preset_id", "") or ""
    if dr:
        out["decode_preset_id"] = dr
    if cfg.command == "eval-rerank" and cfg.rerank_preset_id:
        out["rerank_preset_id"] = cfg.rerank_preset_id
    return out


def build_run_manifest(cfg: ResolvedConfig, *, cli_invocation: str | None = None) -> dict[str, Any]:
    """
    供 stdout 摘要、JSON 落盘与外部工具解析。
    字段以结构化嵌套为主（manifest_schema_version 2.0 起不再写入与嵌套重复的扁平键）。

    **Schema 4.3**：运行环境（OMP/MKL/TOKENIZERS/CUDA 等）**仅**出现在顶层 ``runtime_env``；
    ``hyperparameters`` 不含线程或 CUDA 镜像字段。
    """
    if not (cli_invocation or "").strip():
        cli_invocation = (os.environ.get("D4C_MANIFEST_CLI_INVOCATION") or "").strip() or None

    train_csv_res = _resolved_train_csv(cfg)
    model_res = _resolved_model_weights(cfg)

    _train_fp = getattr(cfg, "training_semantic_fingerprint", "") or ""
    _gen_fp = getattr(cfg, "generation_semantic_fingerprint", "") or ""
    _rd_fp = getattr(cfg, "runtime_diagnostics_fingerprint", "") or ""
    _src_json = getattr(cfg, "config_field_sources_json", "") or "{}"
    try:
        _src_obj = json.loads(_src_json) if _src_json.strip() else {}
    except json.JSONDecodeError:
        _src_obj = {}
    _cp = getattr(cfg, "consumed_presets_json", "") or "{}"
    try:
        _consumed = json.loads(_cp) if _cp.strip() else {}
    except json.JSONDecodeError:
        _consumed = {}
    _bcb = getattr(cfg, "config_before_cli_json", "") or "{}"
    try:
        _before_cli = json.loads(_bcb) if _bcb.strip() else {}
    except json.JSONDecodeError:
        _before_cli = {}
    _treq = getattr(cfg, "thread_env_requested_json", "") or "{}"
    _tee = getattr(cfg, "thread_env_effective_json", "") or "{}"
    _lreq = getattr(cfg, "launcher_env_requested_json", "") or "{}"
    _lee = getattr(cfg, "launcher_env_effective_json", "") or "{}"
    try:
        _treq_o = json.loads(_treq) if _treq.strip() else {}
    except json.JSONDecodeError:
        _treq_o = {}
    try:
        _tee_o = json.loads(_tee) if _tee.strip() else {}
    except json.JSONDecodeError:
        _tee_o = {}
    try:
        _lreq_o = json.loads(_lreq) if _lreq.strip() else {}
    except json.JSONDecodeError:
        _lreq_o = {}
    try:
        _lee_o = json.loads(_lee) if _lee.strip() else {}
    except json.JSONDecodeError:
        _lee_o = {}
    m: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(cfg.repo_root.resolve()),
        "mainline_command": cfg.command,
        "stage": _stage_label(cfg.command),
        "task_id": cfg.task_id,
        "invoked_command": getattr(cfg, "invoked_command", None) or cfg.command,
        "resolved_command_kind": getattr(cfg, "resolved_command_kind", None) or cfg.command,
        "cell_command": getattr(cfg, "cell_command", None),
        "matrix_session_id": getattr(cfg, "matrix_session_id", None),
        "matrix_cell_id": getattr(cfg, "matrix_cell_id", None),
        "training_semantic_fingerprint": _train_fp or None,
        "generation_semantic_fingerprint": _gen_fp or None,
        "runtime_diagnostics_fingerprint": _rd_fp or None,
        "config_field_sources": _src_obj,
        "consumed_presets": _consumed,
        "config_before_cli": _before_cli,
        "runtime_env": {
            "thread_env_requested": _treq_o,
            "thread_env_effective": _tee_o,
            "launcher_env_requested": _lreq_o,
            "launcher_env_effective": _lee_o,
            "note": (
                "runtime_env 为唯一运行环境记录区（OMP/MKL/TOKENIZERS/CUDA_VISIBLE_DEVICES 等）；"
                "不计入 training_semantic_fingerprint / generation_semantic_fingerprint"
            ),
        },
        "training_preset": cfg.preset_name,
        "hardware_preset": cfg.hardware_preset_id,
        "decode_preset": cfg.decode_preset_id or None,
        "eval_profile": getattr(cfg, "eval_profile_id", "") or None,
        "generation_semantic_resolved": (
            {
                "decode_preset": cfg.decode_preset_id,
                "decode_strategy": cfg.decode_strategy,
                "decode_seed": cfg.decode_seed,
                "max_explanation_length": cfg.max_explanation_length,
                "decode_max_explanation_length": cfg.max_explanation_length,
                "label_smoothing": cfg.label_smoothing,
                "repetition_penalty": cfg.repetition_penalty,
                "generate_temperature": cfg.generate_temperature,
                "generate_top_p": cfg.generate_top_p,
                "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
                "min_len": cfg.min_len,
                "decode_profile_sha1": hashlib.sha1(
                    (cfg.decode_profile_json or "").encode("utf-8")
                ).hexdigest()[:16],
                "rerank_profile_sha1": hashlib.sha1(
                    (cfg.rerank_profile_json or "").encode("utf-8")
                ).hexdigest()[:16],
                "generation_semantic_family_tag": compute_generation_semantic_family_tag(
                    {
                        "strategy": cfg.decode_strategy,
                        "temperature": cfg.generate_temperature,
                        "top_p": cfg.generate_top_p,
                        "repetition_penalty": cfg.repetition_penalty,
                        "max_explanation_length": cfg.max_explanation_length,
                        "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
                        "min_len": cfg.min_len,
                    }
                ),
            }
            if (cfg.decode_preset_id or "").strip()
            else None
        ),
        "training_label": {
            "train_label_max_length": getattr(cfg, "train_label_max_length", None),
            "train_dynamic_padding": getattr(cfg, "train_dynamic_padding", None),
            "train_padding_strategy": getattr(cfg, "train_padding_strategy", None),
            "decode_max_explanation_length": cfg.max_explanation_length,
        },
        "domain_auxiliary": cfg.auxiliary,
        "domain_target": cfg.target,
        "run_lineage": _run_lineage(cfg),
        "checkpoint_resolution": (
            {
                "default_checkpoint_kind": getattr(cfg, "checkpoint_kind", "best_mainline"),
                "best_checkpoint_path": str(
                    path_layout.best_mainline_model_path(Path(cfg.checkpoint_dir))
                ),
                "last_checkpoint_path": str(path_layout.last_model_path(Path(cfg.checkpoint_dir))),
                "checkpoint_selection_metric": "mainline_composite",
                "checkpoint_selection_decode_semantics": "mainline_greedy_alignment",
            }
            if cfg.command in ("step5", "eval", "eval-rerank")
            else None
        ),
        "paths": {
            "stage_run_dir": cfg.checkpoint_dir,
            "log_dir": cfg.log_dir,
            "iteration_root_dir": cfg.iteration_root_dir,
            "manifest_dir": cfg.manifest_dir,
            "eval_run_dir": cfg.eval_run_dir,
            **(
                {
                    "step3_checkpoint_dir": cfg.step3_checkpoint_dir,
                    "step4_run": cfg.step4_run,
                }
                if cfg.command == "step4" and cfg.step3_checkpoint_dir
                else (
                    {"step4_run": cfg.step4_run}
                    if cfg.step4_run
                    else {}
                )
            ),
        },
        "hyperparameters": {
            "learning_rate": cfg.learning_rate,
            "coef": cfg.coef,
            "adv": cfg.adv,
            "eta": cfg.eta,
            "train_global_batch_size": cfg.train_batch_size,
            "train_per_device_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "effective_global_batch_size": cfg.effective_global_batch_size,
            "epochs": cfg.epochs,
            "num_proc": cfg.num_proc,
            "ddp_world_size": cfg.ddp_world_size,
            "seed": cfg.seed,
            "label_smoothing": cfg.label_smoothing,
            "train_label_max_length": getattr(cfg, "train_label_max_length", None),
            "train_dynamic_padding": getattr(cfg, "train_dynamic_padding", None),
            "train_padding_strategy": getattr(cfg, "train_padding_strategy", None),
            "loss_weight_repeat_ul": getattr(cfg, "loss_weight_repeat_ul", None),
            "loss_weight_terminal_clean": getattr(cfg, "loss_weight_terminal_clean", None),
            "repetition_penalty": cfg.repetition_penalty,
            "generate_temperature": cfg.generate_temperature,
            "generate_top_p": cfg.generate_top_p,
            "training_preset_train_batch_size": cfg.training_preset_train_batch_size,
            **(
                {
                    "global_eval_batch_size": cfg.global_eval_batch_size,
                    "eval_per_gpu_batch_size": cfg.eval_per_gpu_batch_size,
                }
                if cfg.global_eval_batch_size is not None
                else {}
            ),
            **(
                {"full_bleu_eval_resolved": dict(cfg.full_bleu_eval_resolved)}
                if getattr(cfg, "full_bleu_eval_resolved", None)
                else {}
            ),
            "full_bleu_decode_strategy": getattr(cfg, "full_bleu_decode_strategy", "inherit"),
        },
        "step_modes": {
            "step3_mode": cfg.step3_mode,
            "step5_train_only": cfg.step5_train_only,
        },
        "training_diagnostics": training_diagnostics_snapshot(
            diagnostics_scope="parent",
            effective_training_payload_json=str(
                getattr(cfg, "effective_training_payload_json", "") or ""
            ),
        ),
        "governance_layer": {
            "purpose": "repro_orchestration_audit",
            "note": "manifest/fingerprint/matrix/analysis_pack 属工程治理层，不属于核心建模增强。",
        },
    }
    m["effective_config"] = {
        "hyperparameters": m["hyperparameters"],
        "hardware_preset": cfg.hardware_preset_id,
        "training_preset": cfg.preset_name,
        "decode_preset": cfg.decode_preset_id or None,
        "eval_profile_orchestrator": getattr(cfg, "eval_profile_id", "") or None,
        "rerank_preset": (cfg.rerank_preset_id or None) if cfg.command == "eval-rerank" else None,
        "training_semantic_fingerprint": _train_fp or None,
        "generation_semantic_fingerprint": _gen_fp or None,
    }
    if cfg.command in ("eval", "eval-rerank", "eval-matrix", "eval-rerank-matrix", "step4") and getattr(
        cfg, "eval_profile_id", ""
    ):
        _ej = getattr(cfg, "eval_profile_resolution_json", "") or "{}"
        try:
            _eor = json.loads(_ej) if _ej.strip() else {}
        except json.JSONDecodeError:
            _eor = {}
        m["eval_profile_detail"] = {
            "eval_profile": cfg.eval_profile_id,
            "resolved_hardware_preset": cfg.hardware_preset_id,
            "resolved_decode_preset": cfg.decode_preset_id or None,
            "resolved_rerank_preset": (cfg.rerank_preset_id or None)
            if cfg.command in ("eval-rerank", "eval-rerank-matrix")
            else None,
            "global_eval_batch_size": cfg.global_eval_batch_size,
            "eval_per_gpu_batch_size": cfg.eval_per_gpu_batch_size,
            "ddp_world_size": cfg.ddp_world_size,
            "orchestrator_yaml": _eor if isinstance(_eor, dict) else {},
        }
    if cli_invocation:
        m["invoked_command_line"] = cli_invocation

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

    if cfg.command == "eval-rerank":
        m["rerank"] = {
            "rerank_preset": cfg.rerank_preset_id,
            "num_return_sequences": cfg.num_return_sequences,
            "rerank_method": cfg.rerank_method,
            "rerank_top_k": cfg.rerank_top_k,
            "rerank_weight_logprob": cfg.rerank_weight_logprob,
            "rerank_weight_length": cfg.rerank_weight_length,
            "rerank_weight_repeat": cfg.rerank_weight_repeat,
            "rerank_weight_dirty": cfg.rerank_weight_dirty,
            "rerank_target_len_ratio": cfg.rerank_target_len_ratio,
            "export_examples_mode": cfg.export_examples_mode,
            "export_full_rerank_examples": cfg.export_full_rerank_examples,
            "rerank_malformed_tail_penalty": cfg.rerank_malformed_tail_penalty,
            "rerank_malformed_token_penalty": cfg.rerank_malformed_token_penalty,
        }

    return m


def manifest_json_path(cfg: ResolvedConfig) -> Path:
    """与当次 run 产物同目录的 ``manifest.json``。"""
    return Path(cfg.manifest_dir) / MANIFEST_FILENAME


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
    默认写入 JSON（与当次 run 同目录，通常在 runs/ 下）。

    显式关闭：D4C_WRITE_RUN_MANIFEST=0 / false / no / off
    """
    v = (os.environ.get("D4C_WRITE_RUN_MANIFEST") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True
