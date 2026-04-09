"""torchrun 子进程编排：供 ``d4c.py`` 调用；INTERNAL EXECUTOR 仅出现在本子模块。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from d4c_core.artifacts import ensure_step5_csv_symlink
from d4c_core.config_loader import ResolvedConfig, load_resolved_config
from d4c_core.dispatch import TORCHRUN_STEP3_SCRIPT, TORCHRUN_STEP4_SCRIPT, TORCHRUN_STEP5_SCRIPT
from d4c_core.logging_meta import print_pre_run_banner
from d4c_core.manifests import build_run_manifest, should_write_manifest_json, write_run_manifest_json
from d4c_core import path_layout
from d4c_core.validation import validate_resolved_config


def _torchrun_cmd() -> list[str]:
    if shutil.which("torchrun"):
        return ["torchrun"]
    return [sys.executable, "-m", "torch.distributed.run"]


def _scrub_d4c_env(env: dict[str, str]) -> None:
    for k in list(env.keys()):
        if k.startswith("D4C_"):
            del env[k]


def _scrub_training_side_env(env: dict[str, str]) -> None:
    """torchrun 子进程启动前清洗环境。

    移除父 shell 中的 ``TRAIN_*``、``EVAL_BATCH_SIZE``、``MAX_PARALLEL_CPU`` 以及**全部** ``D4C_*``
    （随后由 :func:`_d4c_layout_env` 再注入白名单变量）。因此 **export TRAIN_* / D4C_QUICK_EVAL_***
    **不会**稳定影响子进程；训练语义请以 **CLI + presets** 为准。见 README 与主指南 §3。
    """
    _scrub_d4c_env(env)
    for k in list(env.keys()):
        if k.startswith("TRAIN_"):
            del env[k]
    for k in ("EVAL_BATCH_SIZE", "MAX_PARALLEL_CPU"):
        env.pop(k, None)


def _d4c_layout_env(cfg: ResolvedConfig) -> dict[str, str]:
    hf = str(path_layout.hf_cache_root(cfg.repo_root, cfg.task_id))
    meta = str(path_layout.get_task_meta_dir(cfg.repo_root, cfg.task_id, cfg.iteration_id))
    out: dict[str, str] = {
        "D4C_ROOT": str(cfg.repo_root),
        "D4C_STAGE_RUN_DIR": str(Path(cfg.checkpoint_dir).resolve()),
        "D4C_HF_CACHE_ROOT": hf,
        "D4C_ITERATION_META_DIR": meta,
        "D4C_MANIFEST_DIR": str(Path(cfg.manifest_dir).resolve()),
    }
    _tp = getattr(cfg, "effective_training_payload_json", "") or ""
    if _tp.strip():
        out["D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON"] = _tp
    if cfg.command == "step5" and not cfg.step5_train_only:
        out["D4C_STEP5_EMBEDDED_EVAL_LOG"] = str((Path(cfg.log_dir) / "eval.log").resolve())
    if cfg.eval_run_dir:
        out["D4C_EVAL_RUN_DIR"] = str(Path(cfg.eval_run_dir).resolve())
    if cfg.command == "step4" and cfg.step3_checkpoint_dir:
        out["D4C_STEP3_RUN_DIR"] = str(Path(cfg.step3_checkpoint_dir).resolve())
    _tfp = (getattr(cfg, "training_semantic_fingerprint", "") or "").strip()
    if _tfp:
        out["D4C_TRAINING_SEMANTIC_FINGERPRINT"] = _tfp
    _gfp = (getattr(cfg, "generation_semantic_fingerprint", "") or "").strip()
    if _gfp:
        out["D4C_GENERATION_SEMANTIC_FINGERPRINT"] = _gfp
    _rd = (getattr(cfg, "runtime_diagnostics_fingerprint", "") or "").strip()
    if _rd:
        out["D4C_RUNTIME_DIAGNOSTICS_FINGERPRINT"] = _rd
    _tr = (getattr(cfg, "thread_env_requested_json", "") or "").strip()
    if _tr:
        out["D4C_THREAD_ENV_REQUESTED_JSON"] = _tr
    _te = (getattr(cfg, "thread_env_effective_json", "") or "").strip()
    if _te:
        out["D4C_THREAD_ENV_EFFECTIVE_JSON"] = _te
    _lr = (getattr(cfg, "launcher_env_requested_json", "") or "").strip()
    if _lr:
        out["D4C_LAUNCHER_ENV_REQUESTED_JSON"] = _lr
    _le = (getattr(cfg, "launcher_env_effective_json", "") or "").strip()
    if _le:
        out["D4C_LAUNCHER_ENV_EFFECTIVE_JSON"] = _le
    return out


def _run_torchrun_explicit(
    *,
    code_dir: Path,
    repo_root: Path,
    ddp_world_size: int,
    env_extra: dict[str, str],
    script: str,
    py_args: list[str],
) -> None:
    cmd = [
        *_torchrun_cmd(),
        "--standalone",
        f"--nproc_per_node={ddp_world_size}",
        script,
        *py_args,
    ]
    env = _base_env_raw(repo_root)
    env.update(env_extra)
    _ensure_code_dir_on_pythonpath(env, code_dir)
    print("[d4c] cwd:", code_dir, flush=True)
    print("[d4c] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(code_dir), env=env, check=True)


def _base_env_raw(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    _scrub_training_side_env(env)
    env["D4C_ROOT"] = str(repo_root)
    return env


def _ensure_code_dir_on_pythonpath(env: dict[str, str], code_dir: Path) -> None:
    """torchrun 以 ``executors/*.py`` 为入口时，sys.path 首项为 ``code/executors``，无法 ``import executors``。"""
    root = str(code_dir.resolve())
    prev = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"


def _print_startup_runtime_env(cfg: ResolvedConfig) -> None:
    print(
        "[startup_runtime_env] "
        f"launcher_requested={getattr(cfg, 'launcher_env_requested_json', '')} "
        f"launcher_effective={getattr(cfg, 'launcher_env_effective_json', '')} "
        f"thread_requested={getattr(cfg, 'thread_env_requested_json', '')} "
        f"thread_effective={getattr(cfg, 'thread_env_effective_json', '')}",
        flush=True,
    )


def _maybe_write_run_manifest(cfg: ResolvedConfig) -> None:
    """torchrun 前落盘，保证训练崩溃后仍能拿到本次解析结果。"""
    if not should_write_manifest_json():
        return
    data = build_run_manifest(cfg)
    p = write_run_manifest_json(cfg, data)
    print(f"[Manifest] wrote {p}", flush=True)


def _run_torchrun(
    cfg: ResolvedConfig,
    *,
    env_extra: dict[str, str],
    script: str,
    py_args: list[str],
) -> None:
    env = dict(_base_env_raw(cfg.repo_root))
    env.update(_d4c_layout_env(cfg))
    env.update(_torchrun_hardware_env(cfg))
    env.update(env_extra)
    _ensure_code_dir_on_pythonpath(env, cfg.code_dir)
    cmd = [
        *_torchrun_cmd(),
        "--standalone",
        f"--nproc_per_node={cfg.ddp_world_size}",
        script,
        *py_args,
    ]
    print("[d4c] cwd:", cfg.code_dir, flush=True)
    print("[d4c] exec:", " ".join(cmd), flush=True)
    _print_startup_runtime_env(cfg)
    subprocess.run(cmd, cwd=str(cfg.code_dir), env=env, check=True)


def _run_step3_train(cfg: ResolvedConfig) -> None:
    assert cfg.run_name is not None
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = str(Path(cfg.log_dir) / "train.log")
    model_path = str(Path(cfg.checkpoint_dir) / "model" / "model.pth")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    py_args = [
        "train",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--batch-size",
        str(cfg.train_batch_size),
        "--epochs",
        str(cfg.epochs),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--learning_rate",
        str(cfg.learning_rate),
        "--coef",
        str(cfg.coef),
        "--adv",
        str(cfg.adv),
        "--log_file",
        log_file,
        "--save_file",
        model_path,
    ]
    _run_torchrun(cfg, env_extra={}, script=TORCHRUN_STEP3_SCRIPT, py_args=py_args)


def _run_step3_eval(cfg: ResolvedConfig) -> None:
    assert cfg.run_name is not None
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = str(Path(cfg.log_dir) / "eval.log")
    model_path = str(Path(cfg.checkpoint_dir) / "model" / "model.pth")
    py_args = [
        "eval",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--batch-size",
        str(cfg.train_batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
        "--save_file",
        model_path,
    ]
    _run_torchrun(cfg, env_extra={}, script=TORCHRUN_STEP3_SCRIPT, py_args=py_args)


def run_step3(cfg: ResolvedConfig) -> None:
    assert cfg.run_name is not None
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)

    mode = cfg.step3_mode
    if mode == "eval_only":
        _run_step3_eval(cfg)
        return
    _run_step3_train(cfg)
    if mode == "full":
        _run_step3_eval(cfg)


def run_step4(cfg: ResolvedConfig) -> None:
    assert cfg.from_run is not None
    if cfg.global_eval_batch_size is None:
        raise RuntimeError(
            "内部错误: step4 缺少 global_eval_batch_size；应在 config_loader 中由 --eval-profile 解析 eval_batch_size。"
        )
    if cfg.eval_per_gpu_batch_size is None:
        raise RuntimeError("内部错误: step4 缺少 eval_per_gpu_batch_size（应与 global_eval_batch_size 同时解析）。")
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = str(Path(cfg.log_dir) / "step4.log")

    g_eval = int(cfg.global_eval_batch_size)
    e_per_gpu = int(cfg.eval_per_gpu_batch_size)
    env_extra = dict(_d4c_profile_env(cfg))
    env_extra["D4C_GLOBAL_EVAL_BATCH_SIZE"] = str(g_eval)
    env_extra["D4C_EVAL_PER_GPU_BATCH_SIZE"] = str(e_per_gpu)
    py_args = [
        "--task",
        str(cfg.task_id),
        "--batch-size",
        str(g_eval),
        "--num-proc",
        str(cfg.num_proc),
        "--log_file",
        log_file,
    ]
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP4_SCRIPT, py_args=py_args)


def _step5_decode_cli_args(cfg: ResolvedConfig) -> list[str]:
    """与 presets/decode 解析结果对齐，传给 step5_entry（train/eval）。"""
    out: list[str] = [
        "--decode-strategy",
        str(cfg.decode_strategy),
        "--max-explanation-length",
        str(cfg.max_explanation_length),
        "--label-smoothing",
        str(cfg.label_smoothing),
        "--repetition-penalty",
        str(cfg.repetition_penalty),
        "--generate-temperature",
        str(cfg.generate_temperature),
        "--generate-top-p",
        str(cfg.generate_top_p),
    ]
    if cfg.decode_seed is not None:
        out.extend(["--decode-seed", str(cfg.decode_seed)])
    if cfg.no_repeat_ngram_size is not None:
        out.extend(["--no-repeat-ngram-size", str(cfg.no_repeat_ngram_size)])
    if cfg.min_len is not None:
        out.extend(["--min-len", str(cfg.min_len)])
    return out


def _d4c_profile_env(cfg: ResolvedConfig) -> dict[str, str]:
    out: dict[str, str] = {
        "D4C_DECODE_PROFILE_JSON": cfg.decode_profile_json,
        "D4C_RERANK_PROFILE_JSON": cfg.rerank_profile_json,
        "D4C_DECODE_PRESET_STEM": str(cfg.decode_preset_id),
        "D4C_RERANK_PRESET_STEM": str(cfg.rerank_preset_id or ""),
    }
    if getattr(cfg, "eval_profile_id", "") and cfg.command in (
        "eval",
        "eval-rerank",
        "eval-matrix",
        "eval-rerank-matrix",
        "step4",
    ):
        out["D4C_EVAL_PROFILE_NAME"] = str(cfg.eval_profile_id)
    return out


def _torchrun_hardware_env(cfg: ResolvedConfig) -> dict[str, str]:
    """显式注入子进程：hardware 语义切片 JSON + stem + 线程/CUDA 等 launcher env（不经由训练语义 JSON）。"""
    out: dict[str, str] = {
        "D4C_HARDWARE_PROFILE_JSON": cfg.hardware_profile_json,
        "D4C_HARDWARE_PRESET": str(cfg.hardware_preset_id),
        "OMP_NUM_THREADS": str(int(cfg.omp_num_threads)),
        "MKL_NUM_THREADS": str(int(cfg.mkl_num_threads)),
        "TOKENIZERS_PARALLELISM": "true" if cfg.tokenizers_parallelism else "false",
    }
    try:
        _le = json.loads(getattr(cfg, "launcher_env_effective_json", "") or "{}")
    except json.JSONDecodeError:
        _le = {}
    if isinstance(_le, dict):
        cvd = _le.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and str(cvd).strip() != "":
            out["CUDA_VISIBLE_DEVICES"] = str(cvd).strip()
    return out


def run_step5(cfg: ResolvedConfig) -> None:
    assert cfg.from_run is not None and cfg.step5_run is not None
    ensure_step5_csv_symlink(cfg)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = str(Path(cfg.log_dir) / "train.log")

    env_extra = dict(_d4c_profile_env(cfg))
    model_path = str(path_layout.best_mainline_model_path(Path(cfg.checkpoint_dir)))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    py_args = [
        "train",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--batch-size",
        str(cfg.train_batch_size),
        "--epochs",
        str(cfg.epochs),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--learning_rate",
        str(cfg.learning_rate),
        "--coef",
        str(cfg.coef),
        "--eta",
        str(cfg.eta),
        "--log_file",
        log_file,
        "--save_file",
        model_path,
        *_step5_decode_cli_args(cfg),
    ]
    if cfg.step5_train_only:
        py_args.append("--train-only")
        stub = Path(cfg.log_dir) / "eval.log"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(
            "step5 --train-only：本次跳过训练后 valid 评估；完整指标请运行: python code/d4c.py eval …\n",
            encoding="utf-8",
        )
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args)


def run_eval(cfg: ResolvedConfig) -> None:
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = str(Path(cfg.log_dir) / "eval.log")

    mp = Path(cfg.model_path).expanduser().resolve() if cfg.model_path else None
    if mp is None:
        assert cfg.from_run is not None and cfg.step5_run is not None
        ck = Path(cfg.checkpoint_dir)
        if str(getattr(cfg, "checkpoint_kind", "best_mainline") or "best_mainline") == "last":
            mp = path_layout.last_model_path(ck)
        else:
            mp = path_layout.best_mainline_model_path(ck)

    if not mp.is_file():
        raise FileNotFoundError(f"评测权重不存在: {mp}")

    env_extra: dict[str, str] = dict(_d4c_profile_env(cfg))

    py_args = [
        "eval",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--eval-batch-size",
        str(cfg.global_eval_batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
        "--save_file",
        str(mp),
        *_step5_decode_cli_args(cfg),
    ]
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args)


def _rerank_runner_cli_args(cfg: ResolvedConfig) -> list[str]:
    out = [
        "--num-return-sequences",
        str(cfg.num_return_sequences),
        "--rerank-method",
        str(cfg.rerank_method),
        "--rerank-top-k",
        str(cfg.rerank_top_k),
        "--rerank-weight-logprob",
        str(cfg.rerank_weight_logprob),
        "--rerank-weight-length",
        str(cfg.rerank_weight_length),
        "--rerank-weight-repeat",
        str(cfg.rerank_weight_repeat),
        "--rerank-weight-dirty",
        str(cfg.rerank_weight_dirty),
        "--rerank-target-len-ratio",
        str(cfg.rerank_target_len_ratio),
        "--export-examples-mode",
        str(cfg.export_examples_mode),
        "--rerank-malformed-tail-penalty",
        str(cfg.rerank_malformed_tail_penalty),
        "--rerank-malformed-token-penalty",
        str(cfg.rerank_malformed_token_penalty),
    ]
    if cfg.export_full_rerank_examples:
        out.append("--export-full-rerank-examples")
    return out


def run_eval_rerank(cfg: ResolvedConfig) -> None:
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = str(Path(cfg.log_dir) / "eval.log")

    mp = Path(cfg.model_path).expanduser().resolve() if cfg.model_path else None
    if mp is None:
        assert cfg.from_run is not None and cfg.step5_run is not None
        ck = Path(cfg.checkpoint_dir)
        if str(getattr(cfg, "checkpoint_kind", "best_mainline") or "best_mainline") == "last":
            mp = path_layout.last_model_path(ck)
        else:
            mp = path_layout.best_mainline_model_path(ck)

    if not mp.is_file():
        raise FileNotFoundError(f"评测权重不存在: {mp}")

    env_extra: dict[str, str] = dict(_d4c_profile_env(cfg))

    py_args = [
        "eval-rerank",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--eval-batch-size",
        str(cfg.global_eval_batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
        "--save_file",
        str(mp),
        *_step5_decode_cli_args(cfg),
        *_rerank_runner_cli_args(cfg),
    ]
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args)


def run_pipeline(args: Any) -> ResolvedConfig | None:
    """返回当 ``--with-eval`` 时最后一次 eval 的 ResolvedConfig，否则 ``None``。

    ``--eval-profile`` 在 argparse 层为必填：Step4 与（非 --train-only）Step5 均依赖该 profile 的
    eval_batch_size / hardware 选择等。
    """
    if not getattr(args, "iteration_id", None):
        args.iteration_id = "v1"
    args.from_run = None
    args.eval_only = False
    args.train_only = False
    if getattr(args, "run_id", None) is None:
        args.run_id = "auto"
    cfg3 = load_resolved_config(args, "step3")
    fr = cfg3.run_name
    assert fr is not None
    run_step3(cfg3)

    args.from_run = fr
    cfg4 = load_resolved_config(args, "step4")
    run_step4(cfg4)

    args.step4_run = cfg4.step4_run

    if not getattr(args, "step5_run", None) or str(args.step5_run).strip().lower() in ("", "auto"):
        args.step5_run = "auto"
    args.preset = "step5"
    args.train_only = False
    print("[D4C Mainline] pipeline: Step5 已切换 preset -> 'step5'", flush=True)
    cfg5 = load_resolved_config(args, "step5")
    run_step5(cfg5)

    if not getattr(args, "with_eval", False):
        return None

    assert cfg5.step5_run is not None
    args.from_run = fr
    args.step5_run = cfg5.step5_run
    args.preset = "step5"
    cfg_eval = load_resolved_config(args, "eval")
    print_pre_run_banner("eval", cfg_eval)
    validate_resolved_config(cfg_eval)
    run_eval(cfg_eval)
    return cfg_eval


def run_smoke_ddp(repo_root: Path) -> None:
    """DDP 冒烟：使用 runs/task1/v0/… 隔离命名空间。"""
    code_dir = repo_root / "code"
    os.environ.setdefault("D4C_QUICK_EVAL_MAX_SAMPLES", "32")

    def _opt_int_env(name: str) -> int | None:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    # 可选：D4C_HARDWARE_PRESET → load_resolved_config 选用 presets/hardware/<name>.yaml（num_proc / dataloader 等）
    # D4C_SMOKE_BATCH_SIZE → 覆盖三段 torchrun 的全局 batch（默认 8）
    # DDP_NPROC 或 D4C_SMOKE_DDP_WORLD_SIZE → torchrun --nproc_per_node（默认单卡 1；未设此项时用 yaml 默认 ddp_world_size，多为 2）
    _rt = bool((os.environ.get("D4C_HARDWARE_PRESET") or "").strip())
    _smoke_bs = _opt_int_env("D4C_SMOKE_BATCH_SIZE")
    _ddp_env = _opt_int_env("DDP_NPROC")
    if _ddp_env is None:
        _ddp_env = _opt_int_env("D4C_SMOKE_DDP_WORLD_SIZE")

    class _Base:
        task = 1
        preset = "smoke_ddp"
        run_name = None
        iteration_id = "v0"
        run_id = "auto"
        analysis_pack = "auto"
        from_run = None
        step5_run = None
        step4_run = None
        train_csv = None
        model_path = None
        batch_size = None
        epochs = None
        num_proc = None
        seed = 3407
        ddp_world_size = None
        eta = None
        hardware_preset = None
        decode_preset = None
        eval_profile = None
        eval_batch_size = None
        eval_only = False
        train_only = False
        omp_num_threads = None
        mkl_num_threads = None
        tokenizers_parallelism = None
        cuda_visible_devices = None

    cfg3 = load_resolved_config(_Base(), "step3")
    aux, tgt = cfg3.auxiliary, cfg3.target
    bs, npc = str(cfg3.train_batch_size), str(cfg3.num_proc)
    ddp = cfg3.ddp_world_size
    log_train = str(Path(cfg3.log_dir) / "smoke_step3_train.log")

    print("========== DDP smoke: runs/task1/v0/train/step3/... ==========", flush=True)

    print(">>> Step 3 train (max-steps=2, --save-final-checkpoint)", flush=True)
    model_s3 = str(Path(cfg3.checkpoint_dir) / "model" / "model.pth")
    Path(model_s3).parent.mkdir(parents=True, exist_ok=True)
    py_train = [
        "train",
        "--auxiliary",
        aux,
        "--target",
        tgt,
        "--max-steps",
        "2",
        "--save-final-checkpoint",
        "--batch-size",
        bs,
        "--num-proc",
        npc,
        "--log_file",
        log_train,
        "--save_file",
        model_s3,
    ]
    _run_torchrun(cfg3, env_extra={}, script=TORCHRUN_STEP3_SCRIPT, py_args=py_train)

    if not Path(model_s3).is_file():
        raise FileNotFoundError(f"smoke Step3 未产出权重: {model_s3}")

    print(">>> Step 3 eval", flush=True)
    log_eval = str(Path(cfg3.log_dir) / "smoke_step3_eval.log")
    py_eval = [
        "eval",
        "--auxiliary",
        aux,
        "--target",
        tgt,
        "--batch-size",
        bs,
        "--num-proc",
        npc,
        "--log_file",
        log_eval,
        "--save_file",
        model_s3,
    ]
    _run_torchrun(cfg3, env_extra={}, script=TORCHRUN_STEP3_SCRIPT, py_args=py_eval)

    class _S4(_Base):
        from_run = cfg3.run_name
        eval_profile = "eval_fast_single_gpu"

    cfg4 = load_resolved_config(_S4(), "step4")
    log_s4 = str(Path(cfg4.log_dir) / "step4.log")
    print(">>> Step 4 runner (task 1)", flush=True)
    assert cfg4.global_eval_batch_size is not None and cfg4.eval_per_gpu_batch_size is not None
    g4 = int(cfg4.global_eval_batch_size)
    epg4 = int(cfg4.eval_per_gpu_batch_size)
    env_s4 = dict(_d4c_profile_env(cfg4))
    env_s4["D4C_GLOBAL_EVAL_BATCH_SIZE"] = str(g4)
    env_s4["D4C_EVAL_PER_GPU_BATCH_SIZE"] = str(epg4)
    py_s4 = [
        "--task",
        "1",
        "--batch-size",
        str(g4),
        "--num-proc",
        npc,
        "--log_file",
        log_s4,
    ]
    _run_torchrun(cfg4, env_extra=env_s4, script=TORCHRUN_STEP4_SCRIPT, py_args=py_s4)

    class _S5(_Base):
        from_run = cfg3.run_name
        step4_run = cfg4.step4_run
        step5_run = "auto"
        preset = "smoke_ddp"

    cfg5 = load_resolved_config(_S5(), "step5")
    log_s5 = str(Path(cfg5.log_dir) / "smoke_step5.log")
    model_s5 = str(path_layout.best_mainline_model_path(Path(cfg5.checkpoint_dir)))
    Path(model_s5).parent.mkdir(parents=True, exist_ok=True)
    ensure_step5_csv_symlink(cfg5)
    print(f">>> Step 5 runner (epochs=1, train-only, batch={cfg5.train_batch_size})", flush=True)
    py_s5 = [
        "train",
        "--auxiliary",
        aux,
        "--target",
        tgt,
        "--epochs",
        "1",
        "--train-only",
        "--batch-size",
        str(cfg5.train_batch_size),
        "--num-proc",
        npc,
        "--log_file",
        log_s5,
        "--save_file",
        model_s5,
    ]
    env5 = {
        "D4C_DECODE_PROFILE_JSON": cfg5.decode_profile_json,
        "D4C_RERANK_PROFILE_JSON": cfg5.rerank_profile_json,
    }
    _run_torchrun(cfg5, env_extra=env5, script=TORCHRUN_STEP5_SCRIPT, py_args=py_s5)

    print("========== DDP smoke 完成（仅验证不 crash） ==========", flush=True)
    print(f"Step3 产物: {cfg3.checkpoint_dir}", flush=True)
    print(f"Step5 产物: {cfg5.checkpoint_dir}", flush=True)
