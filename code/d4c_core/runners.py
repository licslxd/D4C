"""torchrun 子进程编排：供 ``d4c.py`` 调用；INTERNAL EXECUTOR 仅出现在本子模块。"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from d4c_core.artifacts import ensure_step5_csv_symlink
from d4c_core.config_loader import ResolvedConfig, load_resolved_config
from d4c_core.dispatch import TORCHRUN_STEP3_SCRIPT, TORCHRUN_STEP4_SCRIPT, TORCHRUN_STEP5_SCRIPT
from d4c_core.manifests import build_run_manifest, should_write_manifest_json, write_run_manifest_json


def _torchrun_cmd() -> list[str]:
    if shutil.which("torchrun"):
        return ["torchrun"]
    return [sys.executable, "-m", "torch.distributed.run"]


def _scrub_d4c_env(env: dict[str, str]) -> None:
    for k in list(env.keys()):
        if k.startswith("D4C_"):
            del env[k]


def _scrub_training_side_env(env: dict[str, str]) -> None:
    """子进程内避免继承 shell 的 TRAIN_/EVAL 等覆盖，使行为仅由 CLI + 代码默认决定。"""
    _scrub_d4c_env(env)
    for k in list(env.keys()):
        if k.startswith("TRAIN_"):
            del env[k]
    for k in ("EVAL_BATCH_SIZE", "MAX_PARALLEL_CPU"):
        env.pop(k, None)


def _base_env(cfg: ResolvedConfig) -> dict[str, str]:
    env = dict(os.environ)
    _scrub_training_side_env(env)
    env["D4C_ROOT"] = str(cfg.repo_root)
    return env


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
    print("[d4c] cwd:", code_dir, flush=True)
    print("[d4c] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(code_dir), env=env, check=True)


def _base_env_raw(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    _scrub_training_side_env(env)
    env["D4C_ROOT"] = str(repo_root)
    return env


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
    _run_torchrun_explicit(
        code_dir=cfg.code_dir,
        repo_root=cfg.repo_root,
        ddp_world_size=cfg.ddp_world_size,
        env_extra=env_extra,
        script=script,
        py_args=py_args,
    )


def _step3_env_extra(cfg: ResolvedConfig) -> dict[str, str]:
    assert cfg.run_name is not None
    return {
        "D4C_CHECKPOINT_GROUP": "step3_optimized",
        "D4C_CHECKPOINT_SUBDIR": cfg.run_name,
        "D4C_LOG_GROUP": "step3_optimized",
    }


def _run_step3_train(cfg: ResolvedConfig) -> None:
    assert cfg.run_name is not None
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = str(Path(cfg.log_dir) / "train.log")

    py_args = [
        "train",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--batch-size",
        str(cfg.batch_size),
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
    ]
    _run_torchrun(cfg, env_extra=_step3_env_extra(cfg), script=TORCHRUN_STEP3_SCRIPT, py_args=py_args)


def _run_step3_eval(cfg: ResolvedConfig) -> None:
    assert cfg.run_name is not None
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = str(Path(cfg.log_dir) / "train.log")
    py_args = [
        "eval",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--batch-size",
        str(cfg.batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
    ]
    _run_torchrun(cfg, env_extra=_step3_env_extra(cfg), script=TORCHRUN_STEP3_SCRIPT, py_args=py_args)


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
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = str(Path(cfg.log_dir) / "train.log")

    env_extra = {
        "D4C_CHECKPOINT_GROUP": "step3_optimized",
        "D4C_CHECKPOINT_SUBDIR": cfg.from_run,
        "D4C_LOG_STEP": "step4_optimized",
    }
    py_args = [
        "--task",
        str(cfg.task_id),
        "--batch-size",
        str(cfg.batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--log_file",
        log_file,
    ]
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP4_SCRIPT, py_args=py_args)


def _step5_extra_argv_from_env() -> list[str]:
    raw = (os.environ.get("D4C_RUN_D4C_EXTRA") or "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def _step5_decode_cli_args(cfg: ResolvedConfig) -> list[str]:
    """与 presets/decode 解析结果对齐，传给 run-d4c step5/eval。"""
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
    return out


def run_step5(cfg: ResolvedConfig) -> None:
    assert cfg.from_run is not None and cfg.step5_run is not None
    ensure_step5_csv_symlink(cfg)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = str(Path(cfg.log_dir) / "train.log")

    env_extra = {
        "D4C_CHECKPOINT_GROUP": "step3_optimized",
        "D4C_CHECKPOINT_SUBDIR": f"{cfg.from_run}/step5/{cfg.step5_run}",
        "D4C_LOG_GROUP": "step5_optimized",
    }
    model_path = str(Path(cfg.checkpoint_dir) / "model.pth")
    py_args = [
        "train",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--batch-size",
        str(cfg.batch_size),
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
    py_args.extend(_step5_extra_argv_from_env())
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args)


def run_eval(cfg: ResolvedConfig) -> None:
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = str(Path(cfg.log_dir) / "eval.log")

    mp = Path(cfg.model_path).expanduser().resolve() if cfg.model_path else None
    if mp is None:
        assert cfg.from_run is not None and cfg.step5_run is not None
        mp = Path(cfg.checkpoint_dir) / "model.pth"

    if not mp.is_file():
        raise FileNotFoundError(f"评测权重不存在: {mp}")

    env_extra: dict[str, str] = {}
    if cfg.from_run is not None and cfg.step5_run is not None:
        env_extra = {
            "D4C_CHECKPOINT_GROUP": "step3_optimized",
            "D4C_CHECKPOINT_SUBDIR": f"{cfg.from_run}/step5/{cfg.step5_run}",
            "D4C_LOG_GROUP": "step5_optimized",
        }

    py_args = [
        "eval",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--eval-batch-size",
        str(cfg.batch_size),
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
    py_args.extend(_step5_extra_argv_from_env())
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args)


def run_pipeline(args: Any) -> None:
    def _ts(prefix: str) -> str:
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    pipeline_preset = args.preset

    rn = args.run_name or _ts("step3_opt")
    s5 = args.step5_run or _ts("step5_opt")
    fr = rn

    args.run_name = rn
    args.from_run = None
    args.eval_only = False
    args.train_only = False
    cfg3 = load_resolved_config(args, "step3")
    run_step3(cfg3)

    args.from_run = fr
    cfg4 = load_resolved_config(args, "step4")
    run_step4(cfg4)

    args.step5_run = s5
    args.preset = "step5"
    args.train_only = False
    print("[D4C Mainline] pipeline: Step5 已切换 preset -> 'step5'", flush=True)
    cfg5 = load_resolved_config(args, "step5")
    run_step5(cfg5)


def run_smoke_ddp(repo_root: Path) -> None:
    """DDP 冒烟：与 sh/smoke_test_ddp.sh 一致（扁平 smoke_ddp/<tag>/，Step5 不传 --save_file）。"""
    code_dir = repo_root / "code"
    smoke_tag = f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    os.environ.setdefault("D4C_QUICK_EVAL_MAX_SAMPLES", "32")

    class _Args:
        task = 1
        preset = "step3"
        run_name = smoke_tag
        from_run = None
        step5_run = None
        train_csv = None
        model_path = None
        batch_size = 8
        epochs = 1
        num_proc = 2
        seed = 3407
        ddp_world_size = 1
        eta = None
        decode_preset = "default"
        eval_only = False
        train_only = False

    cfg = load_resolved_config(_Args(), "step3")

    env_smoke = {
        "D4C_CHECKPOINT_GROUP": "smoke_ddp",
        "D4C_CHECKPOINT_SUBDIR": smoke_tag,
        "D4C_LOG_GROUP": "smoke_ddp",
    }
    run_dir = repo_root / "log" / "1" / "smoke_ddp" / "runs" / smoke_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_train = run_dir / "step3_train.log"
    log_eval = run_dir / "step3_eval.log"
    log_s4 = run_dir / "step4.log"
    log_s5 = run_dir / "step5.log"

    aux, tgt = cfg.auxiliary, cfg.target
    bs, npc = str(cfg.batch_size), str(cfg.num_proc)
    ddp = cfg.ddp_world_size

    print(f"========== DDP smoke: CHECKPOINT_SUBDIR={smoke_tag} ==========", flush=True)

    print(">>> Step 3 train (max-steps=2, --save-final-checkpoint)", flush=True)
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
        str(log_train),
    ]
    _run_torchrun_explicit(
        code_dir=code_dir,
        repo_root=repo_root,
        ddp_world_size=ddp,
        env_extra=env_smoke,
        script=TORCHRUN_STEP3_SCRIPT,
        py_args=py_train,
    )

    smoke_ckpt = repo_root / "checkpoints" / "1" / "smoke_ddp" / smoke_tag / "model.pth"
    if not smoke_ckpt.is_file():
        raise FileNotFoundError(
            f"smoke Step3 未产出 model.pth: {smoke_ckpt}（需 --save-final-checkpoint 与 max-steps）"
        )

    print(">>> Step 3 eval", flush=True)
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
        str(log_eval),
    ]
    _run_torchrun_explicit(
        code_dir=code_dir,
        repo_root=repo_root,
        ddp_world_size=ddp,
        env_extra=env_smoke,
        script=TORCHRUN_STEP3_SCRIPT,
        py_args=py_eval,
    )

    print(">>> Step 4 runner (task 1)", flush=True)
    py_s4 = [
        "--task",
        "1",
        "--batch-size",
        bs,
        "--num-proc",
        npc,
        "--log_file",
        str(log_s4),
    ]
    _run_torchrun_explicit(
        code_dir=code_dir,
        repo_root=repo_root,
        ddp_world_size=ddp,
        env_extra=env_smoke,
        script=TORCHRUN_STEP4_SCRIPT,
        py_args=py_s4,
    )

    print(">>> Step 5 runner (epochs=1, train-only, batch=32)", flush=True)
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
        "32",
        "--num-proc",
        npc,
        "--log_file",
        str(log_s5),
    ]
    _run_torchrun_explicit(
        code_dir=code_dir,
        repo_root=repo_root,
        ddp_world_size=ddp,
        env_extra=env_smoke,
        script=TORCHRUN_STEP5_SCRIPT,
        py_args=py_s5,
    )

    print("========== DDP smoke 完成（仅验证不 crash） ==========", flush=True)
    print(f"产物: {repo_root}/checkpoints/1/smoke_ddp/{smoke_tag}/", flush=True)
    print(f"日志: {run_dir}/", flush=True)
