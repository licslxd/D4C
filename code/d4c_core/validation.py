"""MAINLINE 运行前共性校验：路径、参数组合、清晰错误信息。"""
from __future__ import annotations

from pathlib import Path

from d4c_core import path_layout, run_naming
from d4c_core.artifacts import train_csv_path
from d4c_core.config_loader import ResolvedConfig

_DOC_MAIN = "docs/D4C_Scripts_and_Runtime_Guide.md"
_DOC_PRESETS = "docs/PRESETS.md"


def _hint_tail() -> str:
    return f"文档: {_DOC_MAIN}；配置: {_DOC_PRESETS}；解析快照见 stdout 或与当次 run 同目录的 manifest.json（可用 D4C_WRITE_RUN_MANIFEST=0 关闭）。"


def validate_resolved_config(cfg: ResolvedConfig) -> None:
    """
    在 load_resolved_config 之后、torchrun 之前调用。
    不替代 YAML/任务表解析错误（仍由 config_loader 抛出）。
    """
    cmd = cfg.command
    if cmd in ("step3", "step5", "eval", "eval-rerank"):
        if not (getattr(cfg, "effective_training_payload_json", "") or "").strip():
            raise RuntimeError(
                f"内部错误: command={cmd!r} 缺少 effective_training_payload_json（父进程须生成训练 payload）。"
            )
    ck = Path(cfg.checkpoint_dir)
    it = cfg.iteration_id
    t = cfg.task_id

    if cmd == "step4":
        s3 = Path(cfg.step3_checkpoint_dir or "")
        s3_model = path_layout.model_file_path(s3)
        if not s3.is_dir():
            raise FileNotFoundError(
                "step4 需要已存在的 Step3 run 目录（--from-run 对应 runs/task{T}/vN/train/step3/<run>/）：\n"
                f"  期望目录: {s3}\n"
                f"  task={t} iter={it} from_run={cfg.from_run!r}\n"
                "请先完成 step3，或核对 --from-run 是否为 train/step3 下的 slug（如 1、2）。\n"
                + _hint_tail()
            )
        if not s3_model.is_file():
            raise FileNotFoundError(
                "step4 需要 Step3 已产出的权重 model/model.pth：\n"
                f"  {s3_model}\n"
                f"  task={t} iter={it} step3_run={cfg.from_run!r}\n"
                "请先跑完 step3 训练，或检查上述路径。\n"
                + _hint_tail()
            )
        _eid = (getattr(cfg, "eval_profile_id", "") or "").strip()
        if not _eid:
            raise RuntimeError(
                "内部错误: step4 缺少 eval_profile_id（须由 CLI --eval-profile 解析；step4 已归入 eval 语义侧）。"
            )
        if cfg.global_eval_batch_size is None:
            raise RuntimeError(
                "内部错误: step4 缺少 global_eval_batch_size（须由 eval_profile.eval_batch_size 解析）。"
            )
        if int(cfg.global_eval_batch_size) % int(cfg.ddp_world_size) != 0:
            raise ValueError(
                f"step4: eval_profile「{_eid}」中的 eval_batch_size={int(cfg.global_eval_batch_size)} "
                f"不能整除当前 hardware 预设的 ddp_world_size={int(cfg.ddp_world_size)}。\n"
                f"请修改 presets/eval_profiles/{_eid}.yaml 的 eval_batch_size，"
                f"或修改 presets/hardware/{cfg.hardware_preset_id}.yaml 的 ddp_world_size。\n"
                + _hint_tail()
            )
    elif cmd == "step5":
        assert cfg.from_run is not None
        rid = run_naming.parse_run_id(cfg.from_run)
        step3_dir = path_layout.get_train_step3_run_root(cfg.repo_root, cfg.task_id, cfg.iteration_id, rid)
        if not step3_dir.is_dir():
            raise FileNotFoundError(
                f"step5 需要已存在的 Step3 目录:\n  {step3_dir}\n"
                f"task={t} iter={it} from_run={cfg.from_run!r}\n"
                "请检查 --from-run。\n"
                + _hint_tail()
            )
        assert cfg.step5_run is not None
        s4_slug = run_naming.step4_slug_from_step5_slug(cfg.step5_run)
        step4_dir = path_layout.get_train_step4_run_root(
            cfg.repo_root, cfg.task_id, cfg.iteration_id, s4_slug
        )
        csv_p = train_csv_path(cfg)
        if not csv_p.is_file():
            raise FileNotFoundError(
                "step5 需要 Step4 反事实 CSV（由 step5_run 经 run_naming 反推 step4）：\n"
                f"  期望文件: {csv_p}\n"
                f"  task={t} iter={it} step5_run={cfg.step5_run!r} → step4_run={s4_slug!r}\n"
                f"  对应目录应为: {step4_dir}\n"
                "请先完成 step4，或使 Step5 目录名与 Step4 一致（形如 {{step4}}_{{n}}，例如 step4=2_1 → step5=2_1_1）。\n"
                + _hint_tail()
            )
    elif cmd == "step3":
        if cfg.step3_mode == "eval_only":
            mp = path_layout.model_file_path(ck)
            if not mp.is_file():
                raise FileNotFoundError(
                    f"step3 --eval-only 需要已有权重:\n  {mp}\n"
                    "请先跑完整 step3 训练，或去掉 --eval-only。\n"
                    f"评测日志: {cfg.log_dir}/eval.log\n"
                    + _hint_tail()
                )
    elif cmd in ("eval", "eval-rerank"):
        if cfg.global_eval_batch_size is None:
            raise RuntimeError("内部错误: eval 系命令缺少 global_eval_batch_size。")
        if int(cfg.global_eval_batch_size) % int(cfg.ddp_world_size) != 0:
            raise ValueError(
                f"eval_batch_size={int(cfg.global_eval_batch_size)} 与 world_size={int(cfg.ddp_world_size)} 不整除。"
                "请修改 presets/eval_profiles/*.yaml 的 eval_batch_size，或调整 presets/hardware/*.yaml 的 ddp_world_size。"
            )
        stage = "rerank" if cmd == "eval-rerank" else "eval"
        if cfg.model_path:
            mp = Path(cfg.model_path)
            if not mp.is_file():
                raise FileNotFoundError(
                    f"eval --model-path 不是有效文件:\n  {mp}\n"
                    "请核对路径或改用 --from-run + --step5-run。\n"
                    + _hint_tail()
                )
        else:
            mp = path_layout.model_file_path(ck)
            if not mp.is_file():
                er = cfg.eval_run_dir or "(尚未分配)"
                raise FileNotFoundError(
                    f"{cmd} 需要 Step5 训练权重（默认 model/best_mainline.pth）：\n"
                    f"  {mp}\n"
                    f"  task={t} iter={it} from_run={cfg.from_run!r} step5_run={cfg.step5_run!r}\n"
                    f"  本次将写入: runs/task{t}/{it}/{stage}/<run>/（当前 eval_run_dir={er}）\n"
                    "请确认已完成 step5，且 --from-run / --step5-run 与训练时一致。\n"
                    + _hint_tail()
                )
