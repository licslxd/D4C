"""MAINLINE 运行前共性校验：路径、参数组合、清晰错误信息。"""
from __future__ import annotations

from pathlib import Path

from d4c_core.config_loader import ResolvedConfig
from d4c_core import paths as d4c_paths

_DOC_MAIN = "docs/D4C_Scripts_and_Runtime_Guide.md"
_DOC_PRESETS = "docs/PRESETS.md"


def _hint_tail() -> str:
    return f"文档: {_DOC_MAIN}；配置: {_DOC_PRESETS}；解析快照见 stdout 或与 train.log 同目录的 d4c_run_manifest.json（默认 torchrun 前落盘，可用 D4C_WRITE_RUN_MANIFEST=0 关闭）。"


def validate_resolved_config(cfg: ResolvedConfig) -> None:
    """
    在 load_resolved_config 之后、torchrun 之前调用。
    不替代 YAML/任务表解析错误（仍由 config_loader 抛出）。
    """
    cmd = cfg.command
    ck = Path(cfg.checkpoint_dir)

    if cmd == "step4":
        if not ck.is_dir():
            raise FileNotFoundError(
                f"step4 需要已存在的 Step3 checkpoint 目录（--from-run 对应路径）:\n  {ck}\n"
                "请先完成 step3，或核对 --from-run 是否与 checkpoints/<task>/step3_optimized/<name> 一致。\n"
                + _hint_tail()
            )
    elif cmd == "step5":
        assert cfg.from_run is not None
        step3_dir = d4c_paths.resolve_step3_dir(cfg.repo_root, cfg.task_id, cfg.from_run)
        if not step3_dir.is_dir():
            raise FileNotFoundError(
                f"step5 需要已存在的 Step3 目录:\n  {step3_dir}\n"
                "请检查 --from-run；Step4 反事实 CSV 也应位于该目录下。\n"
                + _hint_tail()
            )
    elif cmd == "step3":
        if cfg.step3_mode == "eval_only":
            mp = ck / "model.pth"
            if not mp.is_file():
                raise FileNotFoundError(
                    f"step3 --eval-only 需要已有权重:\n  {mp}\n"
                    "请先跑完整 step3 训练，或去掉 --eval-only。\n"
                    f"主日志通常在: {cfg.log_dir}/train.log\n"
                    + _hint_tail()
                )
    elif cmd == "eval":
        if cfg.model_path:
            mp = Path(cfg.model_path)
            if not mp.is_file():
                raise FileNotFoundError(
                    f"eval --model-path 不是有效文件:\n  {mp}\n"
                    "请核对路径或改用 --from-run + --step5-run。\n"
                    + _hint_tail()
                )
        else:
            mp = ck / "model.pth"
            if not mp.is_file():
                raise FileNotFoundError(
                    f"eval 需要 Step5 权重:\n  {mp}\n"
                    "请确认已完成 step5，且 --from-run / --step5-run 与训练时一致。\n"
                    f"可查看: {cfg.log_dir}/ 与 checkpoint 目录。\n"
                    + _hint_tail()
                )
