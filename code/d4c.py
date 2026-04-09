#!/usr/bin/env python3
"""
MAINLINE ENTRY — D4C 唯一推荐的用户级 Python 入口。

在项目根目录执行::

  python code/d4c.py step3 --task 1 --preset step3 [--run-name NAME]
  python code/d4c.py step4 --task 1 --preset step3 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu
  python code/d4c.py step5 --task 1 --preset step5 --from-run ... --step5-run ...
  python code/d4c.py eval --task 1 --preset step5 --from-run ... --step5-run ...
  python code/d4c.py pipeline --task 1 --preset step3

各阶段由 ``d4c_core.runners`` 经 torchrun 分发到 **step3 / step4 / step5 runner**（实现位于
``code/executors/``）；shell 与日常实验请只使用本入口。排障时可设环境变量 ``D4C_DISPATCH_DETAIL=1``
查看 torchrun 实际加载的 ``executors/step*_entry.py`` 路径。

**解码（decode）**：step5 / eval 系使用 ``--decode-preset <stem>``（见 ``presets/decode/``）；step3/step4 不加载 decode。
勿将 ``--decode-strategy`` / ``--generate-temperature`` 等 **step5 runner 内部参数** 直接接在本入口后。
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from d4c_core import path_layout, run_naming
from d4c_core.config_loader import ResolvedConfig, load_resolved_config
from d4c_core.logging_meta import (
    print_pipeline_opening,
    print_pre_run_banner,
    print_smoke_ddp_preamble,
)
from d4c_core.runners import (
    run_eval,
    run_eval_rerank,
    run_pipeline,
    run_smoke_ddp,
    run_step3,
    run_step4,
    run_step5,
)
from d4c_core.validation import validate_resolved_config

_EPILOG = """
身份说明
  · 本命令 (code/d4c.py)     — MAINLINE ENTRY，唯一推荐的 Python 用户入口。
  · scripts/entrypoints/*.sh — 编排层（应调用本文件；由 d4c 再 torchrun 到阶段 runner）。
  · executors/step{3,4,5}_entry.py — 仅供 torchrun 加载；日常只记本入口，排障见 D4C_DISPATCH_DETAIL。
  · legacy/（仓库根）      — 考古区（legacy/code、legacy/sh、legacy/tools），不参与主线。

最小示例（仓库根目录）
  python code/d4c.py step3 --task 4 --preset step3 --iter v1
  python code/d4c.py step4 --task 4 --preset step3 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu
  python code/d4c.py step5 --task 4 --preset step5 --iter v1 --from-run 2 --step4-run 2_1 --step5-run auto --eval-profile eval_fast_single_gpu
  python code/d4c.py eval --task 4 --preset step5 --iter v1 --from-run 2 --step5-run 2_1_1 --eval-profile eval_balanced_2gpu
  python code/d4c.py pipeline --task 4 --preset step3 --iter v1 --eval-profile eval_fast_single_gpu
  python code/d4c.py pipeline --task 4 --preset step3 --iter v1 --eval-profile eval_fast_single_gpu --with-eval

pipeline 预设规则
  · Step3 与 Step4 使用命令行 --preset（同一训练预设名，如 step3）。
  · Step5 在流水线内强制使用预设名 step5（加载 presets/training/step5.yaml），与 Step3 的 --preset 分离。

运行清单（manifest）
  · 默认在 torchrun 前写入当次 run 目录下的 manifest.json（与 train/eval 产物同目录）。
  · 关闭 JSON：export D4C_WRITE_RUN_MANIFEST=0
  · 命令行会写入 manifest 的 cli_invocation 字段（供复现）；详见 README「输出产物」。

子进程环境（重要）
  · torchrun 启动前会清洗父 shell 中的 TRAIN_*、EVAL_BATCH_SIZE、MAX_PARALLEL_CPU 及全部 D4C_*，
    再由 d4c 显式注入：布局变量 + D4C_HARDWARE_PROFILE_JSON / D4C_HARDWARE_PRESET + OMP/MKL/TOKENIZERS_PARALLELISM
    + decode/rerank JSON、语义指纹等。勿依赖父 shell export 碰运气覆盖主线。
  · 评测推荐 --eval-profile；亦可显式 --hardware-preset + --decode-preset（见 README）。

Analysis pack（发给 AI 的打包目录）
  · eval / eval-rerank / eval-matrix / eval-rerank-matrix 默认在结束后写入 runs/.../vN/analysis/packNN/
  · 关闭：对上述子命令加 --analysis-pack off
  · 手动：python code/d4c.py analysis-pack --task N --iter vN [--eval-run …] [--rerank-run …] [--matrix-run …]

Baselines（冻结锚点，不重跑源 eval）
  · python code/d4c.py register-baseline --task 4 --iter v1 --baseline-id ID --source-eval-dir runs/task4/v1/eval/1 [--set-default]
  · 登记与快照在 runs/task{T}/vN/baselines/；summary/compare 请用 d4c_core.baseline_registry

解码与 eval profile（重要）
  · 推荐 --eval-profile <stem>（presets/eval_profiles/）：**编排层**，只选 hardware/decode/rerank preset + profile-owned（eval_batch_size、num_return_sequences），不是与 hardware 同级的 merge YAML 层。
  · step4 与 eval/eval-rerank 同属 eval 语义侧：须 --eval-profile，推理全局 batch 仅来自 profile 的 eval_batch_size（不回退 train_batch_size）。
  · step5（非 --train-only）/ eval* 使用 --decode-preset <stem>（合并 presets/decode/default.yaml + overlays）；step3/step4 不加载 decode 预设文件；step5 --train-only 不加载 decode。
  · 勿使用 --decode-strategy / --generate-temperature 等顶层 flag；由 decode preset 或子进程 step5 解析。
"""


# 仅属于 step5 runner（torchrun 内 step5_entry）的 decode 相关参数；若出现在本入口 argv 中多为历史误用。
_LEGACY_STEP5_DECODE_FLAGS: frozenset[str] = frozenset(
    {
        "--decode-strategy",
        "--generate-temperature",
        "--generate-top-p",
        "--decode-seed",
        "--repetition-penalty",
        "--max-explanation-length",
    }
)


def _legacy_step5_decode_flags_in_argv(argv_tail: list[str]) -> list[str]:
    """若 argv（不含程序名）中含下游 decode flag，返回去重后的 flag 列表（有序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for a in argv_tail:
        if a == "--":
            break
        name: str
        if a.startswith("--") and "=" in a:
            name = a.split("=", 1)[0]
        else:
            name = a
        if name in _LEGACY_STEP5_DECODE_FLAGS and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _fail_legacy_decode_flags(flags: list[str]) -> None:
    joined = ", ".join(flags)
    preset_examples = "decode_greedy_default, decode_balanced_v2, decode_diverse_v2"
    msg = f"""错误: 在 code/d4c.py（MAINLINE 顶层入口）中使用了下游 step5 runner 专用参数: {joined}

这些参数由 torchrun 内 step5_entry 在子进程内解析，不应直接写在「python code/d4c.py …」之后。

请改用顶层入口（推荐 eval bundle）:
  python code/d4c.py eval --task … --preset … --from-run … --step5-run … --eval-profile <stem>

或显式解码预设:
  python code/d4c.py eval … --decode-preset <stem>

常用 decode <stem> 见 presets/decode/*.yaml，例如: {preset_examples}
说明见 docs/PRESETS.md §1.4 / §1.6。
"""
    print(msg.strip(), file=sys.stderr)
    raise SystemExit(2)


def _analysis_pack_disabled_flag(raw: object) -> bool:
    s = str(raw).strip().lower()
    return s in ("off", "none", "false", "0", "no")


def _maybe_export_analysis_pack(args: argparse.Namespace, cfg: ResolvedConfig) -> None:
    if cfg.command not in ("eval", "eval-rerank") or not cfg.eval_run_dir:
        return
    if _analysis_pack_disabled_flag(getattr(args, "analysis_pack", "auto")):
        return

    from d4c_core.analysis_pack import export_analysis_pack

    pack_req = getattr(args, "analysis_pack", "auto")
    pr = (
        None
        if pack_req is None or str(pack_req).strip().lower() in ("", "auto")
        else str(pack_req).strip()
    )
    export_analysis_pack(
        repo_root=cfg.repo_root,
        task_id=cfg.task_id,
        iteration_id=cfg.iteration_id,
        pack_id_req=pr,
        eval_run_dirs=[Path(cfg.eval_run_dir)] if cfg.command == "eval" else [],
        rerank_run_dirs=[Path(cfg.eval_run_dir)] if cfg.command == "eval-rerank" else [],
        matrix_run_dir=None,
    )


def _export_post_matrix_analysis_pack(
    args: argparse.Namespace,
    last_cfg: ResolvedConfig,
    matrix_out_d: Path,
    *,
    rerank: bool,
) -> None:
    """eval-matrix / eval-rerank-matrix 结束后：单包纳入 matrix 汇总 + 最后一次 eval/rerank 产物目录。"""
    if _analysis_pack_disabled_flag(getattr(args, "analysis_pack", "auto")):
        return

    from d4c_core.analysis_pack import export_analysis_pack

    pack_req = getattr(args, "analysis_pack", "auto")
    pr = (
        None
        if pack_req is None or str(pack_req).strip().lower() in ("", "auto")
        else str(pack_req).strip()
    )
    ev: list[Path] = []
    rr: list[Path] = []
    if last_cfg.eval_run_dir:
        p = Path(last_cfg.eval_run_dir)
        if rerank:
            rr = [p]
        else:
            ev = [p]
    export_analysis_pack(
        repo_root=last_cfg.repo_root,
        task_id=last_cfg.task_id,
        iteration_id=last_cfg.iteration_id,
        pack_id_req=pr,
        eval_run_dirs=ev,
        rerank_run_dirs=rr,
        matrix_run_dir=matrix_out_d,
    )


def _matrix_runtime_ns_fields(args: argparse.Namespace) -> dict:
    """eval-matrix / eval-rerank-matrix 循环内复刻与主入口一致的 runtime / launcher CLI。"""
    return {
        "omp_num_threads": getattr(args, "omp_num_threads", None),
        "mkl_num_threads": getattr(args, "mkl_num_threads", None),
        "tokenizers_parallelism": getattr(args, "tokenizers_parallelism", None),
        "cuda_visible_devices": getattr(args, "cuda_visible_devices", None),
    }


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--task", type=int, required=True, choices=range(1, 9), metavar="N")
    sp.add_argument(
        "--preset",
        "--training-preset",
        type=str,
        required=True,
        help="训练预设 → presets/training/<name>.yaml。pipeline 内 Step5 强制 step5。",
    )
    sp.add_argument("--run-name", type=str, default=None, dest="run_name")
    sp.add_argument(
        "--iter",
        type=str,
        default="v1",
        dest="iteration_id",
        help="迭代目录 vN，对应 runs/task{T}/vN/…",
    )
    sp.add_argument(
        "--run-id",
        type=str,
        default="auto",
        dest="run_id",
        help="当前阶段 run 目录名；auto 表示同级下一空目录（默认 1、2、…）；须为 slug（如 1、2_1）",
    )
    sp.add_argument(
        "--analysis-pack",
        type=str,
        default="auto",
        dest="analysis_pack",
        help="eval / eval-rerank / eval-matrix / eval-rerank-matrix 结束后写入 analysis/packNN；auto=递增；off 关闭",
    )
    sp.add_argument(
        "--from-run",
        type=str,
        default=None,
        dest="from_run",
        help="Step3 训练 run 目录名（runs/.../train/step3/<run>/）",
    )
    sp.add_argument(
        "--step5-run",
        type=str,
        default=None,
        dest="step5_run",
        help="Step5 目录名（须为 {step4}_{n}，如 2_1_1）；auto 时必须同时传 --step4-run；显式 slug 时 CSV 仅由本目录名反推 step4",
    )
    sp.add_argument(
        "--step4-run",
        type=str,
        default=None,
        dest="step4_run",
        help="Step4 产物目录名（runs/.../train/step4/<name>/）；step4 可省略→按 {from_run}_n 自动分配；"
        "step5 在 --step5-run auto 时必填；显式 --step5-run 时训练 CSV 仅由 step5 目录名反推 train/step4/{去掉末段}/（不再用本参数）",
    )
    sp.add_argument("--train-csv", type=str, default=None, dest="train_csv")
    sp.add_argument("--model-path", type=str, default=None, dest="model_path")
    sp.add_argument("--epochs", type=int, default=None, dest="epochs")
    sp.add_argument("--num-proc", type=int, default=None, dest="num_proc")
    sp.add_argument("--seed", type=int, default=None, dest="seed")
    sp.add_argument("--ddp-world-size", type=int, default=None, dest="ddp_world_size")
    sp.add_argument(
        "--hardware-preset",
        type=str,
        default=None,
        dest="hardware_preset",
        metavar="STEM",
        help="presets/hardware/<STEM>.yaml；num_proc / DataLoader / ddp / 线程 / CUDA_VISIBLE_DEVICES 等由编排层注入子进程 env",
    )
    sp.add_argument(
        "--omp-num-threads",
        type=int,
        default=None,
        dest="omp_num_threads",
        metavar="N",
        help="runtime：覆盖 hardware preset 的 OMP_NUM_THREADS（优先级高于 shell 中的 OMP_NUM_THREADS）",
    )
    sp.add_argument(
        "--mkl-num-threads",
        type=int,
        default=None,
        dest="mkl_num_threads",
        metavar="N",
        help="runtime：覆盖 hardware preset 的 MKL_NUM_THREADS（优先级高于 shell）",
    )
    sp.add_argument(
        "--tokenizers-parallelism",
        type=str,
        default=None,
        choices=("true", "false"),
        dest="tokenizers_parallelism",
        help="runtime：true/false，覆盖 hardware preset 的 TOKENIZERS_PARALLELISM（优先级高于 shell）",
    )
    sp.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        dest="cuda_visible_devices",
        metavar="LIST",
        help="launcher-only：torchrun 子进程的 CUDA_VISIBLE_DEVICES（优先级高于 shell 与 hardware preset；不计入语义指纹）",
    )
    sp.add_argument("--eta", type=float, default=None, dest="eta")


def _add_checkpoint_kind_arg(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--checkpoint-kind",
        type=str,
        choices=("best_mainline", "last"),
        default="best_mainline",
        dest="checkpoint_kind",
        help="无 --model-path 时：best_mainline=model/best_mainline.pth（默认），last=model/last.pth",
    )


def _add_decode_preset_arg(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--decode-preset",
        type=str,
        default=None,
        dest="decode_preset",
        metavar="NAME",
        help="解码预设：与 presets/decode/default.yaml 浅合并；eval 系可省略若已用 --eval-profile",
    )


def _add_eval_profile_arg(sp: argparse.ArgumentParser, *, required: bool = False) -> None:
    sp.add_argument(
        "--eval-profile",
        type=str,
        default=None,
        dest="eval_profile",
        metavar="NAME",
        required=required,
        help="编排层 presets/eval_profiles/<NAME>.yaml：selector 选 hardware/decode/rerank preset；仅可含 eval_batch_size、num_return_sequences（后者仅 rerank 子命令）",
    )


def _add_rerank_cli(sp: argparse.ArgumentParser) -> None:
    """仅保留高频 override；其余 rerank 语义由 presets/rerank/*.yaml 与 eval_profile 提供。"""
    sp.add_argument(
        "--rerank-preset",
        type=str,
        default=None,
        dest="rerank_preset",
        help="presets/rerank/<name>.yaml；默认可由 eval_profile 指定",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="d4c",
        description=(
            "D4C — 跨域可反事实解释生成。主入口：在仓库根执行 python code/d4c.py <子命令> …\n"
            "配置由 presets/ 下 YAML 与 CLI 共同决定（合并顺序见 docs/PRESETS.md）。"
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND", help="子命令")

    p3 = sub.add_parser(
        "step3",
        help="Step3：域对抗（默认 train + 收尾 eval；d4c 分发到 step3 runner / torchrun）",
        description=(
            "Step3 域对抗。默认：train 后接 eval。\n"
            "须: --task, --preset, --iter。可选 --run-id（默认 auto 分配 1、2、…）。\n"
            "互斥: --eval-only（仅 eval） / --train-only（仅 train，跳过收尾 eval）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "由 d4c 编排 step3 runner。排障：训练 runs/.../logs/train.log；step3 收尾 eval 与 --eval-only 写 runs/.../logs/eval.log；manifest.json 在同 run 目录。"
            "完整规范见 docs/D4C_Scripts_and_Runtime_Guide.md。"
        ),
    )
    _add_common(p3)
    g3 = p3.add_mutually_exclusive_group()
    g3.add_argument(
        "--eval-only",
        action="store_true",
        dest="eval_only",
        help="仅 step3 runner 的 eval（须已有 checkpoint；与 --train-only 互斥）",
    )
    g3.add_argument(
        "--train-only",
        action="store_true",
        dest="train_only",
        help="仅 train，跳过训练后的收尾 eval（与 --eval-only 互斥）",
    )

    p4 = sub.add_parser(
        "step4",
        help="Step4：反事实推理生成（eval 语义侧；d4c → step4 runner / torchrun）",
        description=(
            "Step4 反事实生成（与 eval 同属 eval 语义侧）。须: --task, --preset, --iter, --from-run, --eval-profile。\n"
            "推理全局 batch **仅**来自 eval_profile.eval_batch_size（strict 整除 ddp_world_size）；"
            "不再使用 training preset 的 train_batch_size。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "由 d4c 编排 step4 runner。须先完成 step3；--from-run 为 train/step3 下的 run 目录名。"
            "产物写在 train/step4/<step4-run>/（--step4-run 省略时自动分配 {from_run}_n）。"
            "排障：主日志 runs/.../train/step4/<step4-run>/logs/step4.log（含各 rank 性能行）。详见 README。"
        ),
    )
    _add_common(p4)
    _add_eval_profile_arg(p4, required=True)

    p5 = sub.add_parser(
        "step5",
        help="Step5：主模型 train（d4c 分发到 step5 runner train / torchrun）",
        description=(
            "Step5 主训练。须: --task, --preset（通常为 step5）, --from-run；"
            "--step5-run 可省略/auto，此时**必须**同时传 --step4-run。"
            "训练后 valid：推荐 --eval-profile（与 eval 对齐选 hardware/decode/eval_batch_size）；"
            "step5 非 --train-only 模式要求 eval_batch_size 由 eval_profile 提供；"
            "hardware/decode 可显式 --hardware-preset / --decode-preset。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="由 d4c 编排 step5 runner。预设通常用 --preset step5。权重与指标路径见 manifest 与 README。",
    )
    _add_common(p5)
    _add_eval_profile_arg(p5)
    _add_decode_preset_arg(p5)
    p5.add_argument(
        "--train-only",
        action="store_true",
        dest="train_only",
        help="step5 runner train 跳过训练后收尾评测",
    )

    p_es = sub.add_parser(
        "eval-summary",
        help="Phase 1：扫描 runs/.../vN/eval/*/metrics.json，写入 matrix/<run>/",
        description=(
            "以 metrics.json 为唯一真相源；不写 torchrun。\n"
            "汇总写入 runs/task{T}/vN/matrix/<run>/phase1_summary.* 与 matrix_manifest.json。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_es.add_argument("--task", type=int, required=True, choices=range(1, 9), metavar="N")
    p_es.add_argument("--iter", type=str, required=True, dest="iteration_id", help="迭代 vN")
    p_es.add_argument(
        "--run-id",
        type=str,
        default="auto",
        dest="matrix_run_id",
        help="matrix 目录名；auto=下一个空目录（默认 1、2、…）",
    )
    p_es.add_argument(
        "--latest",
        type=int,
        default=None,
        metavar="N",
        dest="summary_latest_n",
        help="仅纳入 runs/.../eval/*/ 下 metrics.json 修改时间最新的 N 个目录（与全量 phase1_summary 并存）",
    )
    p_es.add_argument(
        "--output-stem",
        type=str,
        default="phase1_summary",
        metavar="NAME",
        dest="summary_output_stem",
        help=(
            "输出主文件名（不含 .csv/.json）；默认 phase1_summary。"
            "若指定了 --latest 且仍用默认 stem，将自动改为 phase1_summary_latest<N>，避免覆盖全量汇总。"
        ),
    )

    pm = sub.add_parser(
        "eval-matrix",
        help="Phase 1：按多个 --decode-presets 顺序各跑一次 eval（各写独立 eval_run）",
        description="与 eval 相同须 task/preset/from-run/step5-run（或 model-path）；依次切换 decode preset。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(pm)
    _add_checkpoint_kind_arg(pm)
    _add_decode_preset_arg(pm)
    _add_eval_profile_arg(pm)
    pm.add_argument(
        "--matrix-run-id",
        type=str,
        default="auto",
        dest="matrix_run_id",
        help="本矩阵汇总写入 runs/.../matrix/<run>；auto=下一个空目录",
    )
    pm.add_argument(
        "--decode-presets",
        nargs="+",
        required=True,
        metavar="STEM",
        dest="decode_presets",
        help="多个 decode 预设 stem，对应 presets/decode/<STEM>.yaml",
    )

    pr_er = sub.add_parser(
        "eval-rerank",
        help="Step5 评测 + 多候选 rule rerank（独立 eval_run + phase2 侧车文件）",
        description="与 eval 相同须 task/preset/from-run/step5-run（或 model-path）；额外 rerank 参数见下文。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(pr_er)
    _add_checkpoint_kind_arg(pr_er)
    _add_decode_preset_arg(pr_er)
    _add_eval_profile_arg(pr_er)
    _add_rerank_cli(pr_er)

    pr_erm = sub.add_parser(
        "eval-rerank-matrix",
        help="多个 decode preset 上依次 eval-rerank，结束后写 phase2_rerank_summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(pr_erm)
    _add_checkpoint_kind_arg(pr_erm)
    _add_decode_preset_arg(pr_erm)
    _add_eval_profile_arg(pr_erm)
    pr_erm.add_argument(
        "--matrix-run-id",
        type=str,
        default="auto",
        dest="matrix_run_id",
        help="本矩阵汇总写入 runs/.../matrix/<run>；auto=下一个空目录",
    )
    pr_erm.add_argument(
        "--decode-presets",
        nargs="+",
        required=True,
        metavar="STEM",
        dest="decode_presets",
        help="多个 decode 预设 stem，对应 presets/decode/<STEM>.yaml",
    )
    _add_rerank_cli(pr_erm)

    p_rs = sub.add_parser(
        "rerank-summary",
        help="扫描 runs/.../vN/rerank/*/metrics.json，写入 matrix/<run>/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_rs.add_argument("--task", type=int, required=True, choices=range(1, 9), metavar="N")
    p_rs.add_argument("--iter", type=str, required=True, dest="iteration_id")
    p_rs.add_argument(
        "--run-id",
        type=str,
        default="auto",
        dest="matrix_run_id",
        help="matrix 目录名；auto=下一个空目录（默认 1、2、…）",
    )

    p_ap = sub.add_parser(
        "analysis-pack",
        help="将指定 eval/rerank/matrix 产物打包到 analysis/packNN/（给 AI 阅读）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ap.add_argument("--task", type=int, required=True, choices=range(1, 9), metavar="N")
    p_ap.add_argument("--iter", type=str, required=True, dest="iteration_id")
    p_ap.add_argument("--eval-run", type=str, default=None, dest="pack_eval_run", help="eval 下目录名")
    p_ap.add_argument("--rerank-run", type=str, default=None, dest="pack_rerank_run", help="rerank 下目录名")
    p_ap.add_argument(
        "--matrix-run",
        type=str,
        default=None,
        dest="pack_matrix_run",
        help="matrix 下目录名（可选，用于复制 phase 汇总）",
    )
    p_ap.add_argument(
        "--analysis-pack",
        type=str,
        default="auto",
        dest="analysis_pack_id",
        help="analysis 打包目录名；auto=下一个空目录",
    )

    p_rb = sub.add_parser(
        "register-baseline",
        help="登记 eval 基线：快照 metrics.json 至 runs/.../baselines/（不修改源 eval 目录）",
        description=(
            "校验 source_eval_dir 与其中 metrics.json；在 runs/task{T}/vN/baselines/<baseline_id>/ "
            "写入 baseline_registration.json 与 metrics_snapshot.json。"
            "可选 --set-default 更新 default_baseline_index.json，供 load_baseline_metrics 默认解析。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_rb.add_argument("--task", type=int, required=True, choices=range(1, 9), metavar="N")
    p_rb.add_argument("--iter", type=str, required=True, dest="iteration_id", help="迭代 vN")
    p_rb.add_argument("--baseline-id", type=str, required=True, dest="baseline_id", metavar="ID")
    p_rb.add_argument(
        "--source-eval-dir",
        type=str,
        required=True,
        dest="source_eval_dir",
        help="源 eval run 目录（相对仓库根或绝对路径），须含 metrics.json",
    )
    p_rb.add_argument("--note", type=str, default=None, dest="baseline_note")
    p_rb.add_argument("--purpose", type=str, default=None, dest="baseline_purpose")
    p_rb.add_argument(
        "--set-default",
        action="store_true",
        dest="baseline_set_default",
        help="将本 baseline_id 设为该 (task, iter) 的默认基线（写 baselines/default_baseline_index.json）",
    )
    p_rb.add_argument(
        "--force",
        action="store_true",
        dest="baseline_force",
        help="基线目录已存在时仍覆盖登记文件与快照",
    )

    pe = sub.add_parser(
        "eval",
        help="Step5 评测（d4c 分发到 step5 runner eval / torchrun）",
        description=(
            "对已有 Step5 权重做 eval。须 --task --preset；"
            "以及 (--model-path) 或 (--from-run 与 --step5-run)。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="由 d4c 编排 step5 runner 做 valid 评测。可提供 --model-path 或 from_run+step5_run。",
    )
    _add_common(pe)
    _add_checkpoint_kind_arg(pe)
    _add_decode_preset_arg(pe)
    _add_eval_profile_arg(pe)

    pp = sub.add_parser(
        "pipeline",
        help="串联 Step3 → Step4 → Step5（三步预设规则见下文）",
        description=(
            "依次执行 Step3、Step4、Step5。\n"
            "· **须** --eval-profile：Step4 已并入 eval 语义，推理 batch 仅来自该 profile 的 eval_batch_size；"
            "Step5 非 --train-only 时亦使用该 profile（与单独 step5 行为一致）。\n"
            "· Step3 / Step4：使用本命令的 --preset（例如 step3）。\n"
            "· Step5：强制使用训练预设 step5（presets/training/step5.yaml），"
            "与 Step3 的 --preset 无关；请在日志中确认 [D4C Mainline] pipeline 提示。\n"
            "· 加 --with-eval 时：Step5 结束后用本命令的 --eval-profile（或 hardware+decode）再跑一次 eval。\n"
            "可选 --run-id / --step5-run 控制各阶段目录名；省略则自动分配下一空目录。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="依次 step3→step4→step5；**pipeline 必须带 --eval-profile**（Step4 合同）；每段写 manifest.json（默认）。",
    )
    _add_common(pp)
    _add_decode_preset_arg(pp)
    _add_eval_profile_arg(pp, required=True)
    pp.add_argument(
        "--with-eval",
        action="store_true",
        dest="with_eval",
        help="Step5 完成后继续 eval；推荐 --eval-profile（或 --hardware-preset + --decode-preset）",
    )

    sub.add_parser(
        "smoke-ddp",
        help="DDP 冒烟（3→4→5 极简步数；产物在 runs/task1/v0/train/...）",
        description="与 scripts/entrypoints/smoke_ddp.sh 等价；须在仓库根执行。",
    )

    return p


def main() -> None:
    bad = _legacy_step5_decode_flags_in_argv(sys.argv[1:])
    if bad:
        _fail_legacy_decode_flags(bad)
    args = build_parser().parse_args()

    if args.command == "register-baseline":
        from d4c_core.baseline_registry import register_baseline

        repo = Path(__file__).resolve().parent.parent
        it = run_naming.normalize_iteration_id(args.iteration_id)
        reg_p, snap_p = register_baseline(
            repo,
            args.task,
            it,
            args.baseline_id,
            args.source_eval_dir,
            note=getattr(args, "baseline_note", None),
            purpose=getattr(args, "baseline_purpose", None),
            set_default=bool(getattr(args, "baseline_set_default", False)),
            force=bool(getattr(args, "baseline_force", False)),
        )
        print(f"[register-baseline] wrote {reg_p}", flush=True)
        print(f"[register-baseline] wrote {snap_p}", flush=True)
        if getattr(args, "baseline_set_default", False):
            from d4c_core.baseline_registry import default_baseline_index_path

            print(
                f"[register-baseline] default index: {default_baseline_index_path(repo, args.task, it)}",
                flush=True,
            )
        return

    if args.command == "analysis-pack":
        from d4c_core.analysis_pack import export_analysis_pack

        repo = Path(__file__).resolve().parent.parent
        it = run_naming.normalize_iteration_id(args.iteration_id)
        ev_dirs = []
        if getattr(args, "pack_eval_run", None):
            ev_dirs.append(
                path_layout.get_eval_run_root(
                    repo, args.task, it, run_naming.parse_run_id(args.pack_eval_run)
                )
            )
        rr_dirs = []
        if getattr(args, "pack_rerank_run", None):
            rr_dirs.append(
                path_layout.get_rerank_run_root(
                    repo, args.task, it, run_naming.parse_run_id(args.pack_rerank_run)
                )
            )
        matrix_d = None
        if getattr(args, "pack_matrix_run", None):
            matrix_d = path_layout.get_matrix_run_root(
                repo, args.task, it, run_naming.parse_run_id(args.pack_matrix_run)
            )
        pack_req = getattr(args, "analysis_pack_id", "auto")
        pr = (
            None
            if pack_req is None or str(pack_req).strip().lower() in ("", "auto")
            else str(pack_req).strip()
        )
        pout = export_analysis_pack(
            repo_root=repo,
            task_id=args.task,
            iteration_id=it,
            pack_id_req=pr,
            eval_run_dirs=ev_dirs,
            rerank_run_dirs=rr_dirs,
            matrix_run_dir=matrix_d,
        )
        print(f"[analysis-pack] wrote {pout}", flush=True)
        return

    if args.command == "rerank-summary":
        from d4c_core.phase2_rerank_summary import generate_phase2_rerank_summary

        repo = Path(__file__).resolve().parent.parent
        it = run_naming.normalize_iteration_id(args.iteration_id)
        root_it = path_layout.get_iteration_root(repo, args.task, it)
        mparent = root_it / "matrix"
        mparent.mkdir(parents=True, exist_ok=True)
        mreq = (
            None
            if str(getattr(args, "matrix_run_id", "auto")).strip().lower() in ("", "auto")
            else str(args.matrix_run_id).strip()
        )
        mid = run_naming.allocate_child_dir(mparent, requested=mreq, kind="run")
        out_d = path_layout.get_matrix_run_root(repo, args.task, it, mid)
        res = generate_phase2_rerank_summary(str(root_it), out_dir=str(out_d))
        if hasattr(res, "to_string"):
            print(res.to_string(index=False), flush=True)
        else:
            import json as _json

            print(_json.dumps(res, ensure_ascii=False, indent=2, default=str), flush=True)
        return

    if args.command == "eval-summary":
        from d4c_core.phase1_eval_summary import generate_phase1_summary

        repo = Path(__file__).resolve().parent.parent
        it = run_naming.normalize_iteration_id(args.iteration_id)
        root_it = path_layout.get_iteration_root(repo, args.task, it)
        mparent = root_it / "matrix"
        mparent.mkdir(parents=True, exist_ok=True)
        mreq = (
            None
            if str(getattr(args, "matrix_run_id", "auto")).strip().lower() in ("", "auto")
            else str(args.matrix_run_id).strip()
        )
        mid = run_naming.allocate_child_dir(mparent, requested=mreq, kind="run")
        out_d = path_layout.get_matrix_run_root(repo, args.task, it, mid)

        latest_n = getattr(args, "summary_latest_n", None)
        stem = getattr(args, "summary_output_stem", None) or "phase1_summary"
        if latest_n is not None and latest_n > 0 and stem == "phase1_summary":
            stem = f"phase1_summary_latest{int(latest_n)}"

        res = generate_phase1_summary(
            str(root_it),
            out_dir=str(out_d),
            only_latest_n=latest_n,
            output_stem=stem,
        )
        if hasattr(res, "to_string"):
            print(res.to_string(index=False), flush=True)
        else:
            import json as _json

            print(_json.dumps(res, ensure_ascii=False, indent=2, default=str), flush=True)
        return

    if args.command == "eval-rerank-matrix":
        import argparse as ap
        import json as _json

        from d4c_core.phase2_rerank_summary import generate_phase2_rerank_summary

        repo = Path(__file__).resolve().parent.parent
        it = run_naming.normalize_iteration_id(getattr(args, "iteration_id", "v1"))
        root_it = path_layout.get_iteration_root(repo, args.task, it)
        mparent = root_it / "matrix"
        mparent.mkdir(parents=True, exist_ok=True)
        mreq = (
            None
            if str(getattr(args, "matrix_run_id", "auto")).strip().lower() in ("", "auto")
            else str(args.matrix_run_id).strip()
        )
        mid = run_naming.allocate_child_dir(mparent, requested=mreq, kind="run")
        out_d = path_layout.get_matrix_run_root(repo, args.task, it, mid)

        stems: list[str] = list(args.decode_presets)
        last_cfg: ResolvedConfig | None = None
        for stem in stems:
            ns = ap.Namespace(
                task=args.task,
                preset=args.preset,
                run_name=getattr(args, "run_name", None),
                iteration_id=getattr(args, "iteration_id", "v1"),
                run_id="auto",
                analysis_pack=getattr(args, "analysis_pack", "auto"),
                from_run=args.from_run,
                step5_run=args.step5_run,
                step4_run=getattr(args, "step4_run", None),
                train_csv=getattr(args, "train_csv", None),
                model_path=getattr(args, "model_path", None),
                eval_profile=getattr(args, "eval_profile", None),
                hardware_preset=getattr(args, "hardware_preset", None),
                epochs=getattr(args, "epochs", None),
                num_proc=getattr(args, "num_proc", None),
                seed=getattr(args, "seed", None),
                ddp_world_size=getattr(args, "ddp_world_size", None),
                eta=getattr(args, "eta", None),
                decode_preset=stem,
                rerank_preset=getattr(args, "rerank_preset", None),
                **_matrix_runtime_ns_fields(args),
            )
            print(f"[eval-rerank-matrix] ---- decode-preset={stem} ----", flush=True)
            try:
                os.environ["D4C_MATRIX_CONTEXT_JSON"] = _json.dumps(
                    {
                        "matrix_session_id": mid,
                        "matrix_cell_id": stem,
                        "invoked_command": "eval-rerank-matrix",
                        "cell_command": "eval-rerank",
                        "resolved_command_kind": "eval-rerank",
                    },
                    ensure_ascii=False,
                )
                os.environ["D4C_MANIFEST_CLI_INVOCATION"] = shlex.join(sys.argv) + f" # rerank_matrix:{stem}"
                cfg = load_resolved_config(ns, "eval-rerank")
                last_cfg = cfg
                print_pre_run_banner("eval-rerank-matrix", cfg)
                validate_resolved_config(cfg)
                run_eval_rerank(cfg)
            finally:
                os.environ.pop("D4C_MANIFEST_CLI_INVOCATION", None)
                os.environ.pop("D4C_MATRIX_CONTEXT_JSON", None)
        if last_cfg:
            print("[eval-rerank-matrix] 生成 phase2_rerank_summary …", flush=True)
            generate_phase2_rerank_summary(str(root_it), out_dir=str(out_d))
            _export_post_matrix_analysis_pack(args, last_cfg, out_d, rerank=True)
        return

    if args.command == "eval-matrix":
        import argparse as ap
        import json as _json

        from d4c_core.phase1_eval_summary import generate_phase1_summary

        repo = Path(__file__).resolve().parent.parent
        it = run_naming.normalize_iteration_id(getattr(args, "iteration_id", "v1"))
        root_it = path_layout.get_iteration_root(repo, args.task, it)
        mparent = root_it / "matrix"
        mparent.mkdir(parents=True, exist_ok=True)
        mreq = (
            None
            if str(getattr(args, "matrix_run_id", "auto")).strip().lower() in ("", "auto")
            else str(args.matrix_run_id).strip()
        )
        mid = run_naming.allocate_child_dir(mparent, requested=mreq, kind="run")
        out_d = path_layout.get_matrix_run_root(repo, args.task, it, mid)

        stems: list[str] = list(args.decode_presets)
        last_cfg: ResolvedConfig | None = None
        for stem in stems:
            ns = ap.Namespace(
                task=args.task,
                preset=args.preset,
                run_name=getattr(args, "run_name", None),
                iteration_id=getattr(args, "iteration_id", "v1"),
                run_id="auto",
                analysis_pack=getattr(args, "analysis_pack", "auto"),
                from_run=args.from_run,
                step5_run=args.step5_run,
                step4_run=getattr(args, "step4_run", None),
                train_csv=getattr(args, "train_csv", None),
                model_path=getattr(args, "model_path", None),
                eval_profile=getattr(args, "eval_profile", None),
                hardware_preset=getattr(args, "hardware_preset", None),
                epochs=getattr(args, "epochs", None),
                num_proc=getattr(args, "num_proc", None),
                seed=getattr(args, "seed", None),
                ddp_world_size=getattr(args, "ddp_world_size", None),
                eta=getattr(args, "eta", None),
                decode_preset=stem,
                **_matrix_runtime_ns_fields(args),
            )
            print(f"[eval-matrix] ---- decode-preset={stem} ----", flush=True)
            try:
                os.environ["D4C_MATRIX_CONTEXT_JSON"] = _json.dumps(
                    {
                        "matrix_session_id": mid,
                        "matrix_cell_id": stem,
                        "invoked_command": "eval-matrix",
                        "cell_command": "eval",
                        "resolved_command_kind": "eval",
                    },
                    ensure_ascii=False,
                )
                os.environ["D4C_MANIFEST_CLI_INVOCATION"] = shlex.join(sys.argv) + f" # matrix:{stem}"
                cfg = load_resolved_config(ns, "eval")
                last_cfg = cfg
                print_pre_run_banner("eval", cfg)
                validate_resolved_config(cfg)
                run_eval(cfg)
            finally:
                os.environ.pop("D4C_MANIFEST_CLI_INVOCATION", None)
                os.environ.pop("D4C_MATRIX_CONTEXT_JSON", None)
        if last_cfg:
            print("[eval-matrix] 生成 phase1_summary …", flush=True)
            generate_phase1_summary(str(root_it), out_dir=str(out_d))
            _export_post_matrix_analysis_pack(args, last_cfg, out_d, rerank=False)
        return

    if args.command == "pipeline":
        print_pipeline_opening(step3_preset=args.preset)
        try:
            os.environ["D4C_MANIFEST_CLI_INVOCATION"] = shlex.join(sys.argv)
            cfg_eval_after_pipeline = run_pipeline(args)
        finally:
            os.environ.pop("D4C_MANIFEST_CLI_INVOCATION", None)
        if cfg_eval_after_pipeline is not None:
            _maybe_export_analysis_pack(args, cfg_eval_after_pipeline)
        return

    if args.command == "smoke-ddp":
        print_smoke_ddp_preamble()
        repo = Path(__file__).resolve().parent.parent
        run_smoke_ddp(repo)
        return

    try:
        os.environ["D4C_MANIFEST_CLI_INVOCATION"] = shlex.join(sys.argv)
        cfg = load_resolved_config(args, args.command)
        print_pre_run_banner(args.command, cfg)
        validate_resolved_config(cfg)

        if args.command == "step3":
            run_step3(cfg)
        elif args.command == "step4":
            run_step4(cfg)
        elif args.command == "step5":
            run_step5(cfg)
        elif args.command == "eval":
            run_eval(cfg)
            _maybe_export_analysis_pack(args, cfg)
        elif args.command == "eval-rerank":
            run_eval_rerank(cfg)
            _maybe_export_analysis_pack(args, cfg)
        else:
            print(f"未知子命令: {args.command}", file=sys.stderr)
            raise SystemExit(2)
    finally:
        os.environ.pop("D4C_MANIFEST_CLI_INVOCATION", None)


if __name__ == "__main__":
    main()
