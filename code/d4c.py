#!/usr/bin/env python3
"""
MAINLINE ENTRY — D4C 唯一推荐的用户级 Python 入口。

在项目根目录执行::

  python code/d4c.py step3 --task 1 --preset step3 [--run-name NAME]
  python code/d4c.py step4 --task 1 --preset step3 --from-run step3_opt_YYYYMMDD_HHMM
  python code/d4c.py step5 --task 1 --preset step5 --from-run ... --step5-run ...
  python code/d4c.py eval --task 1 --preset step5 --from-run ... --step5-run ...
  python code/d4c.py pipeline --task 1 --preset step3

各阶段由 ``d4c_core.runners`` 经 torchrun 分发到 **step3 / step4 / step5 runner**（实现位于
``code/executors/``）；shell 与日常实验请只使用本入口。排障时可设环境变量 ``D4C_DISPATCH_DETAIL=1``
查看 torchrun 实际加载的兼容薄壳文件名。

**解码（decode）**：主线请在子命令上使用 ``--decode-preset <name>``（见 ``presets/decode/``）。
勿将 ``--decode-strategy`` / ``--generate-temperature`` 等 **step5 runner 内部参数** 直接接在本入口后
（误用时会提示迁移到 ``--decode-preset``）。
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from d4c_core.config_loader import load_resolved_config
from d4c_core.logging_meta import (
    print_pipeline_opening,
    print_pre_run_banner,
    print_smoke_ddp_preamble,
)
from d4c_core.runners import run_eval, run_pipeline, run_smoke_ddp, run_step3, run_step4, run_step5
from d4c_core.validation import validate_resolved_config

_EPILOG = """
身份说明
  · 本命令 (code/d4c.py)     — MAINLINE ENTRY，唯一推荐的 Python 用户入口。
  · sh/*.sh、scripts/train_ddp.sh — 编排层（应调用本文件；由 d4c 再 torchrun 到阶段 runner）。
  · code/ 下历史薄壳文件名 — 仅供 torchrun 加载；日常不必关心，排障见 D4C_DISPATCH_DETAIL。
  · code/legacy/            — 历史脚本，不参与新主线。

最小示例（仓库根目录）
  python code/d4c.py step3 --task 1 --preset step3
  python code/d4c.py step4 --task 2 --preset step3 --from-run step3_opt_20260329_1200
  python code/d4c.py step5 --task 2 --preset step5 --from-run step3_opt_20260329_1200 --step5-run step5_opt_20260329_1300
  python code/d4c.py pipeline --task 1 --preset step3

pipeline 预设规则
  · Step3 与 Step4 使用命令行 --preset（同一训练预设名，如 step3）。
  · Step5 在流水线内强制使用预设名 step5（加载 presets/training/step5.yaml），与 Step3 的 --preset 分离。

运行清单（manifest）
  · 默认在 torchrun 前写入 <log_dir>/d4c_run_manifest.json（复现/排障；文件名固定）。
  · 关闭 JSON：export D4C_WRITE_RUN_MANIFEST=0
  · 命令行会写入 manifest 的 cli_invocation 字段（供复现）；详见 README「输出产物」。

解码（重要）
  · 顶层仅支持 --decode-preset <stem>（合并 presets/decode/default.yaml + presets/decode/<stem>.yaml）。
  · 勿在此处使用 --decode-strategy / --generate-temperature 等；那些由 d4c 在 torchrun 子进程内传给 step5 runner。
"""


# 仅属于 step5 runner（run-d4c 薄壳）的 decode 相关参数；若出现在本入口 argv 中多为历史误用。
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
    preset_examples = "default, greedy, nucleus_t08_p09, nucleus_t10_p095"
    msg = f"""错误: 在 code/d4c.py（MAINLINE 顶层入口）中使用了下游 step5 runner 专用参数: {joined}

这些参数由 torchrun 加载 run-d4c 薄壳时在子进程内解析，不应直接写在「python code/d4c.py …」之后。

请改用顶层解码预设:
  python code/d4c.py eval --task … --preset … --from-run … --step5-run … --decode-preset <stem>

常用 <stem> 对应 presets/decode/<stem>.yaml，例如: {preset_examples}
说明见 docs/PRESETS.md §1.4。

高级: 若确需在子进程追加其它 run-d4c 参数，可用环境变量 D4C_RUN_D4C_EXTRA（勿用于替代 --decode-preset 主路径）。
"""
    print(msg.strip(), file=sys.stderr)
    raise SystemExit(2)


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--task", type=int, required=True, choices=range(1, 9), metavar="N")
    sp.add_argument(
        "--preset",
        type=str,
        required=True,
        help="训练预设键 → presets/training/<name>.yaml（与 task/runtime/decode 合并规则见 docs/PRESETS.md）。pipeline 内 Step5 强制 step5。",
    )
    sp.add_argument("--run-name", type=str, default=None, dest="run_name")
    sp.add_argument("--from-run", type=str, default=None, dest="from_run")
    sp.add_argument("--step5-run", type=str, default=None, dest="step5_run")
    sp.add_argument("--train-csv", type=str, default=None, dest="train_csv")
    sp.add_argument("--model-path", type=str, default=None, dest="model_path")
    sp.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    sp.add_argument("--epochs", type=int, default=None, dest="epochs")
    sp.add_argument("--num-proc", type=int, default=None, dest="num_proc")
    sp.add_argument("--seed", type=int, default=None, dest="seed")
    sp.add_argument("--ddp-world-size", type=int, default=None, dest="ddp_world_size")
    sp.add_argument("--eta", type=float, default=None, dest="eta")
    sp.add_argument(
        "--decode-preset",
        type=str,
        default="default",
        dest="decode_preset",
        metavar="NAME",
        help="解码预设：presets/decode/default.yaml 为底，再与 presets/decode/<NAME>.yaml 浅层合并；"
        "传给 step5/eval 的 decode_strategy / temperature / top_p 等。",
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
            "Step3 域对抗。默认：train 后接 eval（与 sh/run_step3_optimized 默认一致）。\n"
            "须: --task, --preset。可选 --run-name（默认自动生成 step3_opt_*）。\n"
            "互斥: --eval-only（仅 eval） / --train-only（仅 train，跳过收尾 eval）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "由 d4c 编排 step3 runner。排障：主日志 log_dir/train.log；解析结果见同目录 d4c_run_manifest.json；"
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
        help="Step4：反事实生成（d4c 分发到 step4 runner / torchrun）",
        description=(
            "Step4 反事实生成。须: --task, --preset, --from-run（与 Step3 产生的子目录名一致）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="由 d4c 编排 step4 runner。须先完成 step3；--from-run 与 step3 的 run 目录名一致。详见 README / 主指南。",
    )
    _add_common(p4)

    p5 = sub.add_parser(
        "step5",
        help="Step5：主模型 train（d4c 分发到 step5 runner train / torchrun）",
        description=(
            "Step5 主训练。须: --task, --preset（通常为 step5）, --from-run, --step5-run。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="由 d4c 编排 step5 runner。预设通常用 --preset step5。权重与指标路径见 manifest 与 README。",
    )
    _add_common(p5)
    p5.add_argument(
        "--train-only",
        action="store_true",
        dest="train_only",
        help="step5 runner train 跳过训练后收尾评测（与 sh/run_step5_optimized 一致）",
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

    pp = sub.add_parser(
        "pipeline",
        help="串联 Step3 → Step4 → Step5（三步预设规则见下文）",
        description=(
            "依次执行 Step3、Step4、Step5。\n"
            "· Step3 / Step4：使用本命令的 --preset（例如 step3）。\n"
            "· Step5：强制使用训练预设 step5（presets/training/step5.yaml），"
            "与 Step3 的 --preset 无关；请在日志中确认 [D4C Mainline] pipeline 提示。\n"
            "可选 --run-name / --step5-run 控制目录名；默认自动生成时间戳名。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="依次 step3→step4→step5；每段各自 log_dir 下会写 d4c_run_manifest.json（默认）。",
    )
    _add_common(pp)

    sub.add_parser(
        "smoke-ddp",
        help="DDP 冒烟（3→4→5 极简步数；产物在 checkpoints/1/smoke_ddp/）",
        description="与 sh/smoke_test_ddp.sh 等价逻辑；须在仓库根执行。",
    )

    return p


def main() -> None:
    bad = _legacy_step5_decode_flags_in_argv(sys.argv[1:])
    if bad:
        _fail_legacy_decode_flags(bad)
    args = build_parser().parse_args()

    if args.command == "pipeline":
        print_pipeline_opening(step3_preset=args.preset)
        try:
            os.environ["D4C_MANIFEST_CLI_INVOCATION"] = shlex.join(sys.argv)
            run_pipeline(args)
        finally:
            os.environ.pop("D4C_MANIFEST_CLI_INVOCATION", None)
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
        else:
            print(f"未知子命令: {args.command}", file=sys.stderr)
            raise SystemExit(2)
    finally:
        os.environ.pop("D4C_MANIFEST_CLI_INVOCATION", None)


if __name__ == "__main__":
    main()
