"""Step5 INTERNAL EXECUTOR：argparse 与 _run_ddp 调用（逻辑在 step5_engine）。"""
from __future__ import annotations

import argparse
import os
import sys

from executors import bootstrap
from executors import ddp_utils

_STEP5_RUNNER = "step5 runner（torchrun 内部入口）"


def _add_common_run_d4c_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件；默认 logs/task{idx}_时间戳.log；D4C_LOG_DIR 指定目录",
    )
    p.add_argument("--auxiliary", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数")
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--nhid", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--generate-temperature", type=float, default=0.8)
    p.add_argument("--generate-top-p", type=float, default=0.9)
    p.add_argument("--max-explanation-length", type=int, default=25)
    p.add_argument(
        "--decode-strategy",
        type=str,
        choices=["greedy", "nucleus"],
        default="greedy",
        help="torchrun 子进程参数；日常请用: python code/d4c.py … --decode-preset <stem>。greedy 可复现；nucleus 可配合 --decode-seed",
    )
    p.add_argument("--decode-seed", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None, help="覆盖 FinalTrainingConfig.eval_batch_size 解析链")
    p.add_argument(
        "--eval-single-process-safe",
        action="store_true",
        help="多卡时仅 rank0 顺序跑全量评测（避免 DDP 分片聚合差异；与 DDP 指标对照用）",
    )
    p.add_argument(
        "--sanity-compare-ddp-single",
        action="store_true",
        help="rank0 在 DDP 评测后再跑一遍单进程顺序，打印 MAE/RMSE/BLEU4 差值",
    )


def _add_train_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率；不传则由 build_resolved_training_config 解析",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--coef", type=float, default=None)
    p.add_argument(
        "--eta",
        type=float,
        default=None,
        help="不传则 resolve 默认 1e-3",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="同 --batch-size",
    )
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--per-device-batch-size", type=int, default=None)
    p.add_argument(
        "--train-only",
        action="store_true",
        help="训练结束后跳过 valid 收尾评测（训练中仍按 epoch 做 valid）",
    )
    p.add_argument("--min-epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--early-stop-patience-full", type=int, default=None)
    p.add_argument("--early-stop-patience-loss", type=int, default=None)
    p.add_argument("--checkpoint-metric", type=str, choices=["loss", "bleu4"], default="bleu4")
    p.add_argument("--bleu4-max-samples", type=int, default=None)
    p.add_argument("--quick-eval-max-samples", type=int, default=None)
    p.add_argument("--full-eval-every", type=int, default=None)
    p.add_argument("--scheduler-initial-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=None)


def print_step5_root_help() -> None:
    p = argparse.ArgumentParser(
        prog="step5-runner",
        description=(
            "Step5 主模型 train / eval / test / generate_samples — torchrun 内部入口（须 NCCL）。"
            "请优先: python code/d4c.py step5|eval|pipeline …"
        ),
        epilog="子命令完整参数: 在 code/ 薄壳上执行 train --help / eval --help（将加载 PyTorch 等依赖）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="训练")
    sub.add_parser("eval", help="valid 评测")
    sub.add_parser("test", help="test.csv 评测")
    sub.add_parser("generate_samples", help="导出小批生成样例")
    p.print_help()


def run_step5_cli() -> None:
    bootstrap.reject_legacy_gpus_argv(
        sys.argv,
        executor_label=_STEP5_RUNNER,
        torchrun_hint=(
            "推荐: python code/d4c.py step5|eval …\n"
            "须自行 torchrun 时见 docs/D4C_Scripts_and_Runtime_Guide.md 附录。\n"
        ),
    )
    epilog = (
        "用户日常（仓库根）:\n"
        "  python code/d4c.py step5 …   python code/d4c.py eval …   python code/d4c.py pipeline …\n"
        "本入口仅在被 d4c.py / sh 以 torchrun 调用时使用。"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Step5 主模型 train / eval / test / generate_samples — torchrun 内部入口。"
            "请优先: python code/d4c.py …"
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    _add_common_run_d4c_args(common_parent)

    train_p = sub.add_parser(
        "train",
        parents=[common_parent],
        help="训练（内部由 d4c.py step5 / sh 调用）",
    )
    _add_train_cli_args(train_p)

    sub.add_parser(
        "eval",
        parents=[common_parent],
        help="valid 评测（内部由 d4c.py eval / sh 调用）",
    )
    sub.add_parser(
        "test",
        parents=[common_parent],
        help="test.csv 评测（高级场景；日常优先 d4c.py）",
    )
    gen_p = sub.add_parser(
        "generate_samples",
        parents=[common_parent],
        help="导出小批生成样例（高级场景）",
    )
    gen_p.add_argument("--generate-max-samples", type=int, default=32)

    args = parser.parse_args()
    gb = getattr(args, "global_batch_size", None)
    if gb is not None:
        if getattr(args, "batch_size", None) is not None and args.batch_size != gb:
            parser.error(
                f"--batch-size ({args.batch_size}) 与 --global-batch-size ({gb}) 冲突，请只指定其一。"
            )
        args.batch_size = gb

    ddp_utils.exit_if_not_torchrun(
        executor_label=_STEP5_RUNNER,
        examples=(
            "推荐: python code/d4c.py step5|eval …\n"
            "附录: 高级 torchrun 排障见 docs/D4C_Scripts_and_Runtime_Guide.md。\n"
        ),
    )
    if os.environ.get("RANK", "0") == "0":
        print(
            f"[step5 runner] {args.command} — 用户入口: python code/d4c.py step5|eval|pipeline …",
            flush=True,
        )

    from executors.step5_engine import _run_ddp  # noqa: E402

    _run_ddp(args)
