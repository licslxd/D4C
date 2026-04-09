"""Step4 torchrun 入口：argparse 与主循环（逻辑在 step4_engine）。"""
from __future__ import annotations

import argparse
import os
import sys

from executors import bootstrap
from executors.startup_config_check import print_startup_config_check

_STEP4_RUNNER = "step4 runner（torchrun 内部入口）"


def print_step4_root_help() -> None:
    p = argparse.ArgumentParser(
        prog="step4-runner",
        description=(
            "Step4 反事实生成 — torchrun 内部入口（executors/step4_entry.py）。"
            "请优先: python code/d4c.py step4 …（仓库根）"
        ),
        epilog="完整参数与运行须 torchrun；日常请使用 code/d4c.py step4。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task", type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8], metavar="N", help="仅跑指定任务 1-8"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="全局 eval 推理 batch（= eval_profile.eval_batch_size；须能被 WORLD_SIZE 整除；由 d4c.py 传入）",
    )
    p.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数")
    p.add_argument("--log_file", type=str, default=None, help="PerfMonitor 结构化日志路径")
    p.print_help()


def run_step4_cli() -> None:
    bootstrap.reject_legacy_gpus_argv(
        sys.argv,
        executor_label=_STEP4_RUNNER,
        torchrun_hint=(
            "推荐: python code/d4c.py step4 --task N --preset … --from-run …\n"
            "须自行 torchrun 时见 docs/D4C_Scripts_and_Runtime_Guide.md 附录。\n"
        ),
    )
    epilog = "本入口仅由 d4c.py / sh 以 torchrun 调用；日常不要手工直接运行。"
    parser = argparse.ArgumentParser(
        description=(
            "Step4 反事实生成 — torchrun 内部入口。"
            "请优先: python code/d4c.py step4 …（仓库根）"
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8], metavar="N", help="仅跑指定任务 1-8"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="全局 eval 推理 batch（= eval_profile.eval_batch_size；须能被 WORLD_SIZE 整除；由 d4c.py 传入）",
    )
    parser.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="PerfMonitor 结构化日志路径；不传则沿用当前目录 save.log",
    )
    args = parser.parse_args()
    print_startup_config_check(stage="step4", command="run")

    from executors.step4_engine import (  # noqa: E402 — 重型依赖延后
        _run_one_task,
        _setup_distributed,
        _teardown_distributed,
        get_num_proc,
    )
    import torch

    nproc = args.num_proc if args.num_proc is not None else get_num_proc()
    rank, world_size, local_rank = _setup_distributed()
    if rank == 0:
        print(
            "[step4 runner] — 用户入口: python code/d4c.py step4 …",
            flush=True,
        )

    task_range = [args.task] if args.task else range(1, 9)
    seed = 3407
    torch.manual_seed(seed)

    try:
        for task_idx in task_range:
            if args.batch_size is None:
                raise RuntimeError(
                    "step4 runner 未收到 --batch-size（全局 eval_batch_size）。\n"
                    "请使用: python code/d4c.py step4 … --eval-profile <stem>\n"
                    "由父进程解析 eval_profile 后传入 global_eval_batch_size；禁止回退 train_batch_size。"
                )
            batch_size = int(args.batch_size)
            lf = args.log_file if args.log_file is not None else "save.log"
            lf = os.path.abspath(os.path.expanduser(lf))
            _run_one_task(
                task_idx, batch_size, nproc, rank, world_size, local_rank, log_file=lf
            )
    finally:
        _teardown_distributed()


if __name__ == "__main__":
    run_step4_cli()
