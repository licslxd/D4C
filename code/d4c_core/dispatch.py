"""torchrun 目标脚本名（仅子进程使用）与 MAINLINE 分发叙事。"""
from __future__ import annotations

import os

# 三者均为 code/ 下的历史兼容薄壳；实现位于 executors/*_engine.py
TORCHRUN_STEP3_SCRIPT = "AdvTrain.py"
TORCHRUN_STEP4_SCRIPT = "generate_counterfactual.py"
TORCHRUN_STEP5_SCRIPT = "run-d4c.py"


def print_dispatch_routing(command: str) -> None:
    """用户可见主叙事：阶段 runner，不强调薄壳文件名。"""
    if command == "step3":
        print("[Dispatch] step3 -> step3 runner (torchrun)", flush=True)
    elif command == "step4":
        print("[Dispatch] step4 -> step4 runner (torchrun)", flush=True)
    elif command == "step5":
        print("[Dispatch] step5 -> step5 runner train (torchrun)", flush=True)
    elif command == "eval":
        print("[Dispatch] eval -> step5 runner eval (torchrun)", flush=True)
    elif command == "pipeline":
        print(
            "[Dispatch] pipeline -> step3 runner -> step4 runner -> step5 runner (torchrun x3)",
            flush=True,
        )
    elif command == "smoke-ddp":
        print(
            "[Dispatch] smoke-ddp -> step3/4/5 runners（含额外 eval 段，torchrun x4）",
            flush=True,
        )


def print_dispatch_script_detail(command: str) -> None:
    """排障用：打印实际 torchrun 加载的薄壳文件名。默认关闭，设 D4C_DISPATCH_DETAIL=1 开启。"""
    flag = (os.environ.get("D4C_DISPATCH_DETAIL") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    if command == "step3":
        print(f"[Dispatch][detail] torchrun script={TORCHRUN_STEP3_SCRIPT}", flush=True)
    elif command == "step4":
        print(f"[Dispatch][detail] torchrun script={TORCHRUN_STEP4_SCRIPT}", flush=True)
    elif command in ("step5", "eval"):
        print(f"[Dispatch][detail] torchrun script={TORCHRUN_STEP5_SCRIPT}", flush=True)
    elif command == "pipeline":
        print(
            "[Dispatch][detail] torchrun scripts="
            f"{TORCHRUN_STEP3_SCRIPT}, {TORCHRUN_STEP4_SCRIPT}, {TORCHRUN_STEP5_SCRIPT}",
            flush=True,
        )
    elif command == "smoke-ddp":
        print(
            "[Dispatch][detail] torchrun scripts="
            f"{TORCHRUN_STEP3_SCRIPT} (x2 eval), {TORCHRUN_STEP4_SCRIPT}, {TORCHRUN_STEP5_SCRIPT}",
            flush=True,
        )


def internal_executor_label(step: int) -> str:
    if step == 3:
        return TORCHRUN_STEP3_SCRIPT
    if step == 4:
        return TORCHRUN_STEP4_SCRIPT
    if step == 5:
        return TORCHRUN_STEP5_SCRIPT
    raise ValueError(f"unknown step: {step}")
