"""
历史兼容薄壳 — torchrun 仍可能按此文件名加载；认知入口为 **step4 runner**。

- 核心实现：``executors/step4_engine.py``
- 入口：``executors/step4_entry.py``
- 用户主入口：``python code/d4c.py step4 …``（仓库根）
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        from executors.step4_entry import print_step4_root_help

        print_step4_root_help()
        raise SystemExit(0)

    from executors.step4_entry import run_step4_cli

    run_step4_cli()
