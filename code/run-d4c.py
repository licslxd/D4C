"""
历史兼容薄壳 — torchrun 仍可能按此文件名加载；认知入口为 **step5 runner**。

- 核心实现：``executors/step5_engine.py``
- 入口：``executors/step5_entry.py``
- 用户主入口：``python code/d4c.py step5|eval|pipeline …``（仓库根）
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        from executors.step5_entry import print_step5_root_help

        print_step5_root_help()
        raise SystemExit(0)

    from executors.step5_entry import run_step5_cli

    run_step5_cli()
