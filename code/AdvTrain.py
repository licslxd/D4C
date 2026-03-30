# -*- coding: utf-8 -*-
"""
历史兼容薄壳 — torchrun 仍可能按此文件名加载；认知入口为 **step3 runner**。

- 核心实现：``executors/adv_train_core.py``
- 入口与 argparse：``executors/step3_entry.py``
- 用户主入口：``python code/d4c.py step3 …``（仓库根）

``from AdvTrain import *`` 仍导出 ``executors.adv_train_core`` 中符号，供历史 ``import *`` 兼容。
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        from executors.step3_entry import print_step3_root_help

        print_step3_root_help()
        raise SystemExit(0)

from executors.adv_train_core import *  # noqa: F401,F403

if __name__ == "__main__":
    from executors.step3_entry import run_step3_cli

    run_step3_cli()
