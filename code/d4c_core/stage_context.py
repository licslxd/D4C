"""阶段上下文：封装 ResolvedConfig + 命令名，供 runner 与日志共用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from d4c_core.config_loader import ResolvedConfig, load_resolved_config
from d4c_core.logging_meta import print_pre_run_banner


@dataclass(frozen=True)
class StageContext:
    """单阶段运行上下文（非 mega-runner）：task / preset / 路径已在 ResolvedConfig 中解析。"""

    command: str
    cfg: ResolvedConfig

    @staticmethod
    def from_args(args: Any, command: str) -> "StageContext":
        cfg = load_resolved_config(args, command)
        return StageContext(command=command, cfg=cfg)

    def print_header(self) -> None:
        print_pre_run_banner(self.command, self.cfg)
