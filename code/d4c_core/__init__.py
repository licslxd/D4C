"""MAINLINE 内核（供 ``d4c.py`` 使用）：显式预设链、路径解析、子进程编排。"""
from d4c_core.config_loader import ResolvedConfig, load_resolved_config

__all__ = ["ResolvedConfig", "load_resolved_config"]
