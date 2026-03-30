"""兼容层：torchrun 薄壳名单一来源为 d4c_core.dispatch。"""
from d4c_core.dispatch import (  # noqa: F401
    TORCHRUN_STEP3_SCRIPT,
    TORCHRUN_STEP4_SCRIPT,
    TORCHRUN_STEP5_SCRIPT,
)
