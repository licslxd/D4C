"""跨 Step 共享：torchrun 薄壳名（单一来源 d4c_core.dispatch）。"""

from d4c_core.dispatch import (
    TORCHRUN_STEP3_SCRIPT,
    TORCHRUN_STEP4_SCRIPT,
    TORCHRUN_STEP5_SCRIPT,
)

__all__ = [
    "TORCHRUN_STEP3_SCRIPT",
    "TORCHRUN_STEP4_SCRIPT",
    "TORCHRUN_STEP5_SCRIPT",
]
