#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# DDP 冒烟：python code/d4c.py smoke-ddp（逻辑在 d4c_core.runners.run_smoke_ddp）
# 亦见 docs/D4C_Scripts_and_Runtime_Guide.md
# -----------------------------------------------------------------------------
# =============================================================================
# DDP smoke test — NOT a quality / metrics benchmark.
#
# Runs the official torchrun + DDP entrypoints with tiny settings to verify
# the stack does not crash. nproc_per_node=1 still uses init_process_group and
# DDP; it is NOT a separate non-distributed code path.
#
# Prerequisites: full offline data layout (Merged_data/1, data/*, pretrained_models, …).
# Outputs: checkpoints/1/smoke_ddp/<SMOKE_TAG>/ and log/1/smoke_ddp/ — safe to rm after test.
#
# Usage (from repo root):
#   bash sh/smoke_test_ddp.sh
#   CUDA_VISIBLE_DEVICES=0 bash sh/smoke_test_ddp.sh
# =============================================================================
set -euo pipefail

SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
_MERGED="$D4C_ROOT/Merged_data/1/aug_train.csv"
if [[ ! -f "$_MERGED" ]]; then
    echo "错误: 缺少 smoke 前置数据: $_MERGED（需先完成 Step1+2 与 combine）" >&2
    exit 1
fi

cd "$D4C_ROOT"
exec python code/d4c.py smoke-ddp
