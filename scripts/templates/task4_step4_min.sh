#!/usr/bin/env bash
# 最小 Step4；须与 Step3 的 from-run 一致
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ITER="${ITER:-v1}"
STEP3_RUN="${STEP3_RUN:?设置 STEP3_RUN=train/step3 下目录名，如 1}"
# Step4 须显式 eval_profile stem
STEP4_EVAL_PROFILE="${STEP4_EVAL_PROFILE:?export STEP4_EVAL_PROFILE=如 eval_fast_single_gpu}"
python code/d4c.py step4 --task 4 --preset step3 --iter "$ITER" --from-run "$STEP3_RUN" \
  --eval-profile "$STEP4_EVAL_PROFILE" "$@"
