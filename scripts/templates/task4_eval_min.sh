#!/usr/bin/env bash
# 最小 Eval；推荐 --eval-profile（编排 hardware+decode+eval_batch_size）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ITER="${ITER:-v1}"
STEP3_RUN="${STEP3_RUN:?}"
STEP5_RUN="${STEP5_RUN:?如 2_1_1}"
EVAL_PROFILE="${EVAL_PROFILE:?export EVAL_PROFILE=如 eval_balanced_2gpu}"
python code/d4c.py eval --task 4 --preset step5 --iter "$ITER" \
  --from-run "$STEP3_RUN" --step5-run "$STEP5_RUN" --eval-profile "$EVAL_PROFILE" "$@"
