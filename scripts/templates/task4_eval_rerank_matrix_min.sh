#!/usr/bin/env bash
# Phase2：多 decode × rerank（默认 rule_v3 + rerank_v3_default）→ phase2_rerank_summary
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ITER="${ITER:-v1}"
STEP3_RUN="${STEP3_RUN:?}"
STEP5_RUN="${STEP5_RUN:?}"
PRESETS="${PRESETS:-decode_greedy_default decode_balanced_v2}"
EVAL_PROFILE="${EVAL_PROFILE:?export EVAL_PROFILE=如 eval_rerank_quality}"
RERANK_PRESET="${RERANK_PRESET:-rerank_v3_default}"
python code/d4c.py eval-rerank-matrix --task 4 --preset step5 --iter "$ITER" \
  --from-run "$STEP3_RUN" --step5-run "$STEP5_RUN" \
  --eval-profile "$EVAL_PROFILE" \
  --decode-presets $PRESETS --rerank-preset "$RERANK_PRESET" "$@"
