#!/usr/bin/env bash
# Phase1：多 decode 各跑一次 eval，并在 matrix/<run>/ 写 phase1_summary + matrix_manifest.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ITER="${ITER:-v1}"
STEP3_RUN="${STEP3_RUN:?}"
STEP5_RUN="${STEP5_RUN:?}"
# 空格分隔的 decode preset stem 列表
PRESETS="${PRESETS:-decode_greedy_default decode_balanced_v2}"
EVAL_PROFILE="${EVAL_PROFILE:?export EVAL_PROFILE=如 eval_balanced_2gpu}"
python code/d4c.py eval-matrix --task 4 --preset step5 --iter "$ITER" \
  --from-run "$STEP3_RUN" --step5-run "$STEP5_RUN" \
  --eval-profile "$EVAL_PROFILE" \
  --decode-presets $PRESETS "$@"
