#!/usr/bin/env bash
# 最小 Step5；--step5-run auto 时必须同时 export STEP4_RUN；须 eval_profile（与 config_loader 合同一致）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ITER="${ITER:-v1}"
STEP3_RUN="${STEP3_RUN:?例如与 step4 上游 Step3 目录名}"
STEP4_RUN="${STEP4_RUN:?例如 2_1；与 auto 分配 step5 时必填}"
STEP5_RUN="${STEP5_RUN:-auto}"
EVAL_PROFILE="${EVAL_PROFILE:?例如 eval_fast_single_gpu}"
python code/d4c.py step5 --task 4 --preset step5 --iter "$ITER" \
  --from-run "$STEP3_RUN" --step4-run "$STEP4_RUN" --step5-run "$STEP5_RUN" \
  --eval-profile "$EVAL_PROFILE" "$@"
