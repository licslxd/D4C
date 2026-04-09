#!/usr/bin/env bash
# 最小 Step3（task4）；仓库根执行：bash scripts/templates/task4_step3_min.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ITER="${ITER:-v1}"
RUN_ID="${RUN_ID:-auto}"
python code/d4c.py step3 --task 4 --preset step3 --iter "$ITER" --run-id "$RUN_ID" "$@"
