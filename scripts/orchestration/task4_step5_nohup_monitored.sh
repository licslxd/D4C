#!/usr/bin/env bash
# Task4 Step5：nohup 场景编排（环境快照 / sidecar）；业务语义仅来自 d4c CLI。
#
# 必填环境变量：
#   STEP4_RUN   例如 1_3（与 --step5-run auto 配套）
#   D4C_EVAL_PROFILE  eval_profiles/*.yaml 的 stem（编排 hardware+decode+eval_batch_size）
#
# 可选：D4C_NOHUP_LOG、D4C_FROM_RUN、D4C_TASK、D4C_ITER、CONDA_SH、CONDA_ENV
#
# 用法示例：
#   export STEP4_RUN=1_3
#   export D4C_EVAL_PROFILE=eval_balanced_2gpu
#   nohup bash scripts/orchestration/task4_step5_nohup_monitored.sh &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG="${D4C_NOHUP_LOG:-$ROOT/runs/task4/v1/nohup_logs/task4_step5_alignment_first.nohup.log}"
mkdir -p "$(dirname "$LOG")"

_LOG_DIR="$(dirname "$LOG")"
_LOG_STEM="$(basename "$LOG")"
SIDE="$_LOG_DIR/${_LOG_STEM%.log}"
EXIT_FILE="${SIDE}.exitcode"
FINISHED_AT="${SIDE}.finished_at"
PID_FILE="${SIDE}.pid"
META_ENV="${SIDE}.meta.env"

FROM_RUN="${D4C_FROM_RUN:-1}"
TASK="${D4C_TASK:-4}"
ITER="${D4C_ITER:-v1}"
EVAL_PROFILE="${D4C_EVAL_PROFILE:?须 export D4C_EVAL_PROFILE=<eval_profile_stem>}"

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

cleanup() {
  local x=$?
  printf '%s\n' "$x" >"$EXIT_FILE" || true
  date -Is >"$FINISHED_AT" || true
  if [[ "$x" -ne 0 ]]; then
    echo "[$(date -Is)] === ABNORMAL_EXIT exit=$x ==="
    echo "[$(date -Is)] 建议: tail -80 \"$LOG\""
  else
    echo "[$(date -Is)] === OK finished ==="
  fi
}
trap cleanup EXIT

exec >>"$LOG" 2>&1

: "${STEP4_RUN:?请事先 export STEP4_RUN=你的_step4_目录名 例如 1_3}"

echo "================================================================"
echo "[$(date -Is)] task4_step5_nohup_monitored.sh start"
echo "PID=$$ PPID=$PPID"
printf '%s\n' "$$" >"$PID_FILE" || true

{
  echo "ROOT=$ROOT"
  echo "STEP4_RUN=$STEP4_RUN"
  echo "FROM_RUN=$FROM_RUN TASK=$TASK ITER=$ITER"
  echo "D4C_EVAL_PROFILE=$EVAL_PROFILE"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER"
} >"$META_ENV"

echo "--- uname -a ---"
uname -a || true
echo "--- nvidia-smi -L ---"
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"

_CONDA_SH="${CONDA_SH:-}"
if [[ -z "$_CONDA_SH" ]]; then
  for c in "$HOME/miniconda3/etc/profile.d/conda.sh" "/public/home/zhangliml/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$c" ]]; then
      _CONDA_SH="$c"
      break
    fi
  done
fi
if [[ -z "$_CONDA_SH" || ! -f "$_CONDA_SH" ]]; then
  echo "未找到 conda.sh：请 export CONDA_SH=/你的路径/miniconda3/etc/profile.d/conda.sh"
  exit 2
fi
# shellcheck disable=SC1091
source "$_CONDA_SH"
conda activate "${CONDA_ENV:-D4C}"
command -v python
python -V
echo "--- d4c step5 ---"
python code/d4c.py step5 \
  --task "$TASK" \
  --iter "$ITER" \
  --from-run "$FROM_RUN" \
  --step4-run "$STEP4_RUN" \
  --step5-run auto \
  --preset step5 \
  --eval-profile "$EVAL_PROFILE"
