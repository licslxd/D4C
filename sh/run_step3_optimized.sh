#!/bin/bash
#
# run_step3_optimized.sh — Step 3 工程优化入口（与论文复现 run_step3.sh 隔离）
#
# 作用：在调用 run_step3.sh 之前设置 D4C_TRAIN_MODE=optimized、大 batch、warmup+cosine、
#       quick/full BLEU 与早停相关环境变量，并默认将 checkpoint / 日志目录指向 *optimized* 命名空间，
#       避免与 checkpoints/<task>/step3/step3_<时间>/、log/<task>/step3/ 混淆。
#
# 用法：与 run_step3.sh 相同（--task / --all / --eval-only / --daemon 等），额外可通过环境变量覆盖默认。
#
# 默认（可用环境变量覆盖）：
#   D4C_OPT_BATCH_SIZE          若命令行未出现 --batch-size：默认 batch 为 python config.get_train_batch_size()（含 D4C_TRAIN_PRESET 预设时随之变化；否则模块默认 2048）
#   （不设 D4C_FULL_EVAL_EVERY 时由 Python 采用分阶段 full BLEU：默认 epoch≤10 每 5 轮、之后每 2 轮；固定间隔请 export N）
#   D4C_CHECKPOINT_GROUP=step3_optimized
#   D4C_CHECKPOINT_SUBDIR=step3_opt_<分秒时间戳>（若调用前已 export D4C_CHECKPOINT_SUBDIR 则保留）
#   D4C_LOG_GROUP=step3_optimized（若已设 D4C_LOG_GROUP / D4C_LOG_SUBDIR / D4C_LOG_STEP 则不覆盖）
#
# 示例：
#   bash sh/run_step3_optimized.sh --task 4
#   DDP_NPROC=2 D4C_OPT_BATCH_SIZE=1024 bash sh/run_step3_optimized.sh --task 2
#   D4C_TRAIN_PRESET=gb1024_ep30_fe2 bash sh/run_step3_optimized.sh --task 4   # 见 code/config.py TRAINING_PRESETS
#   D4C_FULL_EVAL_EVERY=3 TRAIN_EARLY_STOP_PATIENCE_FULL=5 bash sh/run_step3_optimized.sh --task 1
# Step 4 请用：bash sh/run_step4_optimized.sh --task N --step3-subdir <与上一步 SUBDIR 相同>
#
# ---------------------------------------------------------------------------

set -euo pipefail
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"

# ---- optimized 训练语义（与 config.py / AdvTrain 环境变量一致）----
export D4C_TRAIN_MODE=optimized
export D4C_LR_SCHEDULER="${D4C_LR_SCHEDULER:-warmup_cosine}"
export D4C_WARMUP_RATIO="${D4C_WARMUP_RATIO:-0.05}"
export D4C_QUICK_EVAL_MAX_SAMPLES="${D4C_QUICK_EVAL_MAX_SAMPLES:-512}"
export TRAIN_EARLY_STOP_PATIENCE_FULL="${TRAIN_EARLY_STOP_PATIENCE_FULL:-4}"
export TRAIN_MIN_EPOCHS="${TRAIN_MIN_EPOCHS:-8}"
export TRAIN_EARLY_STOP_PATIENCE="${TRAIN_EARLY_STOP_PATIENCE:-6}"
export TRAIN_BLEU4_MAX_SAMPLES="${TRAIN_BLEU4_MAX_SAMPLES:-512}"

# ---- 目录命名：checkpoint / 日志均带 optimized 标识 ----
export D4C_CHECKPOINT_GROUP="${D4C_CHECKPOINT_GROUP:-step3_optimized}"
if [ -z "${D4C_CHECKPOINT_SUBDIR:-}" ]; then
    _RUN_TS="$(date +%Y%m%d_%H%M)"
    export D4C_CHECKPOINT_SUBDIR="step3_opt_${_RUN_TS}"
fi
if [ -z "${D4C_LOG_GROUP:-}" ] && [ -z "${D4C_LOG_SUBDIR:-}" ] && [ -z "${D4C_LOG_STEP:-}" ]; then
    export D4C_LOG_GROUP=step3_optimized
fi

# 大 batch：未显式传 --batch-size 时，全局预设追加 get_train_batch_size()；按任务预设（TRAINING_PRESETS 键为 1..8）时不注入，由 AdvTrain 按任务解析
_extra=()
if [[ "$*" != *"--batch-size"* ]]; then
    if [ -n "${D4C_OPT_BATCH_SIZE:-}" ]; then
        _extra+=(--batch-size "$D4C_OPT_BATCH_SIZE")
    elif cd "$CODE_DIR" && python -c "from config import training_preset_is_per_task; import sys; sys.exit(0 if training_preset_is_per_task() else 1)"; then
        :
    else
        _default_bs="$(cd "$CODE_DIR" && python -c "from config import get_train_batch_size; print(get_train_batch_size())")"
        _extra+=(--batch-size "$_default_bs")
    fi
fi

echo "[run_step3_optimized] D4C_TRAIN_MODE=$D4C_TRAIN_MODE CHECKPOINT_GROUP=$D4C_CHECKPOINT_GROUP SUBDIR=$D4C_CHECKPOINT_SUBDIR"
if [ -n "${D4C_FULL_EVAL_EVERY:-}" ]; then
  export D4C_FULL_EVAL_EVERY
  _fe_echo="D4C_FULL_EVAL_EVERY=$D4C_FULL_EVAL_EVERY"
else
  _fe_echo="D4C_FULL_EVAL_EVERY=(unset → Python phased 5/10/2)"
fi
echo "[run_step3_optimized] ${_fe_echo} QUICK_EVAL_MAX=$D4C_QUICK_EVAL_MAX_SAMPLES PATIENCE_FULL=$TRAIN_EARLY_STOP_PATIENCE_FULL"

exec bash "$SH_DIR/run_step3.sh" "$@" "${_extra[@]}"
