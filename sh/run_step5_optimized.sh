#!/bin/bash
# -----------------------------------------------------------------------------
# MAINLINE: python code/d4c.py step5|eval …；本脚本仅编排 / 环境默认值 / nohup。
# Bash 包装: bash scripts/train_ddp.sh --step 5 …
# -----------------------------------------------------------------------------
# run_step5_optimized.sh — Step 5：主训练与评估（由 d4c.py 内部 torchrun）
# checkpoint 根目录为 step3_optimized/，日志默认 step5_optimized/；
# torchrun 前注入与 Step 3 一致的 eval/早停等环境默认值（可用环境变量覆盖）。
#
# 用法: bash run_step5_optimized.sh --task N --step3-subdir step3_opt_YYYYMMDD_HHMM [选项…]
#       --step3-subdir 须对应 checkpoints/<N>/step3_optimized/<该目录>/（通常由 run_step3_optimized.sh 生成）
#
# ========== DDP ==========
#   由 d4c.py 使用 torchrun；DDP_NPROC / --ddp-nproc → --ddp-world-size
#
# 示例:
#   DDP_NPROC=1 bash run_step5_optimized.sh --task 2 --step3-subdir step3_opt_20260324_1339
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step5_optimized.sh --task 4 --step3-subdir step3_opt_20260324_1339 --daemon
#   bash run_step5_optimized.sh --task 4 --step3-subdir step3_opt_… --seed 42   # 转发 d4c step5 --seed
# 多 seed 串行：bash sh/run_step5_multi_seed.sh …（见该脚本；论文汇总 scripts/multi_seed_paper_stats.py）
#
# ---------------------------------------------------------------------------
# Checkpoint（仅嵌套，optimized 命名空间）
# ---------------------------------------------------------------------------
# 目录：checkpoints/<task>/step3_optimized/<step3_opt_id>/step5/step5_opt_<分钟时间戳>/model.pth
# 环境：D4C_CHECKPOINT_GROUP=step3_optimized，D4C_CHECKPOINT_SUBDIR=<step3_opt_id>/step5/<内层>
#
# 日志（MAINLINE）：log/<task>/step5_optimized/<step5_run>/train.log
#
# ---------------------------------------------------------------------------
# D4C_TRAIN_PRESET：未设置时默认为 step5（与 step3 超参相同，见 presets/training/step5.yaml）
# 可选 runtime 预设（与 config.RUNTIME_PRESETS 一致，见 run_step3_optimized.sh 顶部说明）：
#   单卡 export D4C_RUNTIME_PRESET=gpu01_single_12c
#   双卡 export D4C_RUNTIME_PRESET=gpu01_ddp2_12c

set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
export D4C_ROOT
cd "$D4C_ROOT"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

# ---- 默认（与 run_step3_optimized.sh / config 一致；可被环境变量或 CLI 覆盖）----
# 未 export D4C_FULL_EVAL_EVERY 时不传 --full-eval-every，由 Python 使用分阶段 full BLEU 默认
export D4C_LR_SCHEDULER="${D4C_LR_SCHEDULER:-warmup_cosine}"
export D4C_WARMUP_RATIO="${D4C_WARMUP_RATIO:-0.05}"
export D4C_QUICK_EVAL_MAX_SAMPLES="${D4C_QUICK_EVAL_MAX_SAMPLES:-512}"
export TRAIN_EARLY_STOP_PATIENCE_FULL="${TRAIN_EARLY_STOP_PATIENCE_FULL:-4}"
export TRAIN_MIN_EPOCHS="${TRAIN_MIN_EPOCHS:-8}"
export TRAIN_EARLY_STOP_PATIENCE="${TRAIN_EARLY_STOP_PATIENCE:-6}"
export TRAIN_BLEU4_MAX_SAMPLES="${TRAIN_BLEU4_MAX_SAMPLES:-512}"
export D4C_TRAIN_PRESET="${D4C_TRAIN_PRESET:-step5}"

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
    export D4C_CONSOLE_LEVEL="${D4C_CONSOLE_LEVEL:-WARNING}"
fi

DDP_NPROC="${DDP_NPROC:-2}"

BATCH_SIZE=""
EPOCHS=""
NUM_PROC=""
SEED_CLI=()

TASK_ID=""
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
STEP3_SUBDIR_FOR_NEST=""
NEST_SUB_NAME=""
prev=""
for i in "$@"; do
    if [ "$i" = "--all" ]; then
        echo "错误: Step 5 已取消 --all（仅嵌套模式）。请逐任务: --task N --step3-subdir step3_opt_<时间戳>"
        exit 1
    elif [ "$i" = "--eval-only" ]; then EVAL_ONLY=1
    elif [ "$i" = "--train-only" ]; then TRAIN_ONLY=1
    elif [ "$i" = "--task" ]; then prev="--task"
    elif [ "$prev" = "--task" ] && [[ "$i" =~ ^[1-8]$ ]]; then TASK_ID=$i; prev=""
    elif [ "$i" = "--batch-size" ]; then prev="--batch-size"
    elif [ "$prev" = "--batch-size" ]; then BATCH_SIZE="--batch-size $i"; prev=""
    elif [ "$i" = "--epochs" ]; then prev="--epochs"
    elif [ "$prev" = "--epochs" ]; then EPOCHS="--epochs $i"; prev=""
    elif [ "$i" = "--num-proc" ]; then prev="--num-proc"
    elif [ "$prev" = "--num-proc" ]; then NUM_PROC="--num-proc $i"; prev=""
    elif [ "$i" = "--ddp-nproc" ]; then prev="--ddp-nproc"
    elif [ "$prev" = "--ddp-nproc" ]; then DDP_NPROC="$i"; prev=""
    elif [ "$i" = "--step3-subdir" ]; then prev="--step3-subdir"
    elif [ "$prev" = "--step3-subdir" ]; then STEP3_SUBDIR_FOR_NEST="$i"; prev=""
    elif [ "$i" = "--seed" ]; then prev="--seed"
    elif [ "$prev" = "--seed" ]; then SEED_CLI=(--seed "$i"); prev=""
    elif [ "$i" = "--nested-subdir" ]; then prev="--nested-subdir"
    elif [ "$prev" = "--nested-subdir" ]; then NEST_SUB_NAME="$i"; prev=""
    elif [ "$i" = "--gpus" ] || [[ "$i" == --gpus=* ]]; then
        echo "错误: --gpus has been removed. 请使用 CUDA_VISIBLE_DEVICES 与 DDP_NPROC / --ddp-nproc。" >&2
        exit 2
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

if [ -n "${NEST_SUB_NAME:-}" ] && [ -z "${STEP3_SUBDIR_FOR_NEST:-}" ]; then
    echo "错误: --nested-subdir 须与 --step3-subdir 同时使用"
    exit 1
fi

if ! [[ "$TASK_ID" =~ ^[1-8]$ ]]; then
    echo "用法: $0 --task N --step3-subdir step3_opt_YYYYMMDD_HHMM [选项]"
    echo "  必填: --task N（1–8）、--step3-subdir（与 checkpoints/<N>/step3_optimized/ 下目录名一致）"
    echo "  可选: --nested-subdir <内层名>（训练默认 step5_opt_<分钟时间戳>；--eval-only 必填）"
    echo "  可选: --eval-only | --train-only（互斥）、--seed N、--batch-size、--epochs、--num-proc、--ddp-nproc K、--daemon|--bg"
    echo "  布局: …/step3_optimized/<step3_opt_id>/step5/<内层>/model.pth"
    exit 1
fi

if [ -z "${STEP3_SUBDIR_FOR_NEST:-}" ]; then
    echo "错误: Step 5 仅支持嵌套 checkpoint，须指定 --step3-subdir（见用法）"
    exit 1
fi

if [ -n "$EVAL_ONLY" ]; then
    if [ -z "${NEST_SUB_NAME:-}" ]; then
        echo "错误: --eval-only 须指定 --nested-subdir <已有内层目录名>，例如 step5_opt_20260328_1123"
        exit 1
    fi
    _inner="$NEST_SUB_NAME"
elif [ -n "${NEST_SUB_NAME:-}" ]; then
    _inner="$NEST_SUB_NAME"
else
    _inner="step5_opt_$(date +%Y%m%d_%H%M)"
fi

if [ -z "$_inner" ]; then
    echo "错误: --nested-subdir 不能为空"
    exit 1
fi

CKPT_BASE="$D4C_ROOT/checkpoints/$TASK_ID/step3_optimized/$STEP3_SUBDIR_FOR_NEST"
CSV="$CKPT_BASE/factuals_counterfactuals.csv"
STEP3_MODEL="$CKPT_BASE/model.pth"
if [ ! -d "$CKPT_BASE" ]; then
    echo "错误: Step 3 目录不存在: $CKPT_BASE"
    exit 1
fi
if [ ! -f "$CSV" ]; then
    echo "错误: 未找到 Step 4 产物: $CSV"
    exit 1
fi
if [ ! -f "$STEP3_MODEL" ]; then
    echo "提示: 未找到 Step 3 权重（部分场景可忽略）: $STEP3_MODEL"
fi

NEST_DIR="$CKPT_BASE/step5/$_inner"
if [ -n "$EVAL_ONLY" ]; then
    if [ ! -d "$NEST_DIR" ]; then
        echo "错误: --eval-only 所指的嵌套目录不存在: $NEST_DIR"
        exit 1
    fi
    if [ ! -f "$NEST_DIR/model.pth" ]; then
        echo "错误: --eval-only 需要已有权重: $NEST_DIR/model.pth"
        exit 1
    fi
fi

echo "Step 5 checkpoint（预期）: $NEST_DIR"
echo "主日志（d4c）: $D4C_ROOT/log/$TASK_ID/step5_optimized/${_inner}/train.log"

_RUN_TS="$(date +%Y%m%d_%H%M)"

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$D4C_ROOT/log/${TASK_ID}/step5_optimized/${_inner}/train.log"
    mkdir -p "$(dirname "$LOGFILE")"
    NOHUP_OUT="$(dirname "$LOGFILE")/nohup.log"
    mkdir -p "$(dirname "$LOGFILE")"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_TRAIN="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    ABS_NOHUP="$(readlink -f "$NOHUP_OUT" 2>/dev/null || echo "$NOHUP_OUT")"
    echo "已在后台启动 Step 5（d4c.py）"
    echo "  Python 日志 (--log_file): $ABS_TRAIN"
    echo "  nohup 终端输出: $ABS_NOHUP"
    echo "查看训练日志: tail -f $ABS_TRAIN"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$NOHUP_OUT" 2>&1 &
    echo "PID: $!"
    exit 0
fi

if [ -z "$EPOCHS" ] && [ -z "$EVAL_ONLY" ]; then
    EPOCHS="--epochs $(cd "$CODE_DIR" && D4C_PRESET_TASK_ID="$TASK_ID" python -c "from config import get_epochs; print(get_epochs())")"
fi

if [ -z "$BATCH_SIZE" ] && [ -z "$EVAL_ONLY" ]; then
    if [ -n "${D4C_OPT_BATCH_SIZE:-}" ]; then
        BATCH_SIZE="--batch-size $D4C_OPT_BATCH_SIZE"
    elif (cd "$CODE_DIR" && python -c "from config import training_preset_is_per_task; import sys; sys.exit(0 if training_preset_is_per_task() else 1)"); then
        BATCH_SIZE=""
    else
        _default_bs="$(cd "$CODE_DIR" && D4C_PRESET_TASK_ID="$TASK_ID" python -c "from config import get_train_batch_size; print(get_train_batch_size())")"
        BATCH_SIZE="--batch-size $_default_bs"
    fi
fi

PRESET="${D4C_TRAIN_PRESET:-step5}"
_trainonly=()
[ -n "$TRAIN_ONLY" ] && _trainonly=(--train-only)

if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 5 Task $TASK_ID eval（d4c.py, DDP nproc=$DDP_NPROC）=========="
    echo "  评测日志: $D4C_ROOT/log/$TASK_ID/step5_optimized/${_inner}/eval.log"
    # shellcheck disable=SC2086
    python "$D4C_ROOT/code/d4c.py" eval --task "$TASK_ID" --preset "$PRESET" \
        --from-run "$STEP3_SUBDIR_FOR_NEST" --step5-run "$_inner" \
        --ddp-world-size "$DDP_NPROC" \
        $BATCH_SIZE $NUM_PROC "${SEED_CLI[@]}"
else
    echo "========== Step 5 Task $TASK_ID train（d4c.py, DDP nproc=$DDP_NPROC）=========="
    echo "  训练日志: $D4C_ROOT/log/$TASK_ID/step5_optimized/${_inner}/train.log"
    # shellcheck disable=SC2086
    python "$D4C_ROOT/code/d4c.py" step5 --task "$TASK_ID" --preset "$PRESET" \
        --from-run "$STEP3_SUBDIR_FOR_NEST" --step5-run "$_inner" \
        --ddp-world-size "$DDP_NPROC" \
        $BATCH_SIZE $EPOCHS $NUM_PROC "${SEED_CLI[@]}" \
        "${_trainonly[@]}"
fi

if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 5（仅 eval）完成 =========="
elif [ -n "$TRAIN_ONLY" ]; then
    echo "========== Step 5（仅 train）完成 =========="
else
    echo "========== Step 5 完成 =========="
fi
