#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Python 单任务 Step5: python code/d4c.py step5|eval …；本脚本为 **多任务 1–8 批量** Shell 编排（内部 torchrun INTERNAL EXECUTOR）
# 亦见: bash scripts/train_ddp.sh — docs/D4C_Scripts_and_Runtime_Guide.md
# -----------------------------------------------------------------------------
# Step 5 批量：对任务 1–8 依次调用 run_step5_optimized.sh（仅嵌套 checkpoint；每任务自动选最新 step3_opt_*；--eval-only 时再选最新 step5_opt_*）
# 用法: bash run_step5_all.sh [--from N] [--skip N,M,...] [--eval-only|--train-only] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]
#
# 前置：各任务的 Step 3 + Step 4 已完成（checkpoints/<task>/step3_optimized/step3_opt_*/ 含 model.pth 与 factuals_counterfactuals.csv）
# 日志：每任务主日志在 log/<task>/step5_optimized/runs/…（见 run_step5_optimized.sh）；本脚本可选 tee 到 log/step5_all_*.log
#
# 示例:
#   DDP_NPROC=1 bash sh/run_step5_all.sh
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash sh/run_step5_all.sh --batch-size 1024
#   bash sh/run_step5_all.sh --from 4 --skip 2,5
#   bash sh/run_step5_all.sh --eval-only
#   bash sh/run_step5_all.sh --daemon

set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
cd "$CODE_DIR"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"

d4c_latest_step3_subdir() {
    local tid=$1
    local base="$D4C_ROOT/checkpoints/$tid/step3_optimized"
    [ -d "$base" ] || return 1
    local latest
    latest=$(ls -1td "$base"/step3_opt_* 2>/dev/null | head -1)
    [ -n "$latest" ] || return 1
    basename "$latest"
}
d4c_latest_step5_inner() {
    local tid=$1
    local s3=$2
    local sd="$D4C_ROOT/checkpoints/$tid/step3_optimized/$s3/step5"
    [ -d "$sd" ] || return 1
    local latest
    latest=$(ls -1td "$sd"/step5_opt_* 2>/dev/null | head -1)
    [ -n "$latest" ] || return 1
    basename "$latest"
}

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
fi

ORIG_ARGS=("$@")

FROM_TASK=1
BATCH_SIZE=""
EPOCHS=""
NUM_PROC=""
SKIP_LIST=""
DDP_EXTRA=""
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
while [ $# -gt 0 ]; do
    case $1 in
        --from) FROM_TASK=$2; shift 2 ;;
        --skip) SKIP_LIST=" $(echo "$2" | tr ',' ' ') "; shift 2 ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        --train-only) TRAIN_ONLY=1; shift ;;
        --batch-size) BATCH_SIZE="--batch-size $2"; shift 2 ;;
        --epochs) EPOCHS="--epochs $2"; shift 2 ;;
        --num-proc) NUM_PROC="--num-proc $2"; shift 2 ;;
        --ddp-nproc) DDP_EXTRA="--ddp-nproc $2"; shift 2 ;;
        --daemon|--bg) DAEMON=1; shift ;;
        --gpus|--gpus=*)
            echo "错误: --gpus has been removed. 请使用 CUDA_VISIBLE_DEVICES 与 DDP_NPROC / --ddp-nproc（torchrun --nproc_per_node）。" >&2
            echo "单卡 DDP smoke: DDP_NPROC=1 bash sh/run_step5_optimized.sh --task 1 --step3-subdir <NAME> …" >&2
            exit 2
            ;;
        *)
            echo "错误: 未知参数: $1" >&2
            echo "提示: 仅支持 --from / --skip / --eval-only / --train-only / --batch-size / --epochs / --num-proc / --ddp-nproc / --daemon|--bg。" >&2
            exit 2
            ;;
    esac
done

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

TASK_LIST=""
for i in 1 2 3 4 5 6 7 8; do
    [ "$i" -ge "$FROM_TASK" ] && ! should_skip "$i" && TASK_LIST="$TASK_LIST $i"
done
TASK_LIST=$(echo "$TASK_LIST" | xargs)
[ -z "$TASK_LIST" ] && { echo "错误: --from $FROM_TASK 无有效任务"; exit 1; }

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step5_all_$(date +%Y%m%d_%H%M).log"
    args=()
    for a in "${ORIG_ARGS[@]}"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    echo "已在后台启动 Step 5 全部任务 (Task $TASK_LIST)，终端汇总: $ABS_LOG"
    echo "查看进度: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

if [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step5_all_$(date +%Y%m%d_%H%M).log"
fi
[ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
echo "启动 Step 5 全部任务 (Task $TASK_LIST)，终端汇总: $LOGFILE"
echo "（每任务 Python 日志: log/<task>/step5_optimized/runs/…/train.log）"

STEP5_EVAL=""
[ -n "$EVAL_ONLY" ] && STEP5_EVAL="--eval-only"
STEP5_TRAIN=""
[ -n "$TRAIN_ONLY" ] && STEP5_TRAIN="--train-only"

for i in $TASK_LIST; do
    echo ""
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Task $i: Step 5 仅 eval =========="
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "========== Task $i: Step 5 仅 train =========="
    else
        echo "========== Task $i: Step 5 =========="
    fi
    STEP3_SUB="$(d4c_latest_step3_subdir "$i")" || {
        echo "错误: Task $i 未找到 checkpoints/$i/step3_optimized/step3_opt_*（须先完成 Step 3/4）" | tee -a "$LOGFILE"
        exit 1
    }
    NEST_ARG=()
    if [ -n "$EVAL_ONLY" ]; then
        _inn="$(d4c_latest_step5_inner "$i" "$STEP3_SUB")" || {
            echo "错误: Task $i 未找到 …/step3_optimized/$STEP3_SUB/step5/step5_opt_*（--eval-only 须已有 Step 5 训练目录）" | tee -a "$LOGFILE"
            exit 1
        }
        NEST_ARG=(--nested-subdir "$_inn")
    fi
    if ! bash "$SH_DIR/run_step5_optimized.sh" --task "$i" --step3-subdir "$STEP3_SUB" "${NEST_ARG[@]}" $STEP5_EVAL $STEP5_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"; then
        echo "Task $i Step 5 失败，可续跑: $0 --from $i $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA"
        exit 1
    fi
    echo "========== Task $i 完成 =========="
done
echo ""
echo "========== Step 5 全部任务完成 =========="
