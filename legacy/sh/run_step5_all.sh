#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# LEGACY — 多任务 1–8 Step5 批量。主线：bash scripts/train_ddp.sh（见 docs/D4C_Scripts_and_Runtime_Guide.md）
# 考古说明：docs/legacy_batch_shell.md
# -----------------------------------------------------------------------------
# Step 5 批量：对任务 1–8 依次调用 scripts/entrypoints/step5.sh；每任务自动选最新 runs/.../train/step3/ 下子目录（sort -V）；--eval-only 时再选最新 train/step5/ 子目录
# 用法: bash run_step5_all.sh [--iter vN] [--from N] [--skip N,M,...] [--eval-only|--train-only] …
#
# 前置：各任务 Step 3 + Step 4 已完成（runs/task{T}/vN/train/step3/<run>/ 含权重；CSV 在 step4 或 step3 run 下）
# 可选汇总 tee：runs/global/vN/meta/shell_logs/step5_all_*.log；主日志见 runs/task{T}/vN/train/step5/<run>/logs/train.log
#
# LEGACY 批量编排：官方主线批量请用 bash scripts/train_ddp.sh（仓库根）。
# 示例:
#   DDP_NPROC=1 bash legacy/sh/run_step5_all.sh
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash legacy/sh/run_step5_all.sh --batch-size 1024

set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/../.." && pwd)"
MAIN_SH="$D4C_ROOT/scripts/entrypoints"
CODE_DIR="$D4C_ROOT/code"
cd "$CODE_DIR"
# shellcheck source=../../scripts/lib/common_logs.sh
source "$D4C_ROOT/scripts/lib/common_logs.sh"
PIPELINE_EVAL_PROFILE="${D4C_PIPELINE_EVAL_PROFILE:-eval_fast_single_gpu}"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"

d4c_latest_step3_subdir() {
    local tid=$1
    local iter="${D4C_ITER:-v1}"
    local base="$D4C_ROOT/runs/task${tid}/${iter}/train/step3"
    [ -d "$base" ] || return 1
    local latest
    latest=$(find "$base" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | sort -V | tail -1)
    [ -n "$latest" ] || return 1
    basename "$latest"
}
d4c_latest_step4_subdir() {
    local tid=$1
    local iter="${D4C_ITER:-v1}"
    local base="$D4C_ROOT/runs/task${tid}/${iter}/train/step4"
    [ -d "$base" ] || return 1
    local latest
    latest=$(find "$base" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | sort -V | tail -1)
    [ -n "$latest" ] || return 1
    basename "$latest"
}
d4c_latest_step5_run() {
    local tid=$1
    local iter="${D4C_ITER:-v1}"
    local sd="$D4C_ROOT/runs/task${tid}/${iter}/train/step5"
    [ -d "$sd" ] || return 1
    local latest
    latest=$(find "$sd" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | sort -V | tail -1)
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
        --iter) export D4C_ITER="$2"; shift 2 ;;
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
            echo "单卡 DDP smoke: DDP_NPROC=1 bash scripts/entrypoints/step5.sh --task 1 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu …" >&2
            exit 2
            ;;
        *)
            echo "错误: 未知参数: $1" >&2
            echo "提示: 仅支持 --from / --skip / --eval-only / --train-only / --batch-size / --epochs / --num-proc / --ddp-nproc / --daemon|--bg。" >&2
            exit 2
            ;;
    esac
done

export D4C_ITER="${D4C_ITER:-v1}"

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
    LOGFILE="$(d4c_shell_logs_global "$D4C_ROOT")/step5_all_$(date +%Y%m%d_%H%M).log"
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
    LOGFILE="$(d4c_shell_logs_global "$D4C_ROOT")/step5_all_$(date +%Y%m%d_%H%M).log"
fi
[ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
echo "启动 Step 5 全部任务 (Task $TASK_LIST)，终端汇总: $LOGFILE"
echo "（每任务主日志: runs/task{T}/$D4C_ITER/train/step5/<run>/logs/train.log）"

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
    FROM_RUN="$(d4c_latest_step3_subdir "$i")" || {
        echo "错误: Task $i 未找到 runs/task${i}/${D4C_ITER}/train/step3/ 下有效子目录（须先完成 Step 3/4）" | tee -a "$LOGFILE"
        exit 1
    }
    S5_ARG=()
    if [ -n "$EVAL_ONLY" ]; then
        _s5="$(d4c_latest_step5_run "$i")" || {
            echo "错误: Task $i 未找到 runs/task${i}/${D4C_ITER}/train/step5/ 下有效子目录（--eval-only 须已有 Step 5）" | tee -a "$LOGFILE"
            exit 1
        }
        S5_ARG=(--step5-run "$_s5")
    fi
    S5_EP=()
    if [ -n "$STEP5_TRAIN" ] && [ "$STEP5_TRAIN" = "--train-only" ]; then
        S5_EP=()
    else
        S5_EP=(--eval-profile "$PIPELINE_EVAL_PROFILE")
    fi
    S5_S4RUN=()
    if [ -z "$EVAL_ONLY" ]; then
        _s4="$(d4c_latest_step4_subdir "$i")" || {
            echo "错误: Task $i 未找到 train/step4 下有效子目录（须先完成 Step 4）" | tee -a "$LOGFILE"
            exit 1
        }
        S5_S4RUN=(--step4-run "$_s4")
    fi
    if ! bash "$MAIN_SH/step5.sh" --iter "$D4C_ITER" --task "$i" --from-run "$FROM_RUN" "${S5_ARG[@]}" "${S5_S4RUN[@]}" "${S5_EP[@]}" $STEP5_EVAL $STEP5_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"; then
        echo "Task $i Step 5 失败，可续跑: $0 --from $i $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA"
        exit 1
    fi
    echo "========== Task $i 完成 =========="
done
echo ""
echo "========== Step 5 全部任务完成 =========="
