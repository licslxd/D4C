#!/bin/bash
# Step 3-5 全部任务：域对抗预训练 → 生成反事实 → 主训练（顺序执行）
# 用法: bash run_step3_to_step5_all.sh [--from N] [--skip N,M,...] [--eval-only|--train-only] [--gpus 0,1] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]
#
#   --eval-only  Step 3 只跑 AdvTrain eval；Step 5 只跑 run-d4c 评估（均不重训），见 run_step3.sh / run_step5.sh
#   --train-only Step 3 / Step 5 跳过训练后收尾 eval（与 --eval-only 互斥）
#
# ========== DDP（Step 3、Step 4、Step 5）==========
# Step 3：torchrun + AdvTrain；Step 4：torchrun + generate_counterfactual.py；Step 5：torchrun + run-d4c.py。
# 进程数由 DDP_NPROC 或 --ddp-nproc K 传给上述脚本（默认 2）；Step 4 与 3/5 一致须整除全局 batch。
# 单卡请 DDP_NPROC=1 或 --ddp-nproc 1。
# 示例:
#   DDP_NPROC=1 bash run_step3_to_step5_all.sh
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3_to_step5_all.sh --ddp-nproc 2 --batch-size 1024
#   bash run_step3_to_step5_all.sh --ddp-nproc 4 --gpus 0,1,2,3   # --gpus 仅历史兼容；Step4 已 torchrun，以 CUDA_VISIBLE_DEVICES 为准
# 说明见 run_step3.sh、run_step4.sh、run_step5.sh 文件头。
#
# 其它示例:
#       bash run_step3_to_step5_all.sh --from 4           # 从 Task 4 续跑（跳过 1-3）
#       bash run_step3_to_step5_all.sh --skip 2,5        # 跑全部，跳过任务 2 和 5
#       bash run_step3_to_step5_all.sh --gpus 0,1
#       bash run_step3_to_step5_all.sh --batch-size 64 --epochs 30
#       bash run_step3_to_step5_all.sh --daemon                    # 后台跑（日志写入 log/，终端仅打印 PID）
#
#   --daemon / --bg   后台运行：stdout/stderr 写入 log/step3_to_5_all_*.log，终端立即返回

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
    local base="$D4C_ROOT/checkpoints/$tid/step3"
    [ -d "$base" ] || return 1
    local latest
    latest=$(ls -1td "$base"/step3_* 2>/dev/null | head -1)
    [ -n "$latest" ] || return 1
    basename "$latest"
}
d4c_latest_step5_inner() {
    local tid=$1
    local s3=$2
    local sd="$D4C_ROOT/checkpoints/$tid/step3/$s3/step5"
    [ -d "$sd" ] || return 1
    local latest
    latest=$(ls -1td "$sd"/step5_* 2>/dev/null | head -1)
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
GPUS=""
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
        --gpus) GPUS="--gpus $2"; shift 2 ;;
        --batch-size) BATCH_SIZE="--batch-size $2"; shift 2 ;;
        --epochs) EPOCHS="--epochs $2"; shift 2 ;;
        --num-proc) NUM_PROC="--num-proc $2"; shift 2 ;;
        --ddp-nproc) DDP_EXTRA="--ddp-nproc $2"; shift 2 ;;
        --daemon|--bg) DAEMON=1; shift ;;
        *) shift ;;
    esac
done

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

TASK_LIST=""
for i in 1 2 3 4 5 6 7 8; do
    [ $i -ge $FROM_TASK ] && ! should_skip $i && TASK_LIST="$TASK_LIST $i"
done
TASK_LIST=$(echo $TASK_LIST | xargs)
[ -z "$TASK_LIST" ] && { echo "错误: --from $FROM_TASK 无有效任务"; exit 1; }

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step3_to_5_all_$(date +%Y%m%d_%H%M).log"
    args=()
    for a in "${ORIG_ARGS[@]}"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    echo "已在后台启动 Step 3-5 全部任务 (Task $TASK_LIST)，日志: $ABS_LOG"
    echo "查看进度: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

if [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step3_to_5_all_$(date +%Y%m%d_%H%M).log"
fi
[ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
echo "启动 Step 3-5 全部任务 (Task $TASK_LIST)，日志: $LOGFILE"
echo "查看进度: tail -f $LOGFILE"

STEP3_EVAL=""
[ -n "$EVAL_ONLY" ] && STEP3_EVAL="--eval-only"
STEP3_TRAIN=""
[ -n "$TRAIN_ONLY" ] && STEP3_TRAIN="--train-only"
STEP5_EVAL=""
[ -n "$EVAL_ONLY" ] && STEP5_EVAL="--eval-only"
STEP5_TRAIN=""
[ -n "$TRAIN_ONLY" ] && STEP5_TRAIN="--train-only"

for i in $TASK_LIST; do
    echo ""
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Task $i: Step 3 仅 eval（域对抗）=========="
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "========== Task $i: Step 3 域对抗预训练（仅 train）=========="
    else
        echo "========== Task $i: Step 3 域对抗预训练 =========="
    fi
    if ! bash "$SH_DIR/run_step3.sh" --task $i $STEP3_EVAL $STEP3_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $GPUS $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"; then
        echo "Task $i Step 3 失败，可续跑: $0 --from $i $BATCH_SIZE $EPOCHS $NUM_PROC $GPUS $DDP_EXTRA"
        exit 1
    fi
    echo ""
    echo "========== Task $i: Step 4 生成反事实 =========="
    if ! bash "$SH_DIR/run_step4.sh" --task $i $BATCH_SIZE $NUM_PROC $GPUS 2>&1 | tee -a "$LOGFILE"; then
        echo "Task $i Step 4 失败，可续跑: $0 --from $i $BATCH_SIZE $NUM_PROC $GPUS"
        exit 1
    fi
    echo ""
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Task $i: Step 5 仅 eval（主模型评估）=========="
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "========== Task $i: Step 5 主训练（仅 train）=========="
    else
        echo "========== Task $i: Step 5 主训练与评估 =========="
    fi
    STEP3_SUB="$(d4c_latest_step3_subdir "$i")" || {
        echo "错误: Task $i 未找到 checkpoints/$i/step3/step3_*（须先完成 Step 3）" | tee -a "$LOGFILE"
        exit 1
    }
    NEST_ARG=()
    if [ -n "$EVAL_ONLY" ]; then
        _inn="$(d4c_latest_step5_inner "$i" "$STEP3_SUB")" || {
            echo "错误: Task $i 未找到 …/step3/$STEP3_SUB/step5/step5_*（--eval-only 须已有 Step 5 目录）" | tee -a "$LOGFILE"
            exit 1
        }
        NEST_ARG=(--nested-subdir "$_inn")
    fi
    if ! bash "$SH_DIR/run_step5.sh" --task "$i" --step3-subdir "$STEP3_SUB" "${NEST_ARG[@]}" $STEP5_EVAL $STEP5_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"; then
        echo "Task $i Step 5 失败，可续跑: $0 --from $i $BATCH_SIZE $EPOCHS $NUM_PROC $GPUS $DDP_EXTRA"
        exit 1
    fi
    echo "========== Task $i 完成 =========="
done
echo ""
echo "========== Step 3-5 全部任务完成 =========="
