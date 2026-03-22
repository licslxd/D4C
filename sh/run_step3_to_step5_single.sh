#!/bin/bash
# Step 3-5 单个任务：域对抗预训练 → 生成反事实 → 主训练（顺序执行）
# 用法: bash run_step3_to_step5_single.sh --task N [--from 3|4|5] [--eval-only] [--gpus 0,1] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]
#
#   --eval-only  Step 3 只跑 AdvTrain eval；Step 5 只跑 run-d4c 评估（均不重训），见 run_step3.sh / run_step5.sh
#
# ========== DDP（Step 3 与 Step 5）==========
# Step 3 的 train 与 eval 均 torchrun DDP；--gpus 主要给 Step 4。
# Step 3 / Step 5 均为 torchrun + DDP。进程数：
#   - 环境变量 DDP_NPROC，或参数 --ddp-nproc K（默认一般为 2）
#   - 全局 batch 须能被进程数整除；单卡：DDP_NPROC=1 或 --ddp-nproc 1
# 示例:
#   DDP_NPROC=1 bash run_step3_to_step5_single.sh --task 2
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3_to_step5_single.sh --task 2 --batch-size 1024
# 完整说明见 run_step3.sh 文件头。
#
# 其它示例:
#       bash run_step3_to_step5_single.sh --task 2 --from 4   # 跳过 Step 3，从 Step 4 续跑
#       bash run_step3_to_step5_single.sh --task 5 --gpus 0,1
#       bash run_step3_to_step5_single.sh --task 2 --batch-size 64 --epochs 30
#       bash run_step3_to_step5_single.sh --task 2 --daemon              # 后台跑（日志写入 log/，终端仅打印 PID）
#
#   --daemon / --bg   后台运行：stdout/stderr 写入 log/step3_to_5_taskN_*.log，终端立即返回

set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
cd "$CODE_DIR"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
fi

TASK_ID=""
GPUS=""
BATCH_SIZE=""
EPOCHS=""
NUM_PROC=""
DDP_EXTRA=""
FROM_STEP=3
EVAL_ONLY=""
DAEMON=""
prev=""
for arg in "$@"; do
    if [ "$arg" = "--eval-only" ]; then
        EVAL_ONLY=1
    elif [ "$arg" = "--daemon" ] || [ "$arg" = "--bg" ]; then
        DAEMON=1
    elif [ "$prev" = "--task" ] && [[ "$arg" =~ ^[1-8]$ ]]; then
        TASK_ID=$arg
    elif [ "$prev" = "--from" ]; then
        [[ "$arg" =~ ^[345]$ ]] || { echo "错误: --from 须为 3、4 或 5，收到: $arg"; exit 1; }
        FROM_STEP=$arg
    elif [ "$prev" = "--gpus" ]; then
        GPUS="--gpus $arg"
    elif [ "$prev" = "--batch-size" ]; then
        BATCH_SIZE="--batch-size $arg"
    elif [ "$prev" = "--epochs" ]; then
        EPOCHS="--epochs $arg"
    elif [ "$prev" = "--num-proc" ]; then
        NUM_PROC="--num-proc $arg"
    elif [ "$prev" = "--ddp-nproc" ]; then
        DDP_EXTRA="--ddp-nproc $arg"
    fi
    prev="$arg"
done

[ -z "$TASK_ID" ] && {
    echo "用法: $0 --task N [--from 3|4|5] [--eval-only] [--gpus 0,1] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  N 为 1-8；--eval-only：Step 3 / Step 5 只 eval（须已有 checkpoint）；DDP 见 run_step3.sh / run_step5.sh"
    echo "  --daemon / --bg：后台运行，日志写入 log/；详见文件头"
    exit 1
}

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step3_to_5_task${TASK_ID}_$(date +%Y%m%d_%H%M).log"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    echo "已在后台启动 Step 3-5 Task $TASK_ID，日志: $ABS_LOG"
    echo "查看进度: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

if [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step3_to_5_task${TASK_ID}_$(date +%Y%m%d_%H%M).log"
fi
[ "$FROM_STEP" -gt 3 ] && echo "从 Step $FROM_STEP 续跑（跳过 Step 3-$((FROM_STEP - 1))）"
echo "启动 Step 3-5 Task $TASK_ID，日志: $LOGFILE"
echo "查看进度: tail -f $LOGFILE"

if [ "$FROM_STEP" -le 3 ]; then
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Task $TASK_ID: Step 3 仅 eval（域对抗）=========="
    else
        echo "========== Task $TASK_ID: Step 3 域对抗预训练 =========="
    fi
    STEP3_EVAL=""
    [ -n "$EVAL_ONLY" ] && STEP3_EVAL="--eval-only"
    bash "$SH_DIR/run_step3.sh" --task $TASK_ID $STEP3_EVAL $BATCH_SIZE $EPOCHS $NUM_PROC $GPUS $DDP_EXTRA 2>&1 | tee "$LOGFILE"
else
    echo "========== 跳过 Task $TASK_ID Step 3 (--from $FROM_STEP) ==========" | tee "$LOGFILE"
fi

if [ "$FROM_STEP" -le 4 ]; then
    echo ""
    echo "========== Task $TASK_ID: Step 4 生成反事实 =========="
    bash "$SH_DIR/run_step4.sh" --task $TASK_ID $BATCH_SIZE $NUM_PROC $GPUS 2>&1 | tee -a "$LOGFILE"
else
    echo ""
    echo "========== 跳过 Task $TASK_ID Step 4 (--from $FROM_STEP) ==========" | tee -a "$LOGFILE"
fi

if [ "$FROM_STEP" -le 5 ]; then
    echo ""
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Task $TASK_ID: Step 5 仅 eval（主模型评估）=========="
    else
        echo "========== Task $TASK_ID: Step 5 主训练与评估 =========="
    fi
    STEP5_EVAL=""
    [ -n "$EVAL_ONLY" ] && STEP5_EVAL="--eval-only"
    bash "$SH_DIR/run_step5.sh" --task $TASK_ID $STEP5_EVAL $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"
fi

echo ""
echo "========== Task $TASK_ID Step 3-5 完成 =========="
