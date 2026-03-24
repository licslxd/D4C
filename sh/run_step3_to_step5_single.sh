#!/bin/bash
# Step 3-5 单个任务：域对抗预训练 → 生成反事实 → 主训练（顺序执行）
# 用法: bash run_step3_to_step5_single.sh --task N [--from 3|4|5] [--eval-only|--train-only] [--gpus 0,1] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]
#
#   --eval-only  Step 3 只跑 AdvTrain eval；Step 5 只跑 run-d4c 评估（均不重训），见 run_step3.sh / run_step5.sh
#   --train-only Step 3 / Step 5 跳过训练后的收尾 eval（与 --eval-only 互斥）
#
# ========== DDP（Step 3、Step 4、Step 5）==========
# Step 3 / Step 4 / Step 5 均为 torchrun + DDP（见各 run_step*.sh）。进程数：
#   - 环境变量 DDP_NPROC，或参数 --ddp-nproc K（默认一般为 2）
#   - 全局 batch 须能被进程数整除；单卡：DDP_NPROC=1 或 --ddp-nproc 1
# 串联脚本传入的 --gpus 对 Step 4 无实际作用（Step 4 已 torchrun）；请用 CUDA_VISIBLE_DEVICES。
# 示例:
#   DDP_NPROC=1 bash run_step3_to_step5_single.sh --task 2
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3_to_step5_single.sh --task 2 --batch-size 1024
# 完整说明见 run_step3.sh、run_step4.sh、run_step5.sh 文件头。
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

# Step 5 仅嵌套：从磁盘解析当前任务最新的 step3_* / step5_*（供串联调用 run_step5.sh）
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

TASK_ID=""
GPUS=""
BATCH_SIZE=""
EPOCHS=""
NUM_PROC=""
DDP_EXTRA=""
FROM_STEP=3
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
prev=""
for arg in "$@"; do
    if [ "$arg" = "--eval-only" ]; then
        EVAL_ONLY=1
    elif [ "$arg" = "--train-only" ]; then
        TRAIN_ONLY=1
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

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

[ -z "$TASK_ID" ] && {
    echo "用法: $0 --task N [--from 3|4|5] [--eval-only|--train-only] [--gpus 0,1] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  N 为 1-8；--eval-only：Step 3 / Step 5 只 eval（须已有 checkpoint）；--train-only：跳过 Step3/5 训练后收尾 eval（互斥）；DDP 见 run_step3.sh / run_step5.sh"
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
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "========== Task $TASK_ID: Step 3 域对抗预训练（仅 train）=========="
    else
        echo "========== Task $TASK_ID: Step 3 域对抗预训练 =========="
    fi
    STEP3_EVAL=""
    [ -n "$EVAL_ONLY" ] && STEP3_EVAL="--eval-only"
    STEP3_TRAIN=""
    [ -n "$TRAIN_ONLY" ] && STEP3_TRAIN="--train-only"
    bash "$SH_DIR/run_step3.sh" --task $TASK_ID $STEP3_EVAL $STEP3_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $GPUS $DDP_EXTRA 2>&1 | tee "$LOGFILE"
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
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "========== Task $TASK_ID: Step 5 主训练（仅 train）=========="
    else
        echo "========== Task $TASK_ID: Step 5 主训练与评估 =========="
    fi
    STEP5_EVAL=""
    [ -n "$EVAL_ONLY" ] && STEP5_EVAL="--eval-only"
    STEP5_TRAIN=""
    [ -n "$TRAIN_ONLY" ] && STEP5_TRAIN="--train-only"
    STEP3_SUB="$(d4c_latest_step3_subdir "$TASK_ID")" || {
        echo "错误: 未找到 checkpoints/$TASK_ID/step3/step3_*（须先完成 Step 3）" | tee -a "$LOGFILE"
        exit 1
    }
    NEST_ARG=()
    if [ -n "$EVAL_ONLY" ]; then
        _inn="$(d4c_latest_step5_inner "$TASK_ID" "$STEP3_SUB")" || {
            echo "错误: 未找到 …/step3/$STEP3_SUB/step5/step5_*（--eval-only 须已有 Step 5 训练目录）" | tee -a "$LOGFILE"
            exit 1
        }
        NEST_ARG=(--nested-subdir "$_inn")
    fi
    bash "$SH_DIR/run_step5.sh" --task "$TASK_ID" --step3-subdir "$STEP3_SUB" "${NEST_ARG[@]}" $STEP5_EVAL $STEP5_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"
fi

echo ""
echo "========== Task $TASK_ID Step 3-5 完成 =========="
