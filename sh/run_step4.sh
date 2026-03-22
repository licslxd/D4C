#!/bin/bash
# Step 4：生成反事实数据
# 用法: bash run_step4.sh --all              # 跑全部 8 个任务
#       bash run_step4.sh --task N            # 仅跑任务 N (1-8)
#       bash run_step4.sh --all --from 4      # 从 Task 4 起跑到 8
#       bash run_step4.sh --all --skip 2,5    # 跑全部，跳过任务 2 和 5
#       bash run_step4.sh --all --gpus 0,1    # 多卡（generate_counterfactual 内 DataParallel，非 DDP）
#       bash run_step4.sh --task 2 --batch-size 64  # 指定推理 batch
#       bash run_step4.sh --all --daemon             # 后台跑（日志写入 log/，终端仅打印 PID）
# 示例: bash run_step4.sh --task 2
#       bash run_step4.sh --all --skip 2,5
#
#   --daemon / --bg   后台运行：stdout/stderr 写入 log/step4_*.log，终端立即返回
#
# ========== DDP 说明 ==========
# 本脚本不跑域对抗训练，无 torchrun DDP。需先用 Step 3 训好 checkpoint；Step 3 的 DDP 用法见 run_step3.sh，例如：
#   DDP_NPROC=2 bash run_step3.sh --task 2
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3.sh --all

set -e
# 脚本位于项目根下 sh/，Python 入口在 code/
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$CODE_DIR"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
fi

MODE=""
TASK_ID=""
GPUS=""
BATCH_SIZE=""
NUM_PROC=""
SKIP_LIST=""
FROM_TASK=1
DAEMON=""
prev=""
for i in "$@"; do
    if [ "$i" = "--all" ]; then MODE="all"
    elif [ "$i" = "--task" ]; then prev="--task"
    elif [ "$prev" = "--task" ] && [[ "$i" =~ ^[1-8]$ ]]; then TASK_ID=$i; prev=""
    elif [ "$i" = "--from" ]; then prev="--from"
    elif [ "$prev" = "--from" ]; then
        [[ "$i" =~ ^[1-8]$ ]] || { echo "错误: --from 须为 1-8，收到: $i"; exit 1; }
        FROM_TASK=$i
        prev=""
    elif [ "$i" = "--skip" ]; then prev="--skip"
    elif [ "$prev" = "--skip" ]; then SKIP_LIST=" $(echo "$i" | tr ',' ' ') "; prev=""
    elif [ "$i" = "--gpus" ]; then prev="--gpus"
    elif [ "$prev" = "--gpus" ]; then GPUS="--gpus $i"; prev=""
    elif [ "$i" = "--batch-size" ]; then prev="--batch-size"
    elif [ "$prev" = "--batch-size" ]; then BATCH_SIZE="--batch-size $i"; prev=""
    elif [ "$i" = "--num-proc" ]; then prev="--num-proc"
    elif [ "$prev" = "--num-proc" ]; then NUM_PROC="--num-proc $i"; prev=""
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --all | --task N [--from N] [--skip N,M,...] [--gpus 0,1] [--batch-size N] [--num-proc N] [--daemon|--bg]"
    echo "  Step 3 域对抗 DDP：见同目录 run_step3.sh"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step4_$(date +%Y%m%d_%H%M).log"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    echo "已在后台启动 Step 4，日志: $ABS_LOG"
    echo "查看进度: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step4_$(date +%Y%m%d_%H%M).log"
fi
if [ "$MODE" = "all" ]; then
    [ "$FROM_TASK" -gt 1 ] && echo "从 Task $FROM_TASK 起跑（跳过 1-$((FROM_TASK - 1))）"
    [ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
    echo "启动 Step 4 全部 8 个任务，日志: $LOGFILE"
    echo "查看进度: tail -f $LOGFILE"
    for i in 1 2 3 4 5 6 7 8; do
        [ "$i" -lt "$FROM_TASK" ] && { echo "========== 跳过 Task $i (--from $FROM_TASK) ==========" | tee -a "$LOGFILE"; continue; }
        should_skip $i && { echo "========== 跳过 Task $i ==========" | tee -a "$LOGFILE"; continue; }
        python generate_counterfactual.py --task $i $BATCH_SIZE $NUM_PROC $GPUS 2>&1 | tee -a "$LOGFILE" || { echo "Task $i 失败"; exit 1; }
    done
else
    echo "启动 Step 4 Task $TASK_ID，日志: $LOGFILE"
    python generate_counterfactual.py --task $TASK_ID $BATCH_SIZE $NUM_PROC $GPUS 2>&1 | tee "$LOGFILE"
fi
echo "========== Step 4 完成 =========="
