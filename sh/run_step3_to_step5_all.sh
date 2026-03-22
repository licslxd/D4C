#!/bin/bash
# Step 3-5 全部任务：域对抗预训练 → 生成反事实 → 主训练（顺序执行）
# 用法: bash run_step3_to_step5_all.sh [--from N] [--skip N,M,...] [--eval-only] [--gpus 0,1] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]
#
#   --eval-only  Step 3 只跑 AdvTrain eval；Step 5 只跑 run-d4c 评估（均不重训），见 run_step3.sh / run_step5.sh
#
# ========== DDP（Step 3 与 Step 5）==========
# Step 3：torchrun + AdvTrain.py train 与 eval；Step 5：torchrun + run-d4c.py。
# 进程数由 DDP_NPROC 或 --ddp-nproc K 同时传给 run_step3.sh / run_step5.sh（默认 2）。
# 全局 --batch-size 须能被 DDP 进程数整除；单卡请 DDP_NPROC=1 或 --ddp-nproc 1。
# 示例:
#   DDP_NPROC=1 bash run_step3_to_step5_all.sh
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3_to_step5_all.sh --ddp-nproc 2 --batch-size 1024
#   bash run_step3_to_step5_all.sh --ddp-nproc 4 --gpus 0,1,2,3   # --gpus 给 Step4 等；Step3 eval 与 train 同 torchrun DDP
# 说明见 run_step3.sh、run_step5.sh 文件头。
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
DAEMON=""
while [ $# -gt 0 ]; do
    case $1 in
        --from) FROM_TASK=$2; shift 2 ;;
        --skip) SKIP_LIST=" $(echo "$2" | tr ',' ' ') "; shift 2 ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        --gpus) GPUS="--gpus $2"; shift 2 ;;
        --batch-size) BATCH_SIZE="--batch-size $2"; shift 2 ;;
        --epochs) EPOCHS="--epochs $2"; shift 2 ;;
        --num-proc) NUM_PROC="--num-proc $2"; shift 2 ;;
        --ddp-nproc) DDP_EXTRA="--ddp-nproc $2"; shift 2 ;;
        --daemon|--bg) DAEMON=1; shift ;;
        *) shift ;;
    esac
done

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
STEP5_EVAL=""
[ -n "$EVAL_ONLY" ] && STEP5_EVAL="--eval-only"

for i in $TASK_LIST; do
    echo ""
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Task $i: Step 3 仅 eval（域对抗）=========="
    else
        echo "========== Task $i: Step 3 域对抗预训练 =========="
    fi
    if ! bash "$SH_DIR/run_step3.sh" --task $i $STEP3_EVAL $BATCH_SIZE $EPOCHS $NUM_PROC $GPUS $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"; then
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
    echo "========== Task $i: Step 5 主训练与评估 =========="
    if ! bash "$SH_DIR/run_step5.sh" --task $i $STEP5_EVAL $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"; then
        echo "Task $i Step 5 失败，可续跑: $0 --from $i $BATCH_SIZE $EPOCHS $NUM_PROC $GPUS $DDP_EXTRA"
        exit 1
    fi
    echo "========== Task $i 完成 =========="
done
echo ""
echo "========== Step 3-5 全部任务完成 =========="
