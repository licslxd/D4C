#!/bin/bash
# Step 4：生成反事实数据（优先 torchrun + DDP 推理，与 Step 3/5 对齐）
# 用法: bash run_step4.sh --all              # 跑全部 8 个任务
#       bash run_step4.sh --task N            # 仅跑任务 N (1-8)
#       bash run_step4.sh --all --from 4      # 从 Task 4 起跑到 8
#       bash run_step4.sh --all --skip 2,5    # 跑全部，跳过任务 2 和 5
#       bash run_step4.sh --task 2 --batch-size 64
#       bash run_step4.sh --all --daemon             # 后台跑（终端汇总 log/step4_daemon_*.log）
# 示例: bash run_step4.sh --task 2
#       CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step4.sh --task 2 --batch-size 1024
#
#   --daemon / --bg   单任务：get_log_task_dir(task)/runs/<时间戳>/train.log + 同目录 nohup.log
#                     --all：另有终端汇总 log/step4_daemon_*.log（与 step3_daemon 对齐）
#
# 日志与 checkpoint 解耦：本脚本默认 export D4C_LOG_STEP=step4 → log/<task>/step4/runs/.../train.log
# （读权重仍只认 D4C_CHECKPOINT_*，须与 Step 3 训练一致）。可改设 D4C_LOG_GROUP / D4C_LOG_SUBDIR 覆盖布局，见 paths_config.get_log_task_dir。
#
# ========== DDP（torchrun，与 run_step3.sh / run_step5.sh 一致）==========
#   torchrun --standalone --nproc_per_node=$DDP_NPROC generate_counterfactual.py ...
# 进程数：环境变量 DDP_NPROC，或参数 --ddp-nproc K（默认 DDP_NPROC=2）
# 须与可见 GPU 数一致；全局 --batch-size 须能被 DDP_NPROC 整除；单卡：DDP_NPROC=1
# 多卡请勿再传 --gpus（Python 在 DDP 下会忽略）；请用 CUDA_VISIBLE_DEVICES。
#
# 手动单进程（不推荐）：cd code && python generate_counterfactual.py --task N [--gpus 0,1]
# 此时多卡为脚本内 DataParallel，无 torchrun。

set -euo pipefail
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$CODE_DIR"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

# 未设置（unset）时默认 step4；若需主日志与 D4C_CHECKPOINT_* 同表（如 log/<task>/step3/），可 export D4C_LOG_STEP=
export D4C_LOG_STEP="${D4C_LOG_STEP-step4}"

_RUN_TS="$(date +%Y%m%d_%H%M)"
_LOG_TS="$(date +%Y%m%d_%H%M%S)"

d4c_run_log_dir() {
    python -c "from paths_config import get_log_task_dir; print(get_log_task_dir($1))"
}
d4c_run_log_path() {
    python -c "from paths_config import get_log_task_dir; import os; print(os.path.join(get_log_task_dir($1), 'runs', 'run', 'train.log'))"
}
d4c_step4_logfile() {
    local tid=$1
    local base
    base="$(d4c_run_log_dir "$tid")"
    if [ "${D4C_LOG_USE_TIMESTAMP:-1}" != "0" ]; then
        echo "${base}/runs/${_LOG_TS}/train.log"
    else
        d4c_run_log_path "$tid"
    fi
}

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
    export D4C_CONSOLE_LEVEL="${D4C_CONSOLE_LEVEL:-WARNING}"
fi

DDP_NPROC="${DDP_NPROC:-2}"
TORCHRUN_BIN=""

resolve_torchrun() {
    if command -v torchrun >/dev/null 2>&1; then
        TORCHRUN_BIN="torchrun"
        return 0
    fi
    if python -c "import torch" >/dev/null 2>&1; then
        TORCHRUN_BIN="python -m torch.distributed.run"
        echo "提示: 未找到 torchrun，自动回退到: ${TORCHRUN_BIN}"
        return 0
    fi
    echo "错误: 当前环境既没有 torchrun，也无法 'python -c \"import torch\"'。"
    echo "      请先激活安装了 PyTorch 的环境后重试。"
    exit 1
}

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
    elif [ "$i" = "--ddp-nproc" ]; then prev="--ddp-nproc"
    elif [ "$prev" = "--ddp-nproc" ]; then DDP_NPROC="$i"; prev=""
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --all | --task N [--from N] [--skip N,M,...] [--batch-size N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  多卡: CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 $0 --task 1"
    echo "  单卡: DDP_NPROC=1 $0 --task 1"
    echo "  （不推荐）单进程+DataParallel: cd code && python generate_counterfactual.py --gpus 0,1 --task 1"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ "$MODE" = "all" ]; then
        LOGFILE="$LOG_DIR/step4_daemon_${_RUN_TS}.log"
        NOHUP_OUT="$LOGFILE"
    else
        LOGFILE="$(d4c_step4_logfile "$TASK_ID")"
        mkdir -p "$(dirname "$LOGFILE")"
        NOHUP_OUT="$(dirname "$LOGFILE")/nohup.log"
    fi
    mkdir -p "$(dirname "$LOGFILE")"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    if [ "$MODE" = "all" ]; then
        ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
        _EX="$(d4c_step4_logfile 1)"
        echo "已在后台启动 Step 4（torchrun DDP nproc=${DDP_NPROC}）全部 8 任务；终端汇总: $ABS_LOG"
        echo "  每任务 Python 日志示例（Task 1）: $_EX"
        echo "查看汇总: tail -f $ABS_LOG"
    else
        ABS_TRAIN="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
        ABS_NOHUP="$(readlink -f "$NOHUP_OUT" 2>/dev/null || echo "$NOHUP_OUT")"
        echo "已在后台启动 Step 4（torchrun DDP nproc=${DDP_NPROC}）"
        echo "  Python 日志 (--log_file): $ABS_TRAIN"
        echo "  nohup 终端输出: $ABS_NOHUP"
        echo "查看日志: tail -f $ABS_TRAIN"
    fi
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$NOHUP_OUT" 2>&1 &
    echo "PID: $!"
    exit 0
fi

resolve_torchrun

if [ -n "$GPUS" ]; then
    echo "提示: 本脚本已用 torchrun DDP，--gpus 不会传给 Python（DDP 以 LOCAL_RANK 绑卡）。请设置 CUDA_VISIBLE_DEVICES。"
fi

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    LOGFILE="$(d4c_step4_logfile "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
fi

run_one_task_step4() {
    local idx=$1
    local py_log=$2
    mkdir -p "$(dirname "$py_log")"
    # 勿对 train.log 再 tee：PerfMonitor 已通过 append_log_dual 写入 --log_file
    ${TORCHRUN_BIN} --standalone --nproc_per_node="${DDP_NPROC}" generate_counterfactual.py \
        --task "$idx" --log_file "$py_log" $BATCH_SIZE $NUM_PROC || { echo "Task $idx 失败"; exit 1; }
}

if [ "$MODE" = "all" ]; then
    [ "$FROM_TASK" -gt 1 ] && echo "从 Task $FROM_TASK 起跑（跳过 1-$((FROM_TASK - 1))）"
    [ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
    _EX="$(d4c_step4_logfile 1)"
    echo "启动 Step 4 全部 8 个任务（torchrun DDP nproc=${DDP_NPROC}）；每任务主日志 get_log_task_dir(task)/runs/<时间戳>/train.log"
    echo "  示例（Task 1）: $_EX"
    for i in 1 2 3 4 5 6 7 8; do
        PY_LOG="$(d4c_step4_logfile "$i")"
        mkdir -p "$(dirname "$PY_LOG")"
        [ "$i" -lt "$FROM_TASK" ] && { echo "========== 跳过 Task $i (--from $FROM_TASK) =========="; continue; }
        should_skip $i && { echo "========== 跳过 Task $i =========="; continue; }
        echo "---------- Task $i 日志: $PY_LOG ----------"
        run_one_task_step4 "$i" "$PY_LOG"
    done
else
    echo "启动 Step 4 Task $TASK_ID（torchrun DDP nproc=${DDP_NPROC}），日志: $LOGFILE"
    run_one_task_step4 "$TASK_ID" "$LOGFILE"
fi
echo "========== Step 4 完成 =========="
