#!/bin/bash
#
# -----------------------------------------------------------------------------
# 兼容入口（推荐后续收敛至）：bash scripts/train_ddp.sh — 见 docs/D4C_RUNTIME_SPEC.md
# -----------------------------------------------------------------------------
#
# run_step4_optimized.sh — Step 4 正式入口：生成反事实（torchrun + DDP）
#
# 须与 Step 3 的 checkpoint 命名空间一致：
#   checkpoints/<task>/step3_optimized/<step3_opt_id>/model.pth
#
# 必填：--step3-subdir <NAME>（与上一步 SUBDIR 目录名一致，如 step3_opt_20260324_1400）
#
# 用法: bash sh/run_step4_optimized.sh --task N --step3-subdir <NAME>
#       bash sh/run_step4_optimized.sh --all --step3-subdir <NAME>
#       bash sh/run_step4_optimized.sh --all --from 4 --step3-subdir <NAME>
#
# 默认：D4C_CHECKPOINT_GROUP=step3_optimized，D4C_CHECKPOINT_SUBDIR=<NAME>
# 主日志：未设置 D4C_LOG_GROUP / D4C_LOG_SUBDIR 时 D4C_LOG_STEP=step4_optimized
#
set -euo pipefail
# 可选：与 Step 3/5 一致，用一条 runtime 预设收敛 DataLoader / num_proc（详见 code/config.py RUNTIME_PRESETS）
#   单卡：export D4C_RUNTIME_PRESET=gpu01_single_12c
#   双卡：export D4C_RUNTIME_PRESET=gpu01_ddp2_12c
#   推荐：OMP_NUM_THREADS / MKL_NUM_THREADS / TOKENIZERS_PARALLELISM（见 run_step3_optimized.sh 注释）
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"

new_args=()
STEP3_SUBDIR=""
prev=""
for i in "$@"; do
    if [ "$i" = "--gpus" ] || [[ "$i" == --gpus=* ]]; then
        echo "错误: --gpus has been removed. 请使用 CUDA_VISIBLE_DEVICES 与 DDP_NPROC / --ddp-nproc。" >&2
        exit 2
    fi
    if [ "$prev" = "--step3-subdir" ]; then
        STEP3_SUBDIR="$i"
        prev=""
        continue
    fi
    if [ "$i" = "--step3-subdir" ]; then
        prev="--step3-subdir"
        continue
    fi
    new_args+=("$i")
done
set -- "${new_args[@]}"

# nohup 子进程可能未带上 --step3-subdir 参数，继承父进程已 export 的 D4C_CHECKPOINT_SUBDIR
if [ -z "$STEP3_SUBDIR" ] && [ -n "${D4C_CHECKPOINT_SUBDIR:-}" ]; then
    STEP3_SUBDIR="$D4C_CHECKPOINT_SUBDIR"
fi

if [ -z "$STEP3_SUBDIR" ]; then
    echo "错误: 须指定 --step3-subdir <NAME>，与 checkpoints/<task>/step3_optimized/<NAME>/ 一致。"
    echo "  例: bash sh/run_step4_optimized.sh --task 2 --step3-subdir step3_opt_20260324_1400"
    exit 1
fi

export D4C_CHECKPOINT_GROUP="${D4C_CHECKPOINT_GROUP:-step3_optimized}"
export D4C_CHECKPOINT_SUBDIR="$STEP3_SUBDIR"

if [ -z "${D4C_LOG_GROUP:-}" ] && [ -z "${D4C_LOG_SUBDIR:-}" ]; then
    export D4C_LOG_STEP="${D4C_LOG_STEP-step4_optimized}"
fi

cd "$CODE_DIR"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

_extra=()
if [[ "$*" != *"--batch-size"* ]]; then
    if [ -n "${D4C_OPT_BATCH_SIZE:-}" ]; then
        _extra+=(--batch-size "$D4C_OPT_BATCH_SIZE")
    elif python -c "from config import training_preset_is_per_task; import sys; sys.exit(0 if training_preset_is_per_task() else 1)"; then
        :
    else
        _default_bs="$(python -c "from config import get_train_batch_size; print(get_train_batch_size())")"
        _extra+=(--batch-size "$_default_bs")
    fi
fi

echo "[run_step4_optimized] D4C_CHECKPOINT_GROUP=$D4C_CHECKPOINT_GROUP D4C_CHECKPOINT_SUBDIR=$D4C_CHECKPOINT_SUBDIR D4C_LOG_STEP=${D4C_LOG_STEP:-}"

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
    elif [ "$i" = "--batch-size" ]; then prev="--batch-size"
    elif [ "$prev" = "--batch-size" ]; then BATCH_SIZE="--batch-size $i"; prev=""
    elif [ "$i" = "--num-proc" ]; then prev="--num-proc"
    elif [ "$prev" = "--num-proc" ]; then NUM_PROC="--num-proc $i"; prev=""
    elif [ "$i" = "--ddp-nproc" ]; then prev="--ddp-nproc"
    elif [ "$prev" = "--ddp-nproc" ]; then DDP_NPROC="$i"; prev=""
    elif [ "$i" = "--gpus" ] || [[ "$i" == --gpus=* ]]; then
        echo "错误: --gpus has been removed. 请使用 CUDA_VISIBLE_DEVICES 与 DDP_NPROC / --ddp-nproc。" >&2
        exit 2
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --step3-subdir NAME --all | --task N [--from N] [--skip N,M,...] [--batch-size N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  多卡: CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 $0 --step3-subdir NAME --task 1"
    echo "  单卡: DDP_NPROC=1 $0 --step3-subdir NAME --task 1"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ "$MODE" = "all" ]; then
        LOGFILE="$LOG_DIR/step4_optimized_daemon_${_RUN_TS}.log"
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

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    LOGFILE="$(d4c_step4_logfile "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
fi

run_one_task_step4() {
    local idx=$1
    local py_log=$2
    mkdir -p "$(dirname "$py_log")"
    if [ "${#_extra[@]}" -gt 0 ]; then
        ${TORCHRUN_BIN} --standalone --nproc_per_node="${DDP_NPROC}" generate_counterfactual.py \
            --task "$idx" --log_file "$py_log" $BATCH_SIZE $NUM_PROC "${_extra[@]}" || { echo "Task $idx 失败"; exit 1; }
    else
        ${TORCHRUN_BIN} --standalone --nproc_per_node="${DDP_NPROC}" generate_counterfactual.py \
            --task "$idx" --log_file "$py_log" $BATCH_SIZE $NUM_PROC || { echo "Task $idx 失败"; exit 1; }
    fi
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
        should_skip "$i" && { echo "========== 跳过 Task $i =========="; continue; }
        echo "---------- Task $i 日志: $PY_LOG ----------"
        run_one_task_step4 "$i" "$PY_LOG"
    done
else
    echo "启动 Step 4 Task $TASK_ID（torchrun DDP nproc=${DDP_NPROC}），日志: $LOGFILE"
    run_one_task_step4 "$TASK_ID" "$LOGFILE"
fi
echo "========== Step 4 完成 =========="
