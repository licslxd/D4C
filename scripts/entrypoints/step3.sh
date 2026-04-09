#!/bin/bash
#
# -----------------------------------------------------------------------------
# MAINLINE: python code/d4c.py step3 …（项目根）；本脚本仅做 GPU / nohup / 批量循环。
# 批量编排: bash scripts/entrypoints/train_ddp.sh --step 3 …
# -----------------------------------------------------------------------------
#
# scripts/entrypoints/step3.sh — Step 3：域对抗预训练（torchrun + DDP）
#
# 产物：runs/task{T}/vN/train/step3/<run>/（manifest、model、logs/train.log、logs/eval.log；auto 默认 1、2、…）
# 须指定迭代：--iter vN 或 export D4C_ITER=v1（默认 v1）
#
# 命令行要点：
#   --task / --all / --eval-only / --train-only / --preset（默认 step3）/ --batch-size / --epochs / --ddp-nproc / --iter / --run-id / --daemon
# 训练语义只来自 presets/training/<preset>.yaml 与 d4c CLI；勿在 shell 中 export TRAIN_* 或 D4C_TRAIN_PRESET。
# 可选 shell 级：CUDA_VISIBLE_DEVICES、OMP_NUM_THREADS、MKL_NUM_THREADS、TOKENIZERS_PARALLELISM、DDP_NPROC。
#
# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------
#   bash scripts/entrypoints/step3.sh --task 4 --iter v1
#   DDP_NPROC=1 bash scripts/entrypoints/step3.sh --task 2 --iter v1
#
# Step 4：bash scripts/entrypoints/step4.sh … --eval-profile <stem>
#
set -euo pipefail
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/../.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
export D4C_ROOT
cd "$D4C_ROOT"
# shellcheck source=../lib/common_paths.sh
source "$SH_DIR/../lib/common_paths.sh"
# shellcheck source=../lib/common_logs.sh
source "$SH_DIR/../lib/common_logs.sh"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"

# 与 d4c.py step3 解析的 log_dir 一致（便于 tail/nohup 提示）
d4c_step3_logfile() {
    local tid=$1
    d4c_predict_step3_train_log "$tid"
}

d4c_step3_eval_logfile() {
    local tid=$1
    d4c_predict_step3_eval_log "$tid"
}

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
    export D4C_CONSOLE_LEVEL="${D4C_CONSOLE_LEVEL:-WARNING}"
fi

# 不设默认：仅当用户 export DDP_NPROC 或通过 --ddp-nproc 指定时才向 d4c 传 --ddp-world-size
DDP_NPROC="${DDP_NPROC:-}"

BATCH_SIZE=""
EPOCHS=""
NUM_PROC=""
_ddp_ws_args=()

MODE=""
TASK_ID=""
SKIP_LIST=""
FROM_TASK=1
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
PRESET_NAME="step3"
prev=""
for i in "$@"; do
    if [ "$i" = "--all" ]; then MODE="all"
    elif [ "$i" = "--eval-only" ]; then EVAL_ONLY=1
    elif [ "$i" = "--train-only" ]; then TRAIN_ONLY=1
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
    elif [ "$i" = "--epochs" ]; then prev="--epochs"
    elif [ "$prev" = "--epochs" ]; then EPOCHS="--epochs $i"; prev=""
    elif [ "$i" = "--num-proc" ]; then prev="--num-proc"
    elif [ "$prev" = "--num-proc" ]; then NUM_PROC="--num-proc $i"; prev=""
    elif [ "$i" = "--ddp-nproc" ]; then prev="--ddp-nproc"
    elif [ "$prev" = "--ddp-nproc" ]; then DDP_NPROC="$i"; prev=""
    elif [ "$i" = "--gpus" ] || [[ "$i" == --gpus=* ]]; then
        echo "错误: --gpus has been removed. 请使用 CUDA_VISIBLE_DEVICES，并用 DDP_NPROC / --ddp-nproc 与 torchrun --nproc_per_node 对齐。" >&2
        echo "单卡 DDP smoke: DDP_NPROC=1 bash scripts/entrypoints/step3.sh --task 1" >&2
        exit 2
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    elif [ "$i" = "--iter" ]; then prev="--iter"
    elif [ "$prev" = "--iter" ]; then export D4C_ITER="$i"; prev=""
    elif [ "$i" = "--run-id" ]; then prev="--run-id"
    elif [ "$prev" = "--run-id" ]; then export D4C_RUN_ID="$i"; prev=""
    elif [ "$i" = "--preset" ]; then prev="--preset"
    elif [ "$prev" = "--preset" ]; then PRESET_NAME="$i"; prev=""
    fi
done

export D4C_ITER="${D4C_ITER:-v1}"

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

_RUN_TS="$(date +%Y%m%d_%H%M)"
_LOG_TS="$(date +%Y%m%d_%H%M%S)"

echo "[step3] D4C_ITER=$D4C_ITER D4C_RUN_ID=${D4C_RUN_ID:-auto} preset=$PRESET_NAME"
echo "[step3] 训练语义见 presets/training/${PRESET_NAME}.yaml（勿在 shell 中 export TRAIN_* / D4C_TRAIN_PRESET）"

if [ -n "${DDP_NPROC:-}" ]; then
    _ddp_ws_args=(--ddp-world-size "$DDP_NPROC")
fi

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --all | --task N [--iter vN] [--run-id <name>|auto] [--eval-only|--train-only] [--from N] [--skip N,M,...] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  --eval-only：只跑 eval，跳过 train（须已有训练产物）"
    echo "  --train-only：只 train，跳过训练后的 eval（与 --eval-only 互斥）"
    echo "  --daemon / --bg：后台运行；详见文件头"
    echo "  DDP：可选 DDP_NPROC 或 --ddp-nproc；省略时由 presets/runtime 决定 ddp_world_size"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ "$MODE" = "all" ]; then
        _GSL="$(d4c_shell_logs_global "$D4C_ROOT")"
        if [ -n "$EVAL_ONLY" ]; then
            LOGFILE="$_GSL/step3_optimized_eval_daemon_${_RUN_TS}.log"
        else
            LOGFILE="$_GSL/step3_optimized_daemon_${_RUN_TS}.log"
        fi
        NOHUP_OUT="$LOGFILE"
    else
        if [ -n "$EVAL_ONLY" ]; then
            LOGFILE="$(d4c_step3_eval_logfile "$TASK_ID")"
        else
            LOGFILE="$(d4c_step3_logfile "$TASK_ID")"
        fi
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
        if [ -n "$EVAL_ONLY" ]; then
            _EX="$(d4c_step3_eval_logfile 1)"
            echo "已在后台启动 Step 3（DDP）全部 8 任务【仅 eval】；终端汇总: $ABS_LOG"
        elif [ -n "$TRAIN_ONLY" ]; then
            _EX="$(d4c_step3_logfile 1)"
            echo "已在后台启动 Step 3（DDP）全部 8 任务【仅 train】；终端汇总: $ABS_LOG"
        else
            _EX="$(d4c_step3_logfile 1)"
            echo "已在后台启动 Step 3（DDP）全部 8 任务；终端汇总: $ABS_LOG"
        fi
        echo "  每任务 Python 日志示例（Task 1）: $_EX"
        echo "查看汇总: tail -f $ABS_LOG"
    else
        ABS_PY="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
        ABS_NOHUP="$(readlink -f "$NOHUP_OUT" 2>/dev/null || echo "$NOHUP_OUT")"
        echo "已在后台启动 Step 3（DDP）"
        echo "  Python 日志 (--log_file): $ABS_PY"
        echo "  nohup 终端输出: $ABS_NOHUP"
        if [ -n "$EVAL_ONLY" ]; then
            echo "查看 eval 日志: tail -f $ABS_PY"
        else
            echo "查看训练日志: tail -f $ABS_PY"
        fi
    fi
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$NOHUP_OUT" 2>&1 &
    echo "PID: $!"
    exit 0
fi

run_one_task() {
    local idx=$1
    [[ "$idx" =~ ^[1-8]$ ]] || { echo "无效任务 $idx"; return 1; }
    local PRESET="${PRESET_NAME}"
    local _eo=() _to=()
    [ -n "$EVAL_ONLY" ] && _eo=(--eval-only)
    [ -n "$TRAIN_ONLY" ] && _to=(--train-only)
    _ddp_msg="${DDP_NPROC:-<preset/runtime>}"
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Step 3 Task $idx 仅 eval（d4c.py, DDP=$_ddp_msg）=========="
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "========== Step 3 Task $idx 仅 train（d4c.py, DDP=$_ddp_msg）=========="
    else
        echo "========== Step 3 Task $idx（d4c.py, DDP=$_ddp_msg）=========="
    fi
    _rid=()
    [ -n "${D4C_RUN_ID:-}" ] && _rid=(--run-id "$D4C_RUN_ID")
    # shellcheck disable=SC2086
    python "$D4C_ROOT/code/d4c.py" step3 --task "$idx" --preset "$PRESET" \
        --iter "$D4C_ITER" \
        "${_rid[@]}" \
        "${_ddp_ws_args[@]}" \
        $BATCH_SIZE $EPOCHS $NUM_PROC \
        "${_eo[@]}" "${_to[@]}"
}

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    if [ -n "$EVAL_ONLY" ]; then
        LOGFILE="$(d4c_step3_eval_logfile "$TASK_ID")"
    else
        LOGFILE="$(d4c_step3_logfile "$TASK_ID")"
    fi
    mkdir -p "$(dirname "$LOGFILE")"
fi
if [ "$MODE" = "all" ]; then
    # 与 Python --log_file（train.log / eval.log）分离：shell/tee 只写汇总文件，避免与 FileHandler 多写者竞争
    _GSL_ALL="$(d4c_shell_logs_global "$D4C_ROOT")"
    STEP3_ALL_SHELL_LOG="${D4C_STEP3_ALL_SHELL_LOG:-$_GSL_ALL/step3_optimized_all_${_LOG_TS}.log}"
    mkdir -p "$(dirname "$STEP3_ALL_SHELL_LOG")"
    [ "$FROM_TASK" -gt 1 ] && echo "从 Task $FROM_TASK 起跑（跳过 1-$((FROM_TASK - 1))）"
    [ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 3（DDP）全部 8 个任务【仅 eval】；每任务 Python 日志: runs/task{T}/$D4C_ITER/train/step3/<run>/logs/eval.log"
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "启动 Step 3（DDP）全部 8 个任务【仅 train】；每任务 Python 日志: runs/task{T}/$D4C_ITER/train/step3/<run>/logs/train.log"
    else
        echo "启动 Step 3（DDP）全部 8 个任务；每任务 Python 日志: runs/.../logs/train.log（训练）+ 同目录 eval.log（收尾 eval）"
    fi
    if [ -n "$EVAL_ONLY" ]; then
        _EX="$(d4c_step3_eval_logfile 1)"
    else
        _EX="$(d4c_step3_logfile 1)"
    fi
    echo "  示例（Task 1）: $_EX"
    echo "  Shell 提示/跳过行（tee，非 Python train.log/eval.log）: $STEP3_ALL_SHELL_LOG"
    for i in 1 2 3 4 5 6 7 8; do
        LOGFILE="$(d4c_step3_logfile "$i")"
        mkdir -p "$(dirname "$LOGFILE")"
        [ "$i" -lt "$FROM_TASK" ] && { echo "========== 跳过 Task $i (--from $FROM_TASK) ==========" | tee -a "$STEP3_ALL_SHELL_LOG"; continue; }
        should_skip "$i" && { echo "========== 跳过 Task $i ==========" | tee -a "$STEP3_ALL_SHELL_LOG"; continue; }
        echo "---------- Task $i 日志: $LOGFILE ----------" | tee -a "$STEP3_ALL_SHELL_LOG"
        run_one_task "$i" || { echo "Task $i 失败" | tee -a "$STEP3_ALL_SHELL_LOG"; exit 1; }
    done
else
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 3（DDP）Task $TASK_ID【仅 eval】，日志: $LOGFILE"
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "启动 Step 3（DDP）Task $TASK_ID【仅 train】，日志: $LOGFILE"
    else
        echo "启动 Step 3（DDP）Task $TASK_ID，日志: $LOGFILE"
    fi
    run_one_task "$TASK_ID"
fi
if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 3（仅 eval）完成 =========="
elif [ -n "$TRAIN_ONLY" ]; then
    echo "========== Step 3（仅 train）完成 =========="
else
    echo "========== Step 3 完成 =========="
fi
