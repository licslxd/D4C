#!/bin/bash
#
# -----------------------------------------------------------------------------
# MAINLINE: python code/d4c.py step4 …；本脚本仅编排 / GPU / nohup / 批量。
# -----------------------------------------------------------------------------
#
# scripts/entrypoints/step4.sh — Step 4：生成反事实（torchrun + DDP）
#
# 须已完成 Step 3：权重在 runs/task{T}/vN/train/step3/<run>/；Step4 产物与主日志在 runs/task{T}/vN/train/step4/<step4-run>/logs/step4.log
#
# 必填：--from-run <run>（Step 3 的 run 目录名，与 d4c.py 一致）
# Step4 主线合同：`d4c.py step4` **须** `--eval-profile`（须显式传入，勿依赖 shell 隐式默认）。
# 推理 batch 仅来自该 profile 的 eval_batch_size。
#
# 用法: bash scripts/entrypoints/step4.sh --task N --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu
#
# 可选：--iter vN 或 export D4C_ITER（默认 v1）
#
# torchrun 子进程环境由 d4c 清洗：export TRAIN_* 不会稳定传入子进程（见 docs/D4C_Scripts_and_Runtime_Guide.md）。
#
set -euo pipefail
# 可选：与 Step 3/5 一致，用一条 hardware 预设收敛 DataLoader / num_proc（详见 code/config.py HARDWARE_PRESETS）
#   单卡：export D4C_HARDWARE_PRESET=gpu01_single_12c
#   双卡：export D4C_HARDWARE_PRESET=gpu01_ddp2_12c
#   推荐：OMP_NUM_THREADS / MKL_NUM_THREADS / TOKENIZERS_PARALLELISM（见 step3 编排脚本注释）
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/../.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
export D4C_ROOT
# shellcheck source=../lib/common_logs.sh
source "$SH_DIR/../lib/common_logs.sh"

new_args=()
FROM_RUN_NAME=""
STEP4_RUN_NAME=""
ITER_CLI=""
EVAL_PROFILE_NAME=""
PRESET_NAME="step3"
prev=""
for i in "$@"; do
    if [ "$i" = "--gpus" ] || [[ "$i" == --gpus=* ]]; then
        echo "错误: --gpus has been removed. 请使用 CUDA_VISIBLE_DEVICES 与 DDP_NPROC / --ddp-nproc。" >&2
        exit 2
    fi
    if [ "$prev" = "--from-run" ]; then
        FROM_RUN_NAME="$i"
        prev=""
        continue
    fi
    if [ "$i" = "--from-run" ]; then
        prev="--from-run"
        continue
    fi
    if [ "$prev" = "--step4-run" ]; then
        STEP4_RUN_NAME="$i"
        prev=""
        continue
    fi
    if [ "$i" = "--step4-run" ]; then
        prev="--step4-run"
        continue
    fi
    if [ "$prev" = "--iter" ]; then
        ITER_CLI="$i"
        prev=""
        continue
    fi
    if [ "$i" = "--iter" ]; then
        prev="--iter"
        continue
    fi
    if [ "$prev" = "--eval-profile" ]; then
        EVAL_PROFILE_NAME="$i"
        prev=""
        continue
    fi
    if [ "$i" = "--eval-profile" ]; then
        prev="--eval-profile"
        continue
    fi
    if [ "$prev" = "--preset" ]; then
        PRESET_NAME="$i"
        prev=""
        continue
    fi
    if [ "$i" = "--preset" ]; then
        prev="--preset"
        continue
    fi
    new_args+=("$i")
done
set -- "${new_args[@]}"

export D4C_ITER="${ITER_CLI:-${D4C_ITER:-v1}}"

if [ -z "$FROM_RUN_NAME" ]; then
    echo "错误: 须指定 --from-run <run>（Step 3 的 run 目录名，如 1）。"
    echo "  例: bash scripts/entrypoints/step4.sh --task 2 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu"
    exit 1
fi

if [ -z "$EVAL_PROFILE_NAME" ]; then
    echo "错误: 须显式传入 --eval-profile <stem>（与 d4c.py step4 合同一致）。示例: --eval-profile eval_fast_single_gpu" >&2
    exit 2
fi

cd "$D4C_ROOT"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"

echo "[step4] D4C_ITER=$D4C_ITER from-run=$FROM_RUN_NAME eval-profile=$EVAL_PROFILE_NAME step4-run=${STEP4_RUN_NAME:-auto} preset=$PRESET_NAME"

_RUN_TS="$(date +%Y%m%d_%H%M)"

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
    export D4C_CONSOLE_LEVEL="${D4C_CONSOLE_LEVEL:-WARNING}"
fi

DDP_NPROC="${DDP_NPROC:-}"
_ddp_ws_args_s4=()
if [ -n "${DDP_NPROC:-}" ]; then
    _ddp_ws_args_s4=(--ddp-world-size "$DDP_NPROC")
fi

MODE=""
TASK_ID=""
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
    echo "用法: $0 --from-run <run> --eval-profile STEM [--iter vN] [--preset step3] --all | --task N [--from N] [--skip N,M,...] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  多卡: CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 $0 --iter v1 --from-run 1 --task 1"
    echo "  单卡: DDP_NPROC=1 $0 --iter v1 --from-run 1 --task 1"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ "$MODE" = "all" ]; then
        _GSL="$(d4c_shell_logs_global "$D4C_ROOT")"
        LOGFILE="$_GSL/step4_optimized_daemon_${_RUN_TS}.log"
        NOHUP_OUT="$LOGFILE"
    else
        _S4META="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/meta/shell_logs"
        mkdir -p "$_S4META"
        if [ -n "${STEP4_RUN_NAME:-}" ]; then
            LOGFILE="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/train/step4/${STEP4_RUN_NAME}/logs/step4.log"
        else
            LOGFILE="$_S4META/step4_task${TASK_ID}_from_${FROM_RUN_NAME}_${_RUN_TS}.log"
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
        echo "已在后台启动 Step 4（d4c.py step4, DDP nproc=${DDP_NPROC}）全部 8 任务；终端汇总: $ABS_LOG"
        echo "  每任务主日志: runs/task{T}/${D4C_ITER}/train/step4/<step4-run>/logs/step4.log（未传 --step4-run 时见 d4c 解析与 meta/shell_logs）"
        echo "查看汇总: tail -f $ABS_LOG"
    else
        ABS_TRAIN="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
        ABS_NOHUP="$(readlink -f "$NOHUP_OUT" 2>/dev/null || echo "$NOHUP_OUT")"
        echo "已在后台启动 Step 4（d4c.py step4, DDP nproc=${DDP_NPROC}）"
        echo "  Python 日志 (--log_file): $ABS_TRAIN"
        echo "  nohup 终端输出: $ABS_NOHUP"
        echo "查看日志: tail -f $ABS_TRAIN"
    fi
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$NOHUP_OUT" 2>&1 &
    echo "PID: $!"
    exit 0
fi

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    _S4META="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/meta/shell_logs"
    mkdir -p "$_S4META"
    if [ -n "${STEP4_RUN_NAME:-}" ]; then
        LOGFILE="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/train/step4/${STEP4_RUN_NAME}/logs/step4.log"
    else
        LOGFILE="$_S4META/step4_task${TASK_ID}_from_${FROM_RUN_NAME}_${_RUN_TS}.log"
    fi
    mkdir -p "$(dirname "$LOGFILE")"
fi

_s4run=()
[ -n "${STEP4_RUN_NAME:-}" ] && _s4run=(--step4-run "$STEP4_RUN_NAME")

run_one_task_step4() {
    local idx=$1
    local PRESET="${PRESET_NAME}"
    # shellcheck disable=SC2086
    python "$D4C_ROOT/code/d4c.py" step4 --task "$idx" --preset "$PRESET" \
        --iter "$D4C_ITER" \
        --from-run "$FROM_RUN_NAME" \
        --eval-profile "$EVAL_PROFILE_NAME" \
        "${_s4run[@]}" \
        "${_ddp_ws_args_s4[@]}" \
        $NUM_PROC || { echo "Task $idx 失败"; exit 1; }
}

if [ "$MODE" = "all" ]; then
    [ "$FROM_TASK" -gt 1 ] && echo "从 Task $FROM_TASK 起跑（跳过 1-$((FROM_TASK - 1))）"
    [ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
    echo "启动 Step 4 全部 8 个任务（d4c.py step4, DDP nproc=${DDP_NPROC}）；主日志: runs/task{T}/${D4C_ITER}/train/step4/<step4-run>/logs/step4.log"
    for i in 1 2 3 4 5 6 7 8; do
        [ "$i" -lt "$FROM_TASK" ] && { echo "========== 跳过 Task $i (--from $FROM_TASK) =========="; continue; }
        should_skip "$i" && { echo "========== 跳过 Task $i =========="; continue; }
        echo "---------- Task $i: runs/task${i}/${D4C_ITER}/train/step4/<step4-run>/logs/step4.log ----------"
        run_one_task_step4 "$i"
    done
else
    echo "启动 Step 4 Task $TASK_ID（d4c.py step4, DDP nproc=${DDP_NPROC}），主日志: $LOGFILE"
    run_one_task_step4 "$TASK_ID"
fi
echo "========== Step 4 完成 =========="
