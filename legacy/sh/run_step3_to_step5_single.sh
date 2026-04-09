#!/bin/bash
# -----------------------------------------------------------------------------
# LEGACY — 单任务 Shell 串联。主线：python code/d4c.py pipeline … 或 bash scripts/train_ddp.sh --pipeline 3,4,5
# 见 docs/legacy_batch_shell.md
# -----------------------------------------------------------------------------
# Step 3-5 单个任务：域对抗预训练 → 生成反事实 → 主训练（顺序执行）
# 用法: bash run_step3_to_step5_single.sh --task N [--from 3|4|5] [--eval-only|--train-only] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]
#
#   --eval-only  Step 3 只跑 step3 runner eval；Step 5 只跑 step5 runner 评估（均不重训），见 scripts/entrypoints/step3.sh / step5.sh
#   --train-only Step 3 / Step 5 跳过训练后的收尾 eval（与 --eval-only 互斥）
#
# ========== DDP（Step 3、Step 4、Step 5）==========
# Step 3 / Step 4 / Step 5 均为 torchrun + DDP（见 scripts/entrypoints/step3.sh、step4.sh、step5.sh）。进程数：
#   - 环境变量 DDP_NPROC，或参数 --ddp-nproc K（默认一般为 2）
#   - 全局 batch 须能被进程数整除；单卡：DDP_NPROC=1 或 --ddp-nproc 1
# 多卡请用 CUDA_VISIBLE_DEVICES 与 DDP_NPROC / --ddp-nproc 对齐（Step 3/4/5 均为 torchrun DDP）。
# 示例:
#   DDP_NPROC=1 bash run_step3_to_step5_single.sh --task 2
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3_to_step5_single.sh --task 2 --batch-size 1024
# 完整说明见 scripts/entrypoints/step3.sh、step4.sh、step5.sh 文件头。
#
# 其它示例:
#       bash run_step3_to_step5_single.sh --task 2 --from 4   # 跳过 Step 3，从 Step 4 续跑
#       CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3_to_step5_single.sh --task 5
#       bash run_step3_to_step5_single.sh --task 2 --batch-size 64 --epochs 30
#       bash run_step3_to_step5_single.sh --task 2 --daemon              # 后台跑（shell 汇总写入 runs/taskT/…/meta/shell_logs/）
#
#   --daemon / --bg   后台运行：stdout/stderr 写入 runs/taskN/vN/meta/shell_logs/step3_to_5_taskN_*.log

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

TASK_ID=""
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
    elif [ "$arg" = "--gpus" ] || [[ "$arg" == --gpus=* ]]; then
        echo "错误: --gpus has been removed. 请使用 CUDA_VISIBLE_DEVICES 与 DDP_NPROC / --ddp-nproc。" >&2
        exit 2
    elif [ "$arg" = "--daemon" ] || [ "$arg" = "--bg" ]; then
        DAEMON=1
    elif [ "$prev" = "--task" ] && [[ "$arg" =~ ^[1-8]$ ]]; then
        TASK_ID=$arg
    elif [ "$prev" = "--from" ]; then
        [[ "$arg" =~ ^[345]$ ]] || { echo "错误: --from 须为 3、4 或 5，收到: $arg"; exit 1; }
        FROM_STEP=$arg
    elif [ "$prev" = "--batch-size" ]; then
        BATCH_SIZE="--batch-size $arg"
    elif [ "$prev" = "--epochs" ]; then
        EPOCHS="--epochs $arg"
    elif [ "$prev" = "--num-proc" ]; then
        NUM_PROC="--num-proc $arg"
    elif [ "$prev" = "--ddp-nproc" ]; then
        DDP_EXTRA="--ddp-nproc $arg"
    elif [ "$prev" = "--iter" ]; then
        export D4C_ITER="$arg"
    fi
    prev="$arg"
done

export D4C_ITER="${D4C_ITER:-v1}"

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

[ -z "$TASK_ID" ] && {
    echo "用法: $0 --task N [--from 3|4|5] [--eval-only|--train-only] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  N 为 1-8；--eval-only：Step 3 / Step 5 只 eval（须已有 checkpoint）；--train-only：跳过 Step3/5 训练后收尾 eval（互斥）；DDP 见 scripts/entrypoints/step3.sh / step5.sh"
    echo "  --daemon / --bg：后台运行，shell 汇总写入 runs/taskN/\${D4C_ITER}/meta/shell_logs/；详见文件头"
    exit 1
}

STEP3_SUB=""

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$(d4c_shell_logs_task "$D4C_ROOT" "$TASK_ID")/step3_to_5_task${TASK_ID}_$(date +%Y%m%d_%H%M).log"
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
    LOGFILE="$(d4c_shell_logs_task "$D4C_ROOT" "$TASK_ID")/step3_to_5_task${TASK_ID}_$(date +%Y%m%d_%H%M).log"
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
    bash "$MAIN_SH/step3.sh" --task $TASK_ID --iter "$D4C_ITER" $STEP3_EVAL $STEP3_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee "$LOGFILE"
    STEP3_SUB="$(d4c_latest_step3_subdir "$TASK_ID")" || {
        echo "错误: 未找到 runs/task${TASK_ID}/${D4C_ITER}/train/step3/ 下有效子目录（Step 3 未完成）" | tee -a "$LOGFILE"
        exit 1
    }
else
    echo "========== 跳过 Task $TASK_ID Step 3 (--from $FROM_STEP) ==========" | tee "$LOGFILE"
fi

if [ "$FROM_STEP" -le 4 ]; then
    echo ""
    echo "========== Task $TASK_ID: Step 4 生成反事实 =========="
    if [ -z "$STEP3_SUB" ]; then
        STEP3_SUB="$(d4c_latest_step3_subdir "$TASK_ID")" || {
            echo "错误: 未找到 runs/task${TASK_ID}/${D4C_ITER}/train/step3/ 下有效子目录（须先完成 Step 3）" | tee -a "$LOGFILE"
            exit 1
        }
    fi
    bash "$MAIN_SH/step4.sh" --iter "$D4C_ITER" --from-run "$STEP3_SUB" --task $TASK_ID --eval-profile "$PIPELINE_EVAL_PROFILE" $BATCH_SIZE $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"
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
    if [ -z "$STEP3_SUB" ]; then
        STEP3_SUB="$(d4c_latest_step3_subdir "$TASK_ID")" || {
            echo "错误: 未找到 runs/task${TASK_ID}/${D4C_ITER}/train/step3/ 下有效子目录（须先完成 Step 3）" | tee -a "$LOGFILE"
            exit 1
        }
    fi
    S5_ARG=()
    if [ -n "$EVAL_ONLY" ]; then
        _s5="$(d4c_latest_step5_run "$TASK_ID")" || {
            echo "错误: 未找到 runs/task${TASK_ID}/${D4C_ITER}/train/step5/ 下有效子目录（--eval-only 须已有 Step 5）" | tee -a "$LOGFILE"
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
        _s4="$(d4c_latest_step4_subdir "$TASK_ID")" || {
            echo "错误: 未找到 train/step4 下有效子目录（Step 4 未完成）" | tee -a "$LOGFILE"
            exit 1
        }
        S5_S4RUN=(--step4-run "$_s4")
    fi
    bash "$MAIN_SH/step5.sh" --iter "$D4C_ITER" --task "$TASK_ID" --from-run "$STEP3_SUB" "${S5_ARG[@]}" "${S5_S4RUN[@]}" "${S5_EP[@]}" $STEP5_EVAL $STEP5_TRAIN $BATCH_SIZE $EPOCHS $NUM_PROC $DDP_EXTRA 2>&1 | tee -a "$LOGFILE"
fi

echo ""
echo "========== Task $TASK_ID Step 3-5 完成 =========="
