#!/bin/bash
# Step 5：主训练与评估（DDP：torchrun + run-d4c.py）
# 仅支持「嵌套在 Step 3 目录下」布局（最终版），不再使用 checkpoints/<task>/step5/step5_<时间>/。
#
# 用法: bash run_step5.sh --task N --step3-subdir step3_YYYYMMDD_HHMM [选项…]
#       bash run_step5.sh --task N --step3-subdir … --nested-subdir step5_… --eval-only
#       bash run_step5.sh --task 2 --step3-subdir step3_… --batch-size 64 --epochs 30
#       bash run_step5.sh --task 2 --step3-subdir step3_… --train-only
#
# ========== DDP（本步唯一路径）==========
#   torchrun --standalone --nproc_per_node=$DDP_NPROC run-d4c.py …
# 进程数：环境变量 DDP_NPROC，或参数 --ddp-nproc K（默认 DDP_NPROC=2）
# 须与可见 GPU 数一致；全局 batch 须能被 DDP_NPROC 整除；单卡：DDP_NPROC=1
#
# 示例:
#   DDP_NPROC=1 bash run_step5.sh --task 2 --step3-subdir step3_20260323_1123
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step5.sh --task 4 --step3-subdir step3_20260323_1123 --daemon
#
# ---------------------------------------------------------------------------
# Checkpoint（仅嵌套）
# ---------------------------------------------------------------------------
# 目录：checkpoints/<task>/step3/<step3_id>/step5/step5_<分钟时间戳>/model.pth
#       （内层目录名可用 --nested-subdir 覆盖；训练时未指定则自动生成 step5_<时间>）
# 软链：内层目录内 factuals_counterfactuals.csv -> ../../factuals_counterfactuals.csv
# 环境：D4C_CHECKPOINT_GROUP=step3，D4C_CHECKPOINT_SUBDIR=<step3_id>/step5/<内层>
#
# --eval-only：须同时 --nested-subdir <已有内层名>，且该目录下已有 model.pth
# --train-only：训练后跳过收尾 valid 评估（与 --eval-only 互斥）
#
# 日志：默认 log/<task>/step5/runs/<秒级时间戳>/train.log（export D4C_LOG_GROUP=step5，与 checkpoint 的 GROUP=step3 解耦）
# 固定路径：export D4C_LOG_USE_TIMESTAMP=0 → .../runs/run/train.log
#
# ---------------------------------------------------------------------------

set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$CODE_DIR"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

d4c_run_log_dir() {
    python -c "from paths_config import get_log_task_dir; import os; print(get_log_task_dir($1))"
}
d4c_run_log_path() {
    python -c "from paths_config import get_log_task_dir; import os; print(os.path.join(get_log_task_dir($1), 'runs', 'run', 'train.log'))"
}
d4c_step5_logfile() {
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

BATCH_SIZE=""
EPOCHS=""
NUM_PROC=""

get_task_params() {
    case $1 in
        1) echo "AM_Electronics AM_CDs 5e-4 1 1e-3" ;;
        2) echo "AM_Movies AM_CDs 1e-3 0.1 1e-3" ;;
        3) echo "AM_CDs AM_Electronics 5e-4 0.5 1e-3" ;;
        4) echo "AM_Movies AM_Electronics 1e-3 0.5 1e-3" ;;
        5) echo "AM_CDs AM_Movies 1e-3 0.5 1e-3" ;;
        6) echo "AM_Electronics AM_Movies 1e-3 0.5 1e-3" ;;
        7) echo "Yelp TripAdvisor 1e-4 0.5 1e-3" ;;
        8) echo "TripAdvisor Yelp 5e-4 1 1e-3" ;;
        *) echo "" ;;
    esac
}

TASK_ID=""
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
STEP3_SUBDIR_FOR_NEST=""
NEST_SUB_NAME=""
prev=""
for i in "$@"; do
    if [ "$i" = "--all" ]; then
        echo "错误: Step 5 已取消 --all（仅嵌套模式）。请逐任务: --task N --step3-subdir step3_…"
        exit 1
    elif [ "$i" = "--eval-only" ]; then EVAL_ONLY=1
    elif [ "$i" = "--train-only" ]; then TRAIN_ONLY=1
    elif [ "$i" = "--task" ]; then prev="--task"
    elif [ "$prev" = "--task" ] && [[ "$i" =~ ^[1-8]$ ]]; then TASK_ID=$i; prev=""
    elif [ "$i" = "--batch-size" ]; then prev="--batch-size"
    elif [ "$prev" = "--batch-size" ]; then BATCH_SIZE="--batch-size $i"; prev=""
    elif [ "$i" = "--epochs" ]; then prev="--epochs"
    elif [ "$prev" = "--epochs" ]; then EPOCHS="--epochs $i"; prev=""
    elif [ "$i" = "--num-proc" ]; then prev="--num-proc"
    elif [ "$prev" = "--num-proc" ]; then NUM_PROC="--num-proc $i"; prev=""
    elif [ "$i" = "--ddp-nproc" ]; then prev="--ddp-nproc"
    elif [ "$prev" = "--ddp-nproc" ]; then DDP_NPROC="$i"; prev=""
    elif [ "$i" = "--step3-subdir" ]; then prev="--step3-subdir"
    elif [ "$prev" = "--step3-subdir" ]; then STEP3_SUBDIR_FOR_NEST="$i"; prev=""
    elif [ "$i" = "--nested-subdir" ]; then prev="--nested-subdir"
    elif [ "$prev" = "--nested-subdir" ]; then NEST_SUB_NAME="$i"; prev=""
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

if [ -n "${NEST_SUB_NAME:-}" ] && [ -z "${STEP3_SUBDIR_FOR_NEST:-}" ]; then
    echo "错误: --nested-subdir 须与 --step3-subdir 同时使用"
    exit 1
fi

if ! [[ "$TASK_ID" =~ ^[1-8]$ ]]; then
    echo "用法: $0 --task N --step3-subdir step3_YYYYMMDD_HHMM [选项]"
    echo "  必填: --task N（1–8）、--step3-subdir（与 checkpoints/<N>/step3/ 下目录名一致）"
    echo "  可选: --nested-subdir <内层名>（训练默认 step5_<分钟时间戳>；--eval-only 必填）"
    echo "  可选: --eval-only | --train-only（互斥）、--batch-size、--epochs、--num-proc、--ddp-nproc K、--daemon|--bg"
    echo "  布局: …/step3/<step3_id>/step5/<内层>/model.pth"
    exit 1
fi

if [ -z "${STEP3_SUBDIR_FOR_NEST:-}" ]; then
    echo "错误: Step 5 仅支持嵌套 checkpoint，须指定 --step3-subdir（见用法）"
    exit 1
fi

if [ -n "$EVAL_ONLY" ]; then
    if [ -z "${NEST_SUB_NAME:-}" ]; then
        echo "错误: --eval-only 须指定 --nested-subdir <已有内层目录名>，例如 step5_20260328_1123"
        exit 1
    fi
    _inner="$NEST_SUB_NAME"
elif [ -n "${NEST_SUB_NAME:-}" ]; then
    _inner="$NEST_SUB_NAME"
else
    _inner="step5_$(date +%Y%m%d_%H%M)"
fi

if [ -z "$_inner" ]; then
    echo "错误: --nested-subdir 不能为空"
    exit 1
fi

CKPT_BASE="$D4C_ROOT/checkpoints/$TASK_ID/step3/$STEP3_SUBDIR_FOR_NEST"
CSV="$CKPT_BASE/factuals_counterfactuals.csv"
STEP3_MODEL="$CKPT_BASE/model.pth"
if [ ! -d "$CKPT_BASE" ]; then
    echo "错误: Step 3 目录不存在: $CKPT_BASE"
    exit 1
fi
if [ ! -f "$CSV" ]; then
    echo "错误: 未找到 Step 4 产物: $CSV"
    exit 1
fi
if [ ! -f "$STEP3_MODEL" ]; then
    echo "提示: 未找到 Step 3 权重（部分场景可忽略）: $STEP3_MODEL"
fi

NEST_DIR="$CKPT_BASE/step5/$_inner"
if [ -n "$EVAL_ONLY" ]; then
    if [ ! -d "$NEST_DIR" ]; then
        echo "错误: --eval-only 所指的嵌套目录不存在: $NEST_DIR"
        exit 1
    fi
    if [ ! -f "$NEST_DIR/model.pth" ]; then
        echo "错误: --eval-only 需要已有权重: $NEST_DIR/model.pth"
        exit 1
    fi
else
    mkdir -p "$NEST_DIR"
fi

LINK="$NEST_DIR/factuals_counterfactuals.csv"
CSV_ABS="$(readlink -f "$CSV")"
if [ -L "$LINK" ]; then
    if [ "$(readlink -f "$LINK")" != "$CSV_ABS" ]; then
        echo "错误: 已存在软链但目标不是 Step 3 目录下的 csv: $LINK"
        exit 1
    fi
elif [ -e "$LINK" ]; then
    echo "错误: 已存在文件且非预期软链，请手动处理: $LINK"
    exit 1
else
    ln -s ../../factuals_counterfactuals.csv "$LINK"
fi

export D4C_CHECKPOINT_GROUP=step3
export D4C_CHECKPOINT_SUBDIR="${STEP3_SUBDIR_FOR_NEST}/step5/$_inner"
# 主日志 / 每任务 eval 汇总：log/<task>/step5/…（勿与 Step 3 共用 log/…/step3/）
if [ -z "${D4C_LOG_GROUP:-}" ] && [ -z "${D4C_LOG_SUBDIR:-}" ] && [ -z "${D4C_LOG_STEP:-}" ]; then
    export D4C_LOG_GROUP=step5
fi
echo "Step 5 checkpoint: $NEST_DIR （GROUP=step3 SUBDIR=$D4C_CHECKPOINT_SUBDIR）"
if [ -n "${D4C_LOG_GROUP:-}" ]; then
    echo "Step 5 主日志根（LOG_GROUP）: $D4C_ROOT/log/$TASK_ID/$D4C_LOG_GROUP/"
fi

_RUN_TS="$(date +%Y%m%d_%H%M)"
_LOG_TS="$(date +%Y%m%d_%H%M%S)"

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$(d4c_step5_logfile "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
    NOHUP_OUT="$(dirname "$LOGFILE")/nohup.log"
    mkdir -p "$(dirname "$LOGFILE")"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_TRAIN="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    ABS_NOHUP="$(readlink -f "$NOHUP_OUT" 2>/dev/null || echo "$NOHUP_OUT")"
    echo "已在后台启动 Step 5（DDP）"
    echo "  Python 日志 (--log_file): $ABS_TRAIN"
    echo "  nohup 终端输出: $ABS_NOHUP"
    echo "查看训练日志: tail -f $ABS_TRAIN"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$NOHUP_OUT" 2>&1 &
    echo "PID: $!"
    exit 0
fi

resolve_torchrun

if [ -z "$EPOCHS" ] && [ -z "$EVAL_ONLY" ]; then
    EPOCHS="--epochs $(cd "$CODE_DIR" && python -c "from config import get_epochs; print(get_epochs())")"
fi

if [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$(d4c_step5_logfile "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
fi

p=$(get_task_params "$TASK_ID")
[ -z "$p" ] && { echo "无效任务 $TASK_ID"; exit 1; }
aux=$(echo "$p" | cut -d' ' -f1)
tgt=$(echo "$p" | cut -d' ' -f2)
lr=$(echo "$p" | cut -d' ' -f3)
coef=$(echo "$p" | cut -d' ' -f4)
eta=$(echo "$p" | cut -d' ' -f5)

if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 5 DDP Task $TASK_ID 仅 eval (nproc=$DDP_NPROC): $aux -> $tgt =========="
elif [ -n "$TRAIN_ONLY" ]; then
    echo "========== Step 5 DDP Task $TASK_ID 仅 train (nproc=$DDP_NPROC): $aux -> $tgt =========="
else
    echo "========== Step 5 DDP Task $TASK_ID (nproc=$DDP_NPROC): $aux -> $tgt =========="
fi

_EVAL_FLAG=""
[ -n "$EVAL_ONLY" ] && _EVAL_FLAG="--eval-only"
_TRAIN_ONLY_FLAG=""
[ -n "$TRAIN_ONLY" ] && _TRAIN_ONLY_FLAG="--train-only"

if [ -n "$EVAL_ONLY" ]; then
    echo "启动 Step 5（DDP）Task $TASK_ID【仅 eval】，日志: $LOGFILE"
elif [ -n "$TRAIN_ONLY" ]; then
    echo "启动 Step 5（DDP）Task $TASK_ID【仅 train】，日志: $LOGFILE"
else
    echo "启动 Step 5（DDP）Task $TASK_ID，日志: $LOGFILE"
fi

${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" run-d4c.py \
    --auxiliary "$aux" --target "$tgt" $EPOCHS --learning_rate "$lr" --coef "$coef" --eta "$eta" \
    $BATCH_SIZE $NUM_PROC \
    $_EVAL_FLAG $_TRAIN_ONLY_FLAG \
    --log_file "$LOGFILE"

if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 5（仅 eval）完成 =========="
elif [ -n "$TRAIN_ONLY" ]; then
    echo "========== Step 5（仅 train）完成 =========="
else
    echo "========== Step 5 完成 =========="
fi
