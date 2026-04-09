#!/bin/bash
# -----------------------------------------------------------------------------
# MAINLINE: python code/d4c.py step5|eval …；本脚本仅编排 / nohup。
# 批量编排: bash scripts/entrypoints/train_ddp.sh --step 5 …
# -----------------------------------------------------------------------------
# scripts/entrypoints/step5.sh — Step 5：主训练与评估（由 d4c.py 内部 torchrun）
#
# 用法: bash scripts/entrypoints/step5.sh --task N --iter v1 --from-run 1 --eval-profile <stem> [选项…]
# 非 --train-only 时须 --eval-profile（与 config_loader 合同一致）；--train-only 不需要。
#   --from-run      Step 3 的 run 目录名（runs/task{T}/vN/train/step3/<run>/），与 d4c.py 一致
#   --iter          迭代 vN（默认 v1，可 export D4C_ITER）
#   --step5-run     Step5 目录名（如 2_1_1）；训练可省略→auto（须同时 --step4-run）；--eval-only 时必填
#   --step4-run     仅当训练且 --step5-run auto 时必填；显式 step5-run 时 CSV 仅由 step5 目录名反推 step4（忽略本项）
#
# 示例:
#   DDP_NPROC=1 bash scripts/entrypoints/step5.sh --task 2 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu
#   bash scripts/entrypoints/step5.sh --task 4 --iter v1 --from-run 2 --step4-run 2_1 --eval-only --step5-run 2_1_1 --eval-profile eval_balanced_2gpu
#
set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/../.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
export D4C_ROOT
cd "$D4C_ROOT"
# shellcheck source=../lib/common_paths.sh
source "$SH_DIR/../lib/common_paths.sh"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'EOF'
用法: bash scripts/entrypoints/step5.sh --task N --iter v1 --from-run <run> [选项]
  完整训练+评测: 须 --eval-profile <stem>（与 d4c.py step5 合同一致）
  --train-only:  仅训练，不需要 --eval-profile
  --eval-only:   调用 d4c eval，须 --step5-run 与 --eval-profile
  --preset       默认 step5
详见文件头注释。
EOF
    exit 0
fi

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
    export D4C_CONSOLE_LEVEL="${D4C_CONSOLE_LEVEL:-WARNING}"
fi

DDP_NPROC="${DDP_NPROC:-}"
_ddp_ws_args_s5=()
if [ -n "${DDP_NPROC:-}" ]; then
    _ddp_ws_args_s5=(--ddp-world-size "$DDP_NPROC")
fi

BATCH_SIZE=""
EPOCHS=""
NUM_PROC=""
SEED_CLI=()

TASK_ID=""
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
FROM_RUN_NAME=""
STEP4_RUN_CLI=""
STEP5_RUN_CLI=""
EVAL_PROFILE_CLI=""
PRESET_NAME="step5"
prev=""
for i in "$@"; do
    if [ "$i" = "--all" ]; then
        echo "错误: Step 5 已取消 --all（仅嵌套模式）。请逐任务: --task N --from-run <run> …" >&2
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
    elif [ "$i" = "--from-run" ]; then prev="--from-run"
    elif [ "$prev" = "--from-run" ]; then FROM_RUN_NAME="$i"; prev=""
    elif [ "$i" = "--iter" ]; then prev="--iter"
    elif [ "$prev" = "--iter" ]; then export D4C_ITER="$i"; prev=""
    elif [ "$i" = "--seed" ]; then prev="--seed"
    elif [ "$prev" = "--seed" ]; then SEED_CLI=(--seed "$i"); prev=""
    elif [ "$i" = "--step5-run" ]; then prev="--step5-run"
    elif [ "$prev" = "--step5-run" ]; then STEP5_RUN_CLI="$i"; prev=""
    elif [ "$i" = "--step4-run" ]; then prev="--step4-run"
    elif [ "$prev" = "--step4-run" ]; then STEP4_RUN_CLI="$i"; prev=""
    elif [ "$i" = "--eval-profile" ]; then prev="--eval-profile"
    elif [ "$prev" = "--eval-profile" ]; then EVAL_PROFILE_CLI="$i"; prev=""
    elif [ "$i" = "--preset" ]; then prev="--preset"
    elif [ "$prev" = "--preset" ]; then PRESET_NAME="$i"; prev=""
    elif [ "$i" = "--gpus" ] || [[ "$i" == --gpus=* ]]; then
        echo "错误: --gpus 已移除。请在 presets/hardware/*.yaml 中配置 cuda_visible_devices，或传 d4c.py --cuda-visible-devices / export CUDA_VISIBLE_DEVICES，并用 DDP_NPROC / --ddp-nproc 控制 DDP。" >&2
        exit 2
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

export D4C_ITER="${D4C_ITER:-v1}"

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

if ! [[ "$TASK_ID" =~ ^[1-8]$ ]]; then
    echo "用法: $0 --task N --iter v1 --from-run <run> [选项]"
    echo "  必填: --task N（1–8）、--from-run（Step3 run 目录名，同 d4c.py）"
    echo "  可选: --step5-run <name>（训练可省略→auto，auto 时须同时 --step4-run；--eval-only 时必填）"
    echo "  可选: --step4-run <name>（仅训练且 step5-run 为 auto 时必填；显式 step5-run 时不参与 CSV 路径）"
    echo "  可选: --eval-only | --train-only（互斥）、--seed、--batch-size、--epochs、--num-proc、--ddp-nproc、--daemon|--bg"
    exit 1
fi

if [ -z "${FROM_RUN_NAME:-}" ]; then
    echo "错误: 须指定 --from-run（Step3 run 目录名）"
    exit 1
fi

if [ -n "$EVAL_ONLY" ] && [ -z "$EVAL_PROFILE_CLI" ]; then
    echo "错误: --eval-only 须同时传入 --eval-profile <stem>（或与 d4c eval 等价的显式编排参数请直接用 python code/d4c.py eval …）" >&2
    exit 2
fi

if [ -z "$EVAL_ONLY" ] && [ -z "$TRAIN_ONLY" ] && [ -z "$EVAL_PROFILE_CLI" ]; then
    echo "错误: 完整 Step5（训练+收尾评测）须 --eval-profile <stem>；若只要训练请加 --train-only。" >&2
    exit 2
fi

if [ -n "$EVAL_ONLY" ]; then
    if [ -z "${STEP5_RUN_CLI:-}" ] || [ "${STEP5_RUN_CLI}" = "auto" ]; then
        echo "错误: --eval-only 须指定 --step5-run <Step5 目录名>（不可为 auto），例如 2_1_1"
        exit 1
    fi
fi

STEP5_RUN_ARG="auto"
if [ -n "$EVAL_ONLY" ]; then
    STEP5_RUN_ARG="$STEP5_RUN_CLI"
elif [ -n "${STEP5_RUN_CLI:-}" ]; then
    STEP5_RUN_ARG="$STEP5_RUN_CLI"
fi

S3ROOT="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/train/step3/${FROM_RUN_NAME}"
STEP3_MODEL="$S3ROOT/model/model.pth"
if [ ! -d "$S3ROOT" ]; then
    echo "错误: Step 3 目录不存在: $S3ROOT"
    exit 1
fi
if [ -z "$EVAL_ONLY" ]; then
    CSV=""
    if [ "$STEP5_RUN_ARG" != "auto" ]; then
        _S4_DERIVED="${STEP5_RUN_ARG%_*}"
        if [ "$_S4_DERIVED" = "$STEP5_RUN_ARG" ]; then
            echo "错误: --step5-run 须至少两段 slug（如 2_1_1），以便反推 train/step4/2_1/ 下 CSV。" >&2
            exit 1
        fi
        CSV="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/train/step4/${_S4_DERIVED}/factuals_counterfactuals.csv"
    elif [ -n "${STEP4_RUN_CLI:-}" ]; then
        CSV="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/train/step4/${STEP4_RUN_CLI}/factuals_counterfactuals.csv"
    else
        echo "错误: 训练且 --step5-run auto 时必须指定 --step4-run（与 d4c.py 一致）。" >&2
        exit 1
    fi
    if [ ! -f "$CSV" ]; then
        echo "错误: 未找到 Step 4 产物: $CSV" >&2
        exit 1
    fi
fi
if [ ! -f "$STEP3_MODEL" ]; then
    echo "提示: 未找到 Step 3 权重（部分场景可忽略）: $STEP3_MODEL"
fi

if [ -n "$EVAL_ONLY" ]; then
    S5ROOT="$D4C_ROOT/runs/task${TASK_ID}/${D4C_ITER}/train/step5/${STEP5_RUN_CLI}"
    if [ ! -d "$S5ROOT" ]; then
        echo "错误: --eval-only 所指的 Step5 目录不存在: $S5ROOT"
        exit 1
    fi
    if [ ! -f "$S5ROOT/model/best_mainline.pth" ]; then
        echo "错误: --eval-only 需要已有权重: $S5ROOT/model/best_mainline.pth"
        exit 1
    fi
fi

if [ -n "$EVAL_ONLY" ]; then
    echo "Step 5 将使用: iter=$D4C_ITER from_run=$FROM_RUN_NAME step5_run=$STEP5_RUN_ARG (eval)"
else
    _S4_ECHO="${STEP4_RUN_CLI:-}"
    if [ -z "$_S4_ECHO" ] && [ "$STEP5_RUN_ARG" != "auto" ]; then
        _S4_ECHO="${STEP5_RUN_ARG%_*}"
    fi
    echo "Step 5 将使用: iter=$D4C_ITER from_run=$FROM_RUN_NAME step4_run=${_S4_ECHO:-?} step5_run=$STEP5_RUN_ARG"
fi

_RUN_TS="$(date +%Y%m%d_%H%M)"

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ -n "$EVAL_ONLY" ]; then
        export D4C_EVAL_RUN_ID=""
        LOGFILE="$(d4c_predict_eval_log "$TASK_ID")"
    else
        export D4C_STEP5_RUN_ID=""
        if [ "$STEP5_RUN_ARG" != "auto" ]; then
            export D4C_STEP5_RUN_ID="$STEP5_RUN_ARG"
        fi
        LOGFILE="$(d4c_predict_step5_train_log "$TASK_ID")"
    fi
    mkdir -p "$(dirname "$LOGFILE")"
    NOHUP_OUT="$(dirname "$LOGFILE")/nohup.log"
    mkdir -p "$(dirname "$LOGFILE")"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_TRAIN="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    ABS_NOHUP="$(readlink -f "$NOHUP_OUT" 2>/dev/null || echo "$NOHUP_OUT")"
    echo "已在后台启动 Step 5（d4c.py）"
    echo "  Python 日志 (--log_file): $ABS_TRAIN"
    echo "  nohup 终端输出: $ABS_NOHUP"
    echo "查看训练日志: tail -f $ABS_TRAIN"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$NOHUP_OUT" 2>&1 &
    echo "PID: $!"
    exit 0
fi

PRESET="${PRESET_NAME}"
_trainonly=()
[ -n "$TRAIN_ONLY" ] && _trainonly=(--train-only)

_ep_arg=()
if [ -n "$EVAL_PROFILE_CLI" ]; then
    _ep_arg=(--eval-profile "$EVAL_PROFILE_CLI")
fi

_s5arg=(--step5-run "$STEP5_RUN_ARG")
_s4arg=()
[ -n "${STEP4_RUN_CLI:-}" ] && _s4arg=(--step4-run "$STEP4_RUN_CLI")

_ddp_echo="${DDP_NPROC:-<preset/runtime>}"
if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 5 Task $TASK_ID eval（d4c.py, DDP=$_ddp_echo）=========="
    # shellcheck disable=SC2086
    python "$D4C_ROOT/code/d4c.py" eval --task "$TASK_ID" --preset "$PRESET" \
        --iter "$D4C_ITER" \
        --from-run "$FROM_RUN_NAME" "${_s4arg[@]}" "${_s5arg[@]}" \
        "${_ddp_ws_args_s5[@]}" \
        "${_ep_arg[@]}" \
        $BATCH_SIZE $EPOCHS $NUM_PROC "${SEED_CLI[@]}"
else
    echo "========== Step 5 Task $TASK_ID train（d4c.py, DDP=$_ddp_echo）=========="
    # shellcheck disable=SC2086
    python "$D4C_ROOT/code/d4c.py" step5 --task "$TASK_ID" --preset "$PRESET" \
        --iter "$D4C_ITER" \
        --from-run "$FROM_RUN_NAME" "${_s4arg[@]}" "${_s5arg[@]}" \
        "${_ddp_ws_args_s5[@]}" \
        "${_ep_arg[@]}" \
        $BATCH_SIZE $EPOCHS $NUM_PROC "${SEED_CLI[@]}" \
        "${_trainonly[@]}"
fi

if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 5（仅 eval）完成 =========="
elif [ -n "$TRAIN_ONLY" ]; then
    echo "========== Step 5（仅 train）完成 =========="
else
    echo "========== Step 5 完成 =========="
fi
