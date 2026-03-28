#!/bin/bash
#
# -----------------------------------------------------------------------------
# 兼容入口（推荐后续收敛至）：bash scripts/train_ddp.sh — 见 docs/D4C_RUNTIME_SPEC.md
# -----------------------------------------------------------------------------
#
# run_step3_optimized.sh — Step 3 正式入口：域对抗预训练（torchrun + DDP）
#
# 默认 checkpoint：GROUP=step3_optimized，SUBDIR=step3_opt_<时间戳>
# 默认日志：log/<task>/step3_optimized/…（未预先设置 D4C_LOG_GROUP / D4C_LOG_SUBDIR / D4C_LOG_STEP 时）
#
# 用法与参数见下方「命令行」；另可通过环境变量覆盖训练语义（与 code/config.py 一致）。
#
# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
#   --task N          只跑任务 N（1–8）
#   --all             顺序跑全部 8 个任务
#   --eval-only       只跑 AdvTrain.py eval（跳过 train；需已有对应 checkpoint）
#   --train-only      只跑 AdvTrain.py train（跳过训练后的 AdvTrain.py eval；与 --eval-only 互斥）
#   --from N          仅配合 --all：从任务 N 开始（前面的跳过）
#   --skip a,b,...    跳过指定任务号
#   --batch-size / --epochs / --num-proc  传给 AdvTrain（与 config 一致时可省略部分）
#   --ddp-nproc K     DDP 进程数（也可用环境变量 DDP_NPROC，默认 2）；=1 为单卡 DDP smoke，仍为同一主路径
#   --daemon / --bg   后台：单任务 Python 日志 log/<task>/.../runs/<秒级时间戳>/train.log，nohup 同目录 nohup.log
#
# ---------------------------------------------------------------------------
# 默认环境（可用 export 覆盖；与历史 run_step3_optimized 包装一致）
# ---------------------------------------------------------------------------
#   D4C_RUNTIME_PRESET：可选；统一 CPU / DataLoader 并发（见 code/config.py 中 RUNTIME_PRESETS）。
#       单卡 12 核示例：export D4C_RUNTIME_PRESET=gpu01_single_12c
#       双卡 DDP 12 核示例：export D4C_RUNTIME_PRESET=gpu01_ddp2_12c
#       仍可用 MAX_PARALLEL_CPU、D4C_NUM_PROC、D4C_DATALOADER_WORKERS_* 等覆盖 preset。
#   推荐线程环境（不在 Python import 时设置，请在 shell 中 export）：
#       单卡：OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
#       双卡：OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
#   D4C_TRAIN_PRESET：训练语义预设（与 D4C_RUNTIME_PRESET 独立）；未设置时默认为 step3（presets/training/step3.yaml）
#   D4C_OPT_BATCH_SIZE：命令行未出现 --batch-size 时，若非按任务 TRAINING_PRESETS 则追加 get_train_batch_size()
#   D4C_FULL_EVAL_EVERY：未设置时由 Python 分阶段 full BLEU（epoch≤10 每 5 轮、之后每 2 轮）
#
# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------
#   bash sh/run_step3_optimized.sh --task 4
#   DDP_NPROC=1 bash sh/run_step3_optimized.sh --task 2
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash sh/run_step3_optimized.sh --task 2 --batch-size 1024
#   bash sh/run_step3_optimized.sh --all --from 4
#   bash sh/run_step3_optimized.sh --task 2 --daemon
#
# Step 4：bash sh/run_step4_optimized.sh --task N --step3-subdir <与本脚本 SUBDIR 相同>
#
set -euo pipefail
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$CODE_DIR"
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

# ---- 训练语义默认值（与 config / AdvTrain 环境变量一致）----
export D4C_LR_SCHEDULER="${D4C_LR_SCHEDULER:-warmup_cosine}"
export D4C_WARMUP_RATIO="${D4C_WARMUP_RATIO:-0.05}"
export D4C_QUICK_EVAL_MAX_SAMPLES="${D4C_QUICK_EVAL_MAX_SAMPLES:-512}"
export TRAIN_EARLY_STOP_PATIENCE_FULL="${TRAIN_EARLY_STOP_PATIENCE_FULL:-4}"
export TRAIN_MIN_EPOCHS="${TRAIN_MIN_EPOCHS:-8}"
export TRAIN_EARLY_STOP_PATIENCE="${TRAIN_EARLY_STOP_PATIENCE:-6}"
export TRAIN_BLEU4_MAX_SAMPLES="${TRAIN_BLEU4_MAX_SAMPLES:-512}"
export D4C_TRAIN_PRESET="${D4C_TRAIN_PRESET:-step3}"

if [ -z "${D4C_LOG_GROUP:-}" ] && [ -z "${D4C_LOG_SUBDIR:-}" ] && [ -z "${D4C_LOG_STEP:-}" ]; then
    export D4C_LOG_GROUP=step3_optimized
fi

d4c_run_log_dir() {
    python -c "from paths_config import get_log_task_dir; import os; print(get_log_task_dir($1))"
}
d4c_run_log_path() {
    python -c "from paths_config import get_log_task_dir; import os; print(os.path.join(get_log_task_dir($1), 'runs', 'run', 'train.log'))"
}
d4c_step3_logfile() {
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
    python -c "from config import TASK_DEFAULTS; import sys; t=int(sys.argv[1]); c=TASK_DEFAULTS[t]; sys.stdout.write('{} {} {} {} {}'.format(c['auxiliary'], c['target'], c['lr'], c['coef'], c['adv'])); sys.exit(0)" "$1"
}

MODE=""
TASK_ID=""
SKIP_LIST=""
FROM_TASK=1
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
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
        echo "单卡 DDP smoke: DDP_NPROC=1 bash sh/run_step3_optimized.sh --task 1" >&2
        exit 2
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

_extra=()
if [[ "$*" != *"--batch-size"* ]]; then
    if [ -n "${D4C_OPT_BATCH_SIZE:-}" ]; then
        _extra+=(--batch-size "$D4C_OPT_BATCH_SIZE")
    elif cd "$CODE_DIR" && python -c "from config import training_preset_is_per_task; import sys; sys.exit(0 if training_preset_is_per_task() else 1)"; then
        :
    else
        _default_bs="$(cd "$CODE_DIR" && python -c "from config import get_train_batch_size; print(get_train_batch_size())")"
        _extra+=(--batch-size "$_default_bs")
    fi
fi

export D4C_CHECKPOINT_GROUP="${D4C_CHECKPOINT_GROUP:-step3_optimized}"
if [ -z "${D4C_CHECKPOINT_SUBDIR:-}" ]; then
    _RUN_TS="$(date +%Y%m%d_%H%M)"
    export D4C_CHECKPOINT_SUBDIR="step3_opt_${_RUN_TS}"
else
    _RUN_TS="$(date +%Y%m%d_%H%M)"
fi
_LOG_TS="$(date +%Y%m%d_%H%M%S)"

echo "[run_step3_optimized] CHECKPOINT_GROUP=$D4C_CHECKPOINT_GROUP SUBDIR=$D4C_CHECKPOINT_SUBDIR"
if [ -n "${D4C_FULL_EVAL_EVERY:-}" ]; then
    export D4C_FULL_EVAL_EVERY
    echo "[run_step3_optimized] D4C_FULL_EVAL_EVERY=$D4C_FULL_EVAL_EVERY QUICK_EVAL_MAX=$D4C_QUICK_EVAL_MAX_SAMPLES PATIENCE_FULL=$TRAIN_EARLY_STOP_PATIENCE_FULL"
else
    echo "[run_step3_optimized] D4C_FULL_EVAL_EVERY=(unset → Python phased 5/10/2) QUICK_EVAL_MAX=$D4C_QUICK_EVAL_MAX_SAMPLES PATIENCE_FULL=$TRAIN_EARLY_STOP_PATIENCE_FULL"
fi

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --all | --task N [--eval-only|--train-only] [--from N] [--skip N,M,...] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  --eval-only：只跑 eval，跳过 train（须已有训练产物）"
    echo "  --train-only：只 train，跳过训练后的 eval（与 --eval-only 互斥）"
    echo "  --daemon / --bg：后台运行；详见文件头"
    echo "  DDP：torchrun AdvTrain.py train；DDP_NPROC 或 --ddp-nproc（默认 2）"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ "$MODE" = "all" ]; then
        if [ -n "$EVAL_ONLY" ]; then
            LOGFILE="$LOG_DIR/step3_optimized_eval_daemon_${_RUN_TS}.log"
        else
            LOGFILE="$LOG_DIR/step3_optimized_daemon_${_RUN_TS}.log"
        fi
        NOHUP_OUT="$LOGFILE"
    else
        LOGFILE="$(d4c_step3_logfile "$TASK_ID")"
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
        _EX="$(d4c_step3_logfile 1)"
        if [ -n "$EVAL_ONLY" ]; then
            echo "已在后台启动 Step 3（DDP）全部 8 任务【仅 eval】；终端汇总: $ABS_LOG"
        elif [ -n "$TRAIN_ONLY" ]; then
            echo "已在后台启动 Step 3（DDP）全部 8 任务【仅 train】；终端汇总: $ABS_LOG"
        else
            echo "已在后台启动 Step 3（DDP）全部 8 任务；终端汇总: $ABS_LOG"
        fi
        echo "  每任务 Python 日志示例（Task 1）: $_EX"
        echo "查看汇总: tail -f $ABS_LOG"
    else
        ABS_TRAIN="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
        ABS_NOHUP="$(readlink -f "$NOHUP_OUT" 2>/dev/null || echo "$NOHUP_OUT")"
        echo "已在后台启动 Step 3（DDP）"
        echo "  Python 日志 (--log_file): $ABS_TRAIN"
        echo "  nohup 终端输出: $ABS_NOHUP"
        echo "查看训练日志: tail -f $ABS_TRAIN"
    fi
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$NOHUP_OUT" 2>&1 &
    echo "PID: $!"
    exit 0
fi

resolve_torchrun

_adv_train() {
    local _cmd=$1
    if [ "${#_extra[@]}" -gt 0 ]; then
        ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py "$_cmd" \
            --auxiliary "$aux" --target "$tgt" $_epochs_arg --learning_rate "$lr" --coef "$coef" --adv "$adv" \
            $BATCH_SIZE $NUM_PROC "${_extra[@]}" \
            --log_file "$LOGFILE"
    else
        ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py "$_cmd" \
            --auxiliary "$aux" --target "$tgt" $_epochs_arg --learning_rate "$lr" --coef "$coef" --adv "$adv" \
            $BATCH_SIZE $NUM_PROC \
            --log_file "$LOGFILE"
    fi
}

run_one_task() {
    local idx=$1
    local p
    p=$(get_task_params "$idx")
    [ -z "$p" ] && { echo "无效任务 $idx"; return 1; }
    aux=$(echo "$p" | cut -d' ' -f1)
    tgt=$(echo "$p" | cut -d' ' -f2)
    lr=$(echo "$p" | cut -d' ' -f3)
    coef=$(echo "$p" | cut -d' ' -f4)
    adv=$(echo "$p" | cut -d' ' -f5)
    _epochs_arg=""
    if [ -z "$EVAL_ONLY" ]; then
        if [ -n "$EPOCHS" ]; then
            _epochs_arg="$EPOCHS"
        else
            _epochs_arg="--epochs $(cd "$CODE_DIR" && D4C_PRESET_TASK_ID="$idx" python -c "from config import get_epochs; print(get_epochs())")"
        fi
    fi
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Step 3 DDP Task $idx 仅 eval (nproc=$DDP_NPROC): $aux -> $tgt =========="
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "========== Step 3 DDP Task $idx 仅 train (nproc=$DDP_NPROC): $aux -> $tgt =========="
    else
        echo "========== Step 3 DDP Task $idx (nproc=$DDP_NPROC): $aux -> $tgt =========="
    fi
    if [ -z "$EVAL_ONLY" ]; then
        _adv_train train
    fi
    if [ -z "$TRAIN_ONLY" ]; then
        if [ "${#_extra[@]}" -gt 0 ]; then
            ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py eval \
                --auxiliary "$aux" --target "$tgt" $BATCH_SIZE $NUM_PROC "${_extra[@]}" \
                --log_file "$LOGFILE"
        else
            ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py eval \
                --auxiliary "$aux" --target "$tgt" $BATCH_SIZE $NUM_PROC \
                --log_file "$LOGFILE"
        fi
    fi
}

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    LOGFILE="$(d4c_step3_logfile "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
fi
if [ "$MODE" = "all" ]; then
    # 与 Python --log_file（train.log）分离：shell/tee 只写汇总文件，避免与 FileHandler 多写者竞争
    STEP3_ALL_SHELL_LOG="${D4C_STEP3_ALL_SHELL_LOG:-$LOG_DIR/step3_optimized_all_${_LOG_TS}.log}"
    mkdir -p "$(dirname "$STEP3_ALL_SHELL_LOG")"
    mkdir -p "$LOG_DIR"
    [ "$FROM_TASK" -gt 1 ] && echo "从 Task $FROM_TASK 起跑（跳过 1-$((FROM_TASK - 1))）"
    [ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 3（DDP）全部 8 个任务【仅 eval】；每任务 Python 日志 .../runs/<时间戳>/train.log（D4C_LOG_USE_TIMESTAMP=0 时为 .../runs/run/train.log）"
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "启动 Step 3（DDP）全部 8 个任务【仅 train】；每任务 Python 日志 .../runs/<时间戳>/train.log（D4C_LOG_USE_TIMESTAMP=0 时为 .../runs/run/train.log）"
    else
        echo "启动 Step 3（DDP）全部 8 个任务；每任务 Python 日志 .../runs/<时间戳>/train.log（D4C_LOG_USE_TIMESTAMP=0 时为 .../runs/run/train.log）"
    fi
    _EX="$(d4c_step3_logfile 1)"
    echo "  示例（Task 1）: $_EX"
    echo "  Shell 提示/跳过行（tee，非 train.log）: $STEP3_ALL_SHELL_LOG"
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
