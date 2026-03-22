#!/bin/bash
#
# run_step3.sh — Step 3：域对抗预训练（torchrun + DDP）
# 说明：原 run_step3_ddp.sh 已合并到此脚本，请只用本文件。
#
# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
#   --task N          只跑任务 N（1–8）
#   --all             顺序跑全部 8 个任务
#   --eval-only       只跑 AdvTrain.py eval（跳过 train；需已有对应 checkpoint）
#   --from N          仅配合 --all：从任务 N 开始（前面的跳过）
#   --skip a,b,...    跳过指定任务号
#   --batch-size / --epochs / --num-proc  传给 AdvTrain（与 config 一致时可省略部分）
#   --ddp-nproc K     DDP 进程数（也可用环境变量 DDP_NPROC，默认 2）
#   --gpus 0,1        仅用于「单进程」手动跑 eval 时的 DataParallel；本脚本内 eval 已用 torchrun DDP，不传 --gpus
#   --daemon / --bg   后台运行：单任务 log/<task>/.../run.log；--all 时汇总 log/step3_daemon_*.log
#
# ---------------------------------------------------------------------------
# Checkpoint（D4C_CHECKPOINT_GROUP + D4C_CHECKPOINT_SUBDIR）
# ---------------------------------------------------------------------------
# 默认：export D4C_CHECKPOINT_GROUP=step3，D4C_CHECKPOINT_SUBDIR=step3_YYYYMMDD_HHMM（与 LOGFILE 时间戳一致），
# 权重路径 checkpoints/<task>/step3/step3_<时间>/model.pth。
# 仅自定义子目录名：export D4C_CHECKPOINT_SUBDIR=my_exp（不设 GROUP）→ checkpoints/<task>/my_exp/…
# 自定义且要中间层：export D4C_CHECKPOINT_GROUP=step3 D4C_CHECKPOINT_SUBDIR=my_exp
#
# 日志与 checkpoint 三层对齐（paths_config.get_log_task_dir）：log/<task>/<GROUP>/<SUBDIR>/run.log
# 不设 GROUP/SUBDIR 时退化为 log/<task>/run.log
#
# ---------------------------------------------------------------------------
# 每任务实际执行的命令（概念上等价）
# ---------------------------------------------------------------------------
#   训练（多进程 DDP）：
#     torchrun --standalone --nproc_per_node=$DDP_NPROC AdvTrain.py train \
#       --auxiliary <aux> --target <tgt> ...
#   评估（与训练相同 torchrun DDP；valid 分片并行，rank0 聚合指标）：
#     torchrun --standalone --nproc_per_node=$DDP_NPROC AdvTrain.py eval \
#       --auxiliary <aux> --target <tgt> ...
#
# DDP_NPROC 须与可见 GPU 数量一致（例：CUDA_VISIBLE_DEVICES=0,1 → DDP_NPROC=2）。
# 全局 batch 来自 config 或 --batch-size；每 rank = 全局 / DDP_NPROC，须能整除。
#
# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------
#   DDP_NPROC=1 bash run_step3.sh --task 2                    # 单卡（仍 nproc=1）
#   CUDA_VISIBLE_DEVICES=0,1 bash run_step3.sh --task 2       # 双卡；默认 DDP_NPROC=2，与可见 GPU 数一致
#   DDP_NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \                  # 四卡并行
#     bash run_step3.sh --task 2 --batch-size 1024            # 显式全局 batch（每 rank = 1024/4）
#   bash run_step3.sh --all --from 4                          # 全任务，从任务 4 起跑（跳过 1–3）
#   bash run_step3.sh --all                                 # eval 与 train 同 DDP_NPROC
#   bash run_step3.sh --task 5 --eval-only                  # 仅评估 Task 5（不重训）
#   bash run_step3.sh --all --eval-only --from 6             # 从 Task 6 起只对剩余任务做 eval
#   bash run_step3.sh --task 2 --daemon                      # 后台跑（log/<task>/.../run.log）
#
#   --daemon / --bg   后台运行：单任务写入 log/<task>/.../run.log；--all 时另有汇总 log/step3_daemon_*.log

set -euo pipefail
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$CODE_DIR"
# 离线 NLTK（wordnet / punkt / omw-1.4），避免 torchrun 子进程未继承 shell 里的 NLTK_DATA
export NLTK_DATA="${D4C_ROOT}/pretrained_models/nltk_data"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

d4c_run_log_path() {
    python -c "from paths_config import get_log_task_dir; import os; print(os.path.join(get_log_task_dir($1), 'run.log'))"
}

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
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

# 任务 N → 一行五元组：辅助域 目标域 学习率 coef adv
get_task_params() {
    case $1 in
        1) echo "AM_Electronics AM_CDs 5e-4 1 0.01" ;;
        2) echo "AM_Movies AM_CDs 1e-3 0.1 0.01" ;;
        3) echo "AM_CDs AM_Electronics 5e-4 0.5 0.1" ;;
        4) echo "AM_Movies AM_Electronics 1e-3 0.5 0.01" ;;
        5) echo "AM_CDs AM_Movies 1e-3 0.5 0.01" ;;
        6) echo "AM_Electronics AM_Movies 1e-3 0.5 0.01" ;;
        7) echo "Yelp TripAdvisor 1e-4 0.5 0.01" ;;
        8) echo "TripAdvisor Yelp 5e-4 1 0.01" ;;
        *) echo "" ;;
    esac
}

MODE=""
TASK_ID=""
GPUS=""
SKIP_LIST=""
FROM_TASK=1
EVAL_ONLY=""
DAEMON=""
prev="" # 上一项为需值的选项时，记录键名
for i in "$@"; do
    if [ "$i" = "--all" ]; then MODE="all"
    elif [ "$i" = "--eval-only" ]; then EVAL_ONLY=1
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
    elif [ "$i" = "--epochs" ]; then prev="--epochs"
    elif [ "$prev" = "--epochs" ]; then EPOCHS="--epochs $i"; prev=""
    elif [ "$i" = "--num-proc" ]; then prev="--num-proc"
    elif [ "$prev" = "--num-proc" ]; then NUM_PROC="--num-proc $i"; prev=""
    elif [ "$i" = "--ddp-nproc" ]; then prev="--ddp-nproc"
    elif [ "$prev" = "--ddp-nproc" ]; then DDP_NPROC="$i"; prev=""
    elif [ "$i" = "--daemon" ] || [ "$i" = "--bg" ]; then DAEMON=1
    fi
done

# 未设置 D4C_CHECKPOINT_SUBDIR 时默认 step3_<时间戳>，并默认 GROUP=step3 → …/step3/step3_<时间>/
if [ -z "${D4C_CHECKPOINT_SUBDIR:-}" ]; then
    _RUN_TS="$(date +%Y%m%d_%H%M)"
    export D4C_CHECKPOINT_SUBDIR="step3_${_RUN_TS}"
    export D4C_CHECKPOINT_GROUP="${D4C_CHECKPOINT_GROUP:-step3}"
else
    _RUN_TS="$(date +%Y%m%d_%H%M)"
fi

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --all | --task N [--eval-only] [--from N] [--skip N,M,...] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--gpus ...] [--daemon|--bg]"
    echo "  --eval-only：只跑 eval，跳过 train（须已有训练产物）"
    echo "  --daemon / --bg：后台运行，日志写入 log/；详见文件头"
    echo "  DDP：torchrun AdvTrain.py train；DDP_NPROC 或 --ddp-nproc（默认 2）；batch 须能被进程数整除；详见文件头"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ "$MODE" = "all" ]; then
        if [ -n "$EVAL_ONLY" ]; then
            LOGFILE="$LOG_DIR/step3_eval_daemon_${_RUN_TS}.log"
        else
            LOGFILE="$LOG_DIR/step3_daemon_${_RUN_TS}.log"
        fi
    else
        LOGFILE="$(d4c_run_log_path "$TASK_ID")"
    fi
    mkdir -p "$(dirname "$LOGFILE")"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    if [ "$MODE" = "all" ]; then
        _EX="$(d4c_run_log_path 1)"
        echo "已在后台启动 Step 3（DDP）全部 8 任务；终端汇总: $ABS_LOG"
        echo "  每任务日志示例（Task 1）: $_EX"
    else
        echo "已在后台启动 Step 3（DDP），日志: $ABS_LOG"
    fi
    echo "查看汇总: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

resolve_torchrun

if [ -z "$EPOCHS" ] && [ -z "$EVAL_ONLY" ]; then
    EPOCHS="--epochs $(cd "$CODE_DIR" && python -c "from config import get_epochs; print(get_epochs())")"
fi

run_one_task() {
    local idx=$1
    local p=$(get_task_params $idx)
    [ -z "$p" ] && { echo "无效任务 $idx"; return 1; }
    local aux=$(echo $p | cut -d' ' -f1)
    local tgt=$(echo $p | cut -d' ' -f2)
    local lr=$(echo $p | cut -d' ' -f3)
    local coef=$(echo $p | cut -d' ' -f4)
    local adv=$(echo $p | cut -d' ' -f5)
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Step 3 DDP Task $idx 仅 eval (nproc=$DDP_NPROC): $aux -> $tgt =========="
    else
        echo "========== Step 3 DDP Task $idx (nproc=$DDP_NPROC): $aux -> $tgt =========="
    fi
    [ -n "$GPUS" ] && echo "提示: 本脚本内 eval 已用 torchrun DDP，忽略 --gpus（请用 CUDA_VISIBLE_DEVICES）"
    if [ -z "$EVAL_ONLY" ]; then
        ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py train \
            --auxiliary "$aux" --target "$tgt" $EPOCHS --learning_rate "$lr" --coef "$coef" --adv "$adv" \
            $BATCH_SIZE $NUM_PROC \
            --log_file "$LOGFILE"
    fi
    ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py eval \
        --auxiliary "$aux" --target "$tgt" $BATCH_SIZE $NUM_PROC \
        --log_file "$LOGFILE"
}

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    LOGFILE="$(d4c_run_log_path "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
fi
if [ "$MODE" = "all" ]; then
    [ "$FROM_TASK" -gt 1 ] && echo "从 Task $FROM_TASK 起跑（跳过 1-$((FROM_TASK - 1))）"
    [ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 3（DDP）全部 8 个任务【仅 eval】；每任务独立 log/<task>/<GROUP>/<SUBDIR>/run.log"
    else
        echo "启动 Step 3（DDP）全部 8 个任务；每任务独立 log/<task>/<GROUP>/<SUBDIR>/run.log"
    fi
    _EX="$(d4c_run_log_path 1)"
    echo "  示例（Task 1）: $_EX"
    for i in 1 2 3 4 5 6 7 8; do
        LOGFILE="$(d4c_run_log_path "$i")"
        mkdir -p "$(dirname "$LOGFILE")"
        [ "$i" -lt "$FROM_TASK" ] && { echo "========== 跳过 Task $i (--from $FROM_TASK) ==========" | tee -a "$LOGFILE"; continue; }
        should_skip $i && { echo "========== 跳过 Task $i ==========" | tee -a "$LOGFILE"; continue; }
        echo "---------- Task $i 日志: $LOGFILE ----------" | tee -a "$LOGFILE"
        run_one_task $i 2>&1 | tee -a "$LOGFILE" || { echo "Task $i 失败"; exit 1; }
    done
else
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 3（DDP）Task $TASK_ID【仅 eval】，日志: $LOGFILE"
    else
        echo "启动 Step 3（DDP）Task $TASK_ID，日志: $LOGFILE"
    fi
    run_one_task "$TASK_ID" 2>&1 | tee -a "$LOGFILE"
fi
if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 3（仅 eval）完成 =========="
else
    echo "========== Step 3 完成 =========="
fi
