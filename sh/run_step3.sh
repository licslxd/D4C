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
#   --train-only      只跑 AdvTrain.py train（跳过训练后的 AdvTrain.py eval；与 --eval-only 互斥）
#   --from N          仅配合 --all：从任务 N 开始（前面的跳过）
#   --skip a,b,...    跳过指定任务号
#   --batch-size / --epochs / --num-proc  传给 AdvTrain（与 config 一致时可省略部分）
#   --ddp-nproc K     DDP 进程数（也可用环境变量 DDP_NPROC，默认 2）
#   --gpus 0,1        仅用于「单进程」手动跑 eval 时的 DataParallel；本脚本内 eval 已用 torchrun DDP，不传 --gpus
#   --daemon / --bg   后台：单任务 Python 日志 log/<task>/.../runs/<秒级时间戳>/train.log，nohup 同目录 nohup.log；--all 时汇总仍写 log/step3_daemon_*.log（D4C_LOG_USE_TIMESTAMP=0 时为 runs/run/train.log）
#                     后台子进程默认 D4C_CONSOLE_LEVEL=WARNING，避免与 --log_file 重复一行（见 train_logging / run_step5 注释）
#
# ---------------------------------------------------------------------------
# Checkpoint（D4C_CHECKPOINT_GROUP + D4C_CHECKPOINT_SUBDIR）
# ---------------------------------------------------------------------------
# 默认：export D4C_CHECKPOINT_GROUP=step3、D4C_CHECKPOINT_SUBDIR=step3_YYYYMMDD_HHMM
# 权重：checkpoints/<task>/step3/step3_<时间>/model.pth
# 主日志（Python / --log_file）：log/<task>/.../runs/<秒级时间戳>/train.log；eval 仍在 log/<task>/.../eval/
# 仅 SUBDIR、不设 GROUP：路径规则随 get_log_task_dir；D4C_LOG_USE_TIMESTAMP=0 时为 .../runs/run/train.log
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
#   bash run_step3.sh --task 4 --train-only                 # 只训练，不跑收尾 eval
#   bash run_step3.sh --all --eval-only --from 6             # 从 Task 6 起只对剩余任务做 eval
#   bash run_step3.sh --task 2 --daemon                      # 后台：runs/<时间戳>/train.log + nohup.log
#
#   --daemon / --bg   单任务：runs/<时间戳>/train.log 与 nohup.log；--all 时另有汇总 log/step3_daemon_*.log
#   固定路径（覆盖写入）：export D4C_LOG_USE_TIMESTAMP=0 → runs/run/train.log

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

d4c_run_log_dir() {
    python -c "from paths_config import get_log_task_dir; import os; print(get_log_task_dir($1))"
}
d4c_run_log_path() {
    python -c "from paths_config import get_log_task_dir; import os; print(os.path.join(get_log_task_dir($1), 'runs', 'run', 'train.log'))"
}
# 方案 B：runs/<秒级时间戳>/train.log；D4C_LOG_USE_TIMESTAMP=0 时为 runs/run/train.log（与 run_step5.sh 一致）
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
    # nohup 将 stdout/stderr 重定向到与 --log_file 相同路径时，若控制台仍为 INFO，会与 FileHandler 各写一遍。
    # 默认仅把 WARNING+ 打到 stderr；INFO 由 AdvTrain 写入日志文件。可用 D4C_CONSOLE_LEVEL 覆盖。
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

# 任务 N → 一行五元组：辅助域 目标域 学习率 coef adv（与 code/config.py task_configs 一致；若 export D4C_TRAIN_PRESET 则 adversarial_coef 可被预设覆盖）
get_task_params() {
    python -c "from config import format_step3_task_params_line as _line; import sys; s=_line($1); sys.stdout.write(s); sys.exit(0 if s else 1)"
}

MODE=""
TASK_ID=""
GPUS=""
SKIP_LIST=""
FROM_TASK=1
EVAL_ONLY=""
TRAIN_ONLY=""
DAEMON=""
prev="" # 上一项为需值的选项时，记录键名
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

if [ -n "$EVAL_ONLY" ] && [ -n "$TRAIN_ONLY" ]; then
    echo "错误: --eval-only 与 --train-only 不能同时使用"
    exit 1
fi

# 未设置 D4C_CHECKPOINT_SUBDIR 时默认 step3_<时间戳>，并默认 GROUP=step3 → checkpoints/…/step3/step3_<时间>/
if [ -z "${D4C_CHECKPOINT_SUBDIR:-}" ]; then
    _RUN_TS="$(date +%Y%m%d_%H%M)"
    export D4C_CHECKPOINT_SUBDIR="step3_${_RUN_TS}"
    export D4C_CHECKPOINT_GROUP="${D4C_CHECKPOINT_GROUP:-step3}"
else
    _RUN_TS="$(date +%Y%m%d_%H%M)"
fi
# 与 runs/<秒级时间戳>/ 对齐（与 run_step5.sh 一致）
_LOG_TS="$(date +%Y%m%d_%H%M%S)"

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --all | --task N [--eval-only|--train-only] [--from N] [--skip N,M,...] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--gpus ...] [--daemon|--bg]"
    echo "  --eval-only：只跑 eval，跳过 train（须已有训练产物）"
    echo "  --train-only：只 train，跳过训练后的 eval（与 --eval-only 互斥）"
    echo "  --daemon / --bg：后台运行；单任务 runs/<时间戳>/train.log + nohup.log；详见文件头"
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

run_one_task() {
    local idx=$1
    local p=$(get_task_params $idx)
    [ -z "$p" ] && { echo "无效任务 $idx"; return 1; }
    local aux=$(echo $p | cut -d' ' -f1)
    local tgt=$(echo $p | cut -d' ' -f2)
    local lr=$(echo $p | cut -d' ' -f3)
    local coef=$(echo $p | cut -d' ' -f4)
    local adv=$(echo $p | cut -d' ' -f5)
    local _epochs_arg=""
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
    [ -n "$GPUS" ] && echo "提示: 本脚本内 eval 已用 torchrun DDP，忽略 --gpus（请用 CUDA_VISIBLE_DEVICES）"
    if [ -z "$EVAL_ONLY" ]; then
        ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py train \
            --auxiliary "$aux" --target "$tgt" $_epochs_arg --learning_rate "$lr" --coef "$coef" --adv "$adv" \
            $BATCH_SIZE $NUM_PROC \
            --log_file "$LOGFILE"
    fi
    if [ -z "$TRAIN_ONLY" ]; then
        ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py eval \
            --auxiliary "$aux" --target "$tgt" $BATCH_SIZE $NUM_PROC \
            --log_file "$LOGFILE"
    fi
}

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    LOGFILE="$(d4c_step3_logfile "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
fi
if [ "$MODE" = "all" ]; then
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
    for i in 1 2 3 4 5 6 7 8; do
        LOGFILE="$(d4c_step3_logfile "$i")"
        mkdir -p "$(dirname "$LOGFILE")"
        [ "$i" -lt "$FROM_TASK" ] && { echo "========== 跳过 Task $i (--from $FROM_TASK) ==========" | tee -a "$LOGFILE"; continue; }
        should_skip $i && { echo "========== 跳过 Task $i ==========" | tee -a "$LOGFILE"; continue; }
        echo "---------- Task $i 日志: $LOGFILE ----------" | tee -a "$LOGFILE"
        # 勿对 LOGFILE 再 tee：AdvTrain 已通过 --log_file 写入同一文件，tee 会重复每条 INFO
        run_one_task $i || { echo "Task $i 失败"; exit 1; }
    done
else
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 3（DDP）Task $TASK_ID【仅 eval】，日志: $LOGFILE"
    elif [ -n "$TRAIN_ONLY" ]; then
        echo "启动 Step 3（DDP）Task $TASK_ID【仅 train】，日志: $LOGFILE"
    else
        echo "启动 Step 3（DDP）Task $TASK_ID，日志: $LOGFILE"
    fi
    # 勿对 LOGFILE 再 tee：AdvTrain 已通过 --log_file 写入同一文件，tee 会重复每条 INFO
    run_one_task "$TASK_ID"
fi
if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 3（仅 eval）完成 =========="
elif [ -n "$TRAIN_ONLY" ]; then
    echo "========== Step 3（仅 train）完成 =========="
else
    echo "========== Step 3 完成 =========="
fi
