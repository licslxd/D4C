#!/bin/bash
# Step 5：主训练与评估（DDP：torchrun + run-d4c.py）
# 用法: bash run_step5.sh --all              # 跑全部 8 个任务
#       bash run_step5.sh --task N            # 仅跑任务 N (1-8)
#       bash run_step5.sh --all --from 4     # 从 Task 4 起跑到 8
#       bash run_step5.sh --all --skip 2,5   # 跑全部，跳过任务 2 和 5
#       bash run_step5.sh --task 2 --batch-size 64 --epochs 30
#       bash run_step5.sh --task 3 --eval-only   # 仅评估（须已有 model.pth；见下方 checkpoint）
#
# ========== DDP（本步唯一路径）==========
# 每任务等价于：
#   torchrun --standalone --nproc_per_node=$DDP_NPROC run-d4c.py \
#     --auxiliary <aux> --target <tgt> [--epochs …] [--batch-size 全局batch] …
# 进程数：环境变量 DDP_NPROC，或参数 --ddp-nproc K（默认 DDP_NPROC=2）
# 须与可见 GPU 数一致（常用：CUDA_VISIBLE_DEVICES=0,1 配 DDP_NPROC=2）
# 全局 batch 须能被 DDP_NPROC 整除；单卡：DDP_NPROC=1 或 --ddp-nproc 1
#
# 示例:
#   DDP_NPROC=1 bash run_step5.sh --task 2
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step5.sh --all --batch-size 1024
#   bash run_step5.sh --all --daemon                        # 后台跑（--all 另有汇总 log/step5_daemon_*.log）
#
#   --daemon / --bg   单任务：log/<task>/<GROUP>/<SUBDIR>/run_<时间戳>.log（默认不覆盖；见 D4C_LOG_USE_TIMESTAMP）；--all：汇总 log/step5_daemon_*.log + 各任务日志
#
# ---------------------------------------------------------------------------
# Checkpoint（D4C_CHECKPOINT_GROUP + D4C_CHECKPOINT_SUBDIR）
# ---------------------------------------------------------------------------
# 默认：GROUP=step5，SUBDIR=step5_YYYYMMDD_HHMM → checkpoints/<task>/step5/step5_<时间>/model.pth。
# --eval-only：勿使用自动新建子目录；须 export D4C_CHECKPOINT_SUBDIR=（及可选 GROUP）指向已有 step5 训练目录。
# 仅自定义子目录：export D4C_CHECKPOINT_SUBDIR=my_exp（不设 GROUP）→ checkpoints/<task>/my_exp/…
#
# 日志与 checkpoint 三层对齐（paths_config.get_log_task_dir）：log/<task>/<GROUP>/<SUBDIR>/run_<时间戳>.log
# 不设 GROUP/SUBDIR 时退化为 log/<task>/run_<时间戳>.log（与旧版单层 log/step5_*.log 并存不冲突）
# 若需每次仍写入固定文件名 run.log（覆盖旧内容）：export D4C_LOG_USE_TIMESTAMP=0
#
# ---------------------------------------------------------------------------

set -e
# 须在 cd 到 code/ 之前解析脚本绝对路径，否则 dirname 为相对路径（如 sh）时会在 code/ 下误执行 cd sh
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$CODE_DIR"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

# 与 checkpoints 对称：log/<task>/.../（依赖当前环境的 D4C_CHECKPOINT_GROUP / D4C_CHECKPOINT_SUBDIR）
d4c_run_log_dir() {
    python -c "from paths_config import get_log_task_dir; import os; print(get_log_task_dir($1))"
}
d4c_run_log_path() {
    python -c "from paths_config import get_log_task_dir; import os; print(os.path.join(get_log_task_dir($1), 'run.log'))"
}
# 默认 run_<秒级时间戳>.log 避免覆盖；D4C_LOG_USE_TIMESTAMP=0 时退回 run.log
d4c_step5_logfile() {
    local tid=$1
    if [ "${D4C_LOG_USE_TIMESTAMP:-1}" != "0" ]; then
        echo "$(d4c_run_log_dir "$tid")/run_${_LOG_TS}.log"
    else
        d4c_run_log_path "$tid"
    fi
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

# 任务映射 (task_idx -> auxiliary target lr coef eta)
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

MODE=""
TASK_ID=""
SKIP_LIST=""
FROM_TASK=1
EVAL_ONLY=""
DAEMON=""
prev=""
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

if [ -z "${D4C_CHECKPOINT_SUBDIR:-}" ]; then
    if [ -n "$EVAL_ONLY" ]; then
        echo "错误: --eval-only 须指向已有 checkpoint，请先设置环境变量，例如:"
        echo "  export D4C_CHECKPOINT_GROUP=step5"
        echo "  export D4C_CHECKPOINT_SUBDIR=step5_20250320_1430   # 你的训练子目录名"
        echo "再运行: $0 --task N --eval-only"
        exit 1
    fi
    _RUN_TS="$(date +%Y%m%d_%H%M)"
    export D4C_CHECKPOINT_SUBDIR="step5_${_RUN_TS}"
    export D4C_CHECKPOINT_GROUP="${D4C_CHECKPOINT_GROUP:-step5}"
else
    _RUN_TS="$(date +%Y%m%d_%H%M)"
fi
# 与 run_<时间戳>.log 对齐（秒级，避免同分钟内多次运行重名）
_LOG_TS="$(date +%Y%m%d_%H%M%S)"

[ -z "$MODE" ] && [ -z "$TASK_ID" ] && {
    echo "用法: $0 --all | --task N [--eval-only] [--from N] [--skip N,M,...] [--batch-size N] [--epochs N] [--num-proc N] [--ddp-nproc K] [--daemon|--bg]"
    echo "  --eval-only：只跑 valid 评估（跳过训练），须已有 model.pth；须设置 D4C_CHECKPOINT_SUBDIR 等，见文件头"
    echo "  --daemon / --bg：后台运行，日志写入 log/；详见文件头"
    echo "  DDP：torchrun run-d4c.py；DDP_NPROC 或 --ddp-nproc（默认 2）；全局 batch 须能被进程数整除；单卡请 DDP_NPROC=1"
    echo "  日志默认 log/.../run_<时间戳>.log 不覆盖；若需固定 run.log：export D4C_LOG_USE_TIMESTAMP=0"
    exit 1
}

if [ "$MODE" != "all" ] && [ -n "$TASK_ID" ] && [ "$FROM_TASK" -gt 1 ]; then
    echo "提示: --from 仅用于 --all，已忽略 --from $FROM_TASK"
    FROM_TASK=1
fi

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    if [ "$MODE" = "all" ]; then
        if [ -n "$EVAL_ONLY" ]; then
            LOGFILE="$LOG_DIR/step5_eval_daemon_${_RUN_TS}.log"
        else
            LOGFILE="$LOG_DIR/step5_daemon_${_RUN_TS}.log"
        fi
    else
        LOGFILE="$(d4c_step5_logfile "$TASK_ID")"
    fi
    mkdir -p "$(dirname "$LOGFILE")"
    args=()
    for a in "$@"; do
        if [ "$a" != "--daemon" ] && [ "$a" != "--bg" ]; then args+=("$a"); fi
    done
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    if [ "$MODE" = "all" ]; then
        _EX="$(d4c_step5_logfile 1)"
        if [ -n "$EVAL_ONLY" ]; then
            echo "已在后台启动 Step 5（DDP）全部 8 任务【仅 eval】；终端汇总: $ABS_LOG"
        else
            echo "已在后台启动 Step 5（DDP）全部 8 任务；终端汇总: $ABS_LOG"
        fi
        echo "  每任务训练日志（与 checkpoint 同层）示例（Task 1）: $_EX"
    else
        echo "已在后台启动 Step 5（DDP），日志: $ABS_LOG"
    fi
    echo "查看汇总: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${args[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

resolve_torchrun

# 未指定 --epochs 时从 config 读取默认值（--eval-only 不传 epochs）
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
    local eta=$(echo $p | cut -d' ' -f5)
    if [ -n "$EVAL_ONLY" ]; then
        echo "========== Step 5 DDP Task $idx 仅 eval (nproc=$DDP_NPROC): $aux -> $tgt =========="
    else
        echo "========== Step 5 DDP Task $idx (nproc=$DDP_NPROC): $aux -> $tgt =========="
    fi
    # --log_file 由 run-d4c 内 FileHandler 写入 LOGFILE；勿再对同一文件 tee，否则会与 StreamHandler 重复一行
    _EVAL_FLAG=""
    [ -n "$EVAL_ONLY" ] && _EVAL_FLAG="--eval-only"
    ${TORCHRUN_BIN} --standalone --nproc_per_node="$DDP_NPROC" run-d4c.py \
        --auxiliary "$aux" --target "$tgt" $EPOCHS --learning_rate "$lr" --coef "$coef" --eta "$eta" \
        $BATCH_SIZE $NUM_PROC \
        $_EVAL_FLAG \
        --log_file "$LOGFILE"
}

should_skip() { [[ " $SKIP_LIST " =~ " $1 " ]]; }

# 前台单任务：预先确定日志路径；--all 在每轮循环内按任务设置
if [ -z "${INTERNAL_NOHUP:-}" ] && [ "$MODE" != "all" ]; then
    LOGFILE="$(d4c_step5_logfile "$TASK_ID")"
    mkdir -p "$(dirname "$LOGFILE")"
fi
if [ "$MODE" = "all" ]; then
    [ "$FROM_TASK" -gt 1 ] && echo "从 Task $FROM_TASK 起跑（跳过 1-$((FROM_TASK - 1))）"
    [ -n "$SKIP_LIST" ] && echo "跳过任务: $SKIP_LIST"
    _EX="$(d4c_step5_logfile 1)"
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 5（DDP）全部 8 个任务【仅 eval】；每任务独立 log/<task>/<GROUP>/<SUBDIR>/run_<时间戳>.log"
    else
        echo "启动 Step 5（DDP）全部 8 个任务；每任务独立 log/<task>/<GROUP>/<SUBDIR>/run_<时间戳>.log"
    fi
    echo "  示例（Task 1）: $_EX"
    for i in 1 2 3 4 5 6 7 8; do
        LOGFILE="$(d4c_step5_logfile "$i")"
        mkdir -p "$(dirname "$LOGFILE")"
        [ "$i" -lt "$FROM_TASK" ] && { echo "========== 跳过 Task $i (--from $FROM_TASK) ==========" | tee -a "$LOGFILE"; continue; }
        should_skip $i && { echo "========== 跳过 Task $i ==========" | tee -a "$LOGFILE"; continue; }
        echo "---------- Task $i 日志: $LOGFILE ----------" | tee -a "$LOGFILE"
        run_one_task $i || { echo "Task $i 失败"; exit 1; }
    done
else
    if [ -n "$EVAL_ONLY" ]; then
        echo "启动 Step 5（DDP）Task $TASK_ID【仅 eval】，日志: $LOGFILE"
    else
        echo "启动 Step 5（DDP）Task $TASK_ID，日志: $LOGFILE"
    fi
    run_one_task $TASK_ID
fi
if [ -n "$EVAL_ONLY" ]; then
    echo "========== Step 5（仅 eval）完成 =========="
else
    echo "========== Step 5 完成 =========="
fi
