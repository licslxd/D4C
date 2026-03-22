#!/bin/bash
# 后台顺序运行 Task 2-8（单 GPU 需串行），支持断点续跑
# 用法: bash run_remaining_tasks.sh [--skip-prep] [--from N]
# 示例: bash run_remaining_tasks.sh --skip-prep
#       bash run_remaining_tasks.sh --skip-prep --from 6   # 从 Task 6 续跑（跳过已完成的 2-5）

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"
LOG_DIR="${CODE_DIR}/../log"
mkdir -p "$LOG_DIR"

PREP=""
FROM_TASK=2
prev=""
for arg in "$@"; do
    if [ "$arg" = "--skip-prep" ]; then
        PREP="--skip-prep"
    elif [ "$prev" = "--from" ] && [[ "$arg" =~ ^[2-8]$ ]]; then
        FROM_TASK=$arg
    fi
    prev="$arg"
done

# 生成任务列表
TASK_LIST=""
for i in 2 3 4 5 6 7 8; do
    [ $i -ge $FROM_TASK ] && TASK_LIST="$TASK_LIST $i"
done
TASK_LIST=$(echo $TASK_LIST | xargs)
[ -z "$TASK_LIST" ] && { echo "错误: --from $FROM_TASK 无有效任务"; exit 1; }

LOGFILE="$LOG_DIR/remaining_tasks_$(date +%Y%m%d_%H%M).log"
echo "启动 Task $TASK_LIST（顺序执行），日志: $LOGFILE"
echo "查看进度: tail -f $LOGFILE"

nohup bash -c "
for i in $TASK_LIST; do
    echo \"========== 开始 Task \$i ==========\"
    if bash run_single_task_d4c.sh $PREP \$i; then
        echo \"========== Task \$i 完成 ==========\"
    else
        rc=\$?
        echo \"========== Task \$i 失败 (退出码 \$rc) ==========\"
        echo \"请检查日志，修复后可用: bash run_remaining_tasks.sh $PREP --from \$i\"
        exit \$rc
    fi
done
echo \"========== 全部任务完成 ==========\"
" > "$LOGFILE" 2>&1 &

echo "PID: $!"
