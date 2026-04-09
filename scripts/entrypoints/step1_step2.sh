#!/bin/bash
# -----------------------------------------------------------------------------
# Python 推荐入口: python code/d4c.py …（Step3+ 见主文档）；本脚本为 Step1+2 Shell 编排（非 torchrun 执行体）。
# 批量训练包装: bash scripts/entrypoints/train_ddp.sh — 见 docs/D4C_Scripts_and_Runtime_Guide.md
# -----------------------------------------------------------------------------
# Step 1 + Step 2 合并脚本：数据预处理 + 嵌入与域语义
# 用法: bash run_step1_step2.sh [--embed-batch-size N] [--cuda-device N] [--daemon|--bg]
# 示例: bash run_step1_step2.sh
#       bash run_step1_step2.sh --embed-batch-size 512
#       bash run_step1_step2.sh --embed-batch-size 1024 --cuda-device 0
#       bash run_step1_step2.sh --daemon                            # 后台跑（shell 汇总写入 runs/global/…/meta/shell_logs/）
#
#   --daemon / --bg   后台运行：stdout/stderr 写入 runs/global/vN/meta/shell_logs/step1_step2_*.log
#
# ========== DDP 说明 ==========
# 本脚本只做预处理与嵌入，不使用 PyTorch DDP。域对抗预训练（Step 3）见 scripts/entrypoints/step3.sh。

set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/../.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=../lib/common_logs.sh
source "$SH_DIR/../lib/common_logs.sh"
export D4C_ITER="${D4C_ITER:-v1}"
cd "$CODE_DIR"

if [ "${1:-}" = "_DAEMON_CHILD_" ]; then
    shift
    LOGFILE="$1"
    shift
    INTERNAL_NOHUP=1
fi

FORWARD_ARGS=()
DAEMON=""
for a in "$@"; do
    if [ "$a" = "--daemon" ] || [ "$a" = "--bg" ]; then DAEMON=1; continue; fi
    FORWARD_ARGS+=("$a")
done

if [ -n "$DAEMON" ] && [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$(d4c_shell_logs_global "$D4C_ROOT")/step1_step2_$(date +%Y%m%d_%H%M).log"
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    echo "已在后台启动 Step 1+2（数据预处理 + 嵌入与域语义），日志: $ABS_LOG"
    echo "查看进度: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${FORWARD_ARGS[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

if [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$(d4c_shell_logs_global "$D4C_ROOT")/step1_step2_$(date +%Y%m%d_%H%M).log"
fi
echo "启动 Step 1+2（数据预处理 + 嵌入与域语义），日志: $LOGFILE"
echo "查看进度: tail -f $LOGFILE"
echo "示例: $0 --embed-batch-size 512  # 显存不足时可减小"
echo "      $0 --embed-batch-size 1024 --cuda-device 0"

python run_preprocess_and_embed.py "${FORWARD_ARGS[@]}" 2>&1 | tee "$LOGFILE"
echo "========== Step 1+2 完成 =========="
