#!/bin/bash
# -----------------------------------------------------------------------------
# Python 推荐入口: python code/d4c.py …（Step3+ 见主文档）；本脚本为 Step1+2 Shell 编排（非 torchrun 执行体）。
# 批量训练包装亦可: bash scripts/train_ddp.sh — 见 docs/D4C_Scripts_and_Runtime_Guide.md
# -----------------------------------------------------------------------------
# Step 1 + Step 2 合并脚本：数据预处理 + 嵌入与域语义
# 用法: bash run_step1_step2.sh [--embed-batch-size N] [--cuda-device N] [--daemon|--bg]
# 示例: bash run_step1_step2.sh
#       bash run_step1_step2.sh --embed-batch-size 512
#       bash run_step1_step2.sh --embed-batch-size 1024 --cuda-device 0
#       bash run_step1_step2.sh --daemon                            # 后台跑（日志写入 log/，终端仅打印 PID）
#
#   --daemon / --bg   后台运行：stdout/stderr 写入 log/step1_step2_*.log，终端立即返回
#
# ========== DDP 说明 ==========
# 本脚本只做预处理与嵌入，不使用 PyTorch DDP。域对抗预训练（Step 3）的 DDP 见同目录 run_step3_optimized.sh，例如：
#   DDP_NPROC=1 bash run_step3_optimized.sh --task 1
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash run_step3_optimized.sh --task 1 --batch-size 1024

set -e
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
SCRIPT_PATH="$SH_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$CODE_DIR"
LOG_DIR="$D4C_ROOT/log"
mkdir -p "$LOG_DIR"

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
    LOGFILE="$LOG_DIR/step1_step2_$(date +%Y%m%d_%H%M).log"
    ABS_LOG="$(readlink -f "$LOGFILE" 2>/dev/null || echo "$LOGFILE")"
    echo "已在后台启动 Step 1+2（数据预处理 + 嵌入与域语义），日志: $ABS_LOG"
    echo "查看进度: tail -f $ABS_LOG"
    nohup bash "$SCRIPT_PATH" _DAEMON_CHILD_ "$LOGFILE" "${FORWARD_ARGS[@]}" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

if [ -z "${INTERNAL_NOHUP:-}" ]; then
    LOGFILE="$LOG_DIR/step1_step2_$(date +%Y%m%d_%H%M).log"
fi
echo "启动 Step 1+2（数据预处理 + 嵌入与域语义），日志: $LOGFILE"
echo "查看进度: tail -f $LOGFILE"
echo "示例: $0 --embed-batch-size 512  # 显存不足时可减小"
echo "      $0 --embed-batch-size 1024 --cuda-device 0"

python run_preprocess_and_embed.py "${FORWARD_ARGS[@]}" 2>&1 | tee "$LOGFILE"
echo "========== Step 1+2 完成 =========="
