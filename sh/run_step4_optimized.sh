#!/bin/bash
#
# run_step4_optimized.sh — Step 4 工程优化入口（与论文复现 run_step4.sh 隔离）
#
# 从 run_step3_optimized.sh 产出的权重生成反事实时，须使用与当时一致的 checkpoint 命名空间：
#   checkpoints/<task>/step3_optimized/<step3_opt_id>/model.pth
#
# 本脚本在调用 run_step4.sh 前设置：
#   D4C_CHECKPOINT_GROUP=step3_optimized
#   D4C_CHECKPOINT_SUBDIR=<--step3-subdir 参数值>
#   D4C_LOG_STEP=step4_optimized（若未预先设置 D4C_LOG_STEP / D4C_LOG_GROUP 等）
#   D4C_TRAIN_MODE=optimized（与工程管线一致；generate_counterfactual 主要读 checkpoint 路径）
#
# 用法与 run_step4.sh 相同，但须额外传入：
#   --step3-subdir NAME   与 checkpoints/<task>/step3_optimized/<NAME>/ 下目录名一致（例如 step3_opt_20260324_1400）
#
# 示例：
#   bash sh/run_step4_optimized.sh --task 2 --step3-subdir step3_opt_20260324_1400
#   CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 bash sh/run_step4_optimized.sh --all --step3-subdir step3_opt_20260324_1400
#
# 未传 --batch-size 时默认追加 config.get_train_batch_size()（可用 D4C_OPT_BATCH_SIZE 覆盖）。
#
# ---------------------------------------------------------------------------

set -euo pipefail
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"

new_args=()
STEP3_SUBDIR=""
prev=""
for i in "$@"; do
    if [ "$prev" = "--step3-subdir" ]; then
        STEP3_SUBDIR="$i"
        prev=""
        continue
    fi
    if [ "$i" = "--step3-subdir" ]; then
        prev="--step3-subdir"
        continue
    fi
    new_args+=("$i")
done
set -- "${new_args[@]}"

if [ -z "$STEP3_SUBDIR" ]; then
    echo "错误: 须指定 --step3-subdir <NAME>，与 checkpoints/<task>/step3_optimized/<NAME>/ 一致（run_step3_optimized 产物）。"
    echo "  例: bash sh/run_step4_optimized.sh --task 2 --step3-subdir step3_opt_20260324_1400"
    exit 1
fi

export D4C_TRAIN_MODE=optimized
export D4C_CHECKPOINT_GROUP="${D4C_CHECKPOINT_GROUP:-step3_optimized}"
export D4C_CHECKPOINT_SUBDIR="$STEP3_SUBDIR"

# 与复现 step4 的 log/<task>/step4/ 区分；若已 export D4C_LOG_GROUP 等则勿覆盖（用户自管）
if [ -z "${D4C_LOG_GROUP:-}" ] && [ -z "${D4C_LOG_SUBDIR:-}" ]; then
    export D4C_LOG_STEP="${D4C_LOG_STEP:-step4_optimized}"
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

echo "[run_step4_optimized] D4C_CHECKPOINT_GROUP=$D4C_CHECKPOINT_GROUP D4C_CHECKPOINT_SUBDIR=$D4C_CHECKPOINT_SUBDIR D4C_LOG_STEP=${D4C_LOG_STEP:-}"

exec bash "$SH_DIR/run_step4.sh" "$@" "${_extra[@]}"
