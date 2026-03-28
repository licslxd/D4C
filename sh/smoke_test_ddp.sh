#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 兼容说明：正式 Step 3–5 推荐 bash scripts/train_ddp.sh（见 docs/D4C_RUNTIME_SPEC.md）；本脚本为 DDP 冒烟专用。
# -----------------------------------------------------------------------------
# =============================================================================
# DDP smoke test — NOT a quality / metrics benchmark.
#
# Runs the official torchrun + DDP entrypoints with tiny settings to verify
# the stack does not crash. nproc_per_node=1 still uses init_process_group and
# DDP; it is NOT a separate non-distributed code path.
#
# Prerequisites: full offline data layout (Merged_data/1, data/*, pretrained_models, …).
# Outputs: checkpoints/1/smoke_ddp/<SMOKE_TAG>/ and log/1/smoke_ddp/ — safe to rm after test.
#
# Usage (from repo root):
#   bash sh/smoke_test_ddp.sh
#   CUDA_VISIBLE_DEVICES=0 bash sh/smoke_test_ddp.sh
# =============================================================================
set -euo pipefail

SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
CODE_DIR="$D4C_ROOT/code"
cd "$CODE_DIR"

_MERGED="$D4C_ROOT/Merged_data/1/aug_train.csv"
if [[ ! -f "$_MERGED" ]]; then
    echo "错误: 缺少 smoke 前置数据: $_MERGED（需先完成 Step1+2 与 combine）" >&2
    exit 1
fi

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python -m torch.distributed.run)
    echo "提示: 使用 ${TORCHRUN[*]}（未找到 torchrun 可执行文件）"
fi

SMOKE_TAG="smoke_$(date +%Y%m%d_%H%M%S)_$$"
export D4C_CHECKPOINT_GROUP="smoke_ddp"
export D4C_CHECKPOINT_SUBDIR="${SMOKE_TAG}"
export D4C_LOG_GROUP="smoke_ddp"
unset D4C_LOG_STEP 2>/dev/null || true
unset D4C_LOG_SUBDIR 2>/dev/null || true

# 减轻 Step 5 单 epoch 内的 valid 采样（不改变算法，仅缩子集）
export D4C_QUICK_EVAL_MAX_SAMPLES="${D4C_QUICK_EVAL_MAX_SAMPLES:-32}"

RUN_DIR="$D4C_ROOT/log/1/smoke_ddp/runs/${SMOKE_TAG}"
mkdir -p "$RUN_DIR"
LOG_TRAIN="$RUN_DIR/step3_train.log"
LOG_EVAL="$RUN_DIR/step3_eval.log"
LOG_S4="$RUN_DIR/step4.log"
LOG_S5="$RUN_DIR/step5.log"

AUX="AM_Electronics"
TGT="AM_CDs"
BS=8
NPC=2
# Step3 eval / Step4 依赖 task1 的 model.pth；--max-steps 会在 epoch 内提前 return，不会走到按 valid/BLEU 写盘逻辑
SMOKE_CKPT="$D4C_ROOT/checkpoints/1/smoke_ddp/${SMOKE_TAG}/model.pth"

echo "========== DDP smoke: CHECKPOINT_SUBDIR=$SMOKE_TAG =========="

echo ">>> Step 3 train (max-steps=2, global batch=$BS, --save-final-checkpoint)"
"${TORCHRUN[@]}" --standalone --nproc_per_node=1 AdvTrain.py train \
    --auxiliary "$AUX" --target "$TGT" \
    --max-steps 2 \
    --save-final-checkpoint \
    --batch-size "$BS" \
    --num-proc "$NPC" \
    --log_file "$LOG_TRAIN"

if [[ ! -f "$SMOKE_CKPT" ]]; then
    echo "error: current smoke train config did not produce model.pth" >&2
    echo "  expected: $SMOKE_CKPT" >&2
    echo "  hint: Step3 train only saves on epoch-end metric improvement unless --save-final-checkpoint is used." >&2
    exit 1
fi

echo ">>> Step 3 eval (global batch=$BS)"
"${TORCHRUN[@]}" --standalone --nproc_per_node=1 AdvTrain.py eval \
    --auxiliary "$AUX" --target "$TGT" \
    --batch-size "$BS" \
    --num-proc "$NPC" \
    --log_file "$LOG_EVAL"

echo ">>> Step 4 generate_counterfactual (task 1, global batch=$BS)"
"${TORCHRUN[@]}" --standalone --nproc_per_node=1 generate_counterfactual.py \
    --task 1 \
    --batch-size "$BS" \
    --num-proc "$NPC" \
    --log_file "$LOG_S4"

echo ">>> Step 5 run-d4c (epochs=1, train-only, global batch=32)"
"${TORCHRUN[@]}" --standalone --nproc_per_node=1 run-d4c.py \
    --auxiliary "$AUX" --target "$TGT" \
    --epochs 1 \
    --train-only \
    --batch-size 32 \
    --num-proc "$NPC" \
    --log_file "$LOG_S5"

echo "========== DDP smoke 完成（仅验证不 crash） =========="
echo "产物: $D4C_ROOT/checkpoints/1/smoke_ddp/${SMOKE_TAG}/"
echo "日志: $RUN_DIR/"
echo "可手动删除上述目录以清理。"
