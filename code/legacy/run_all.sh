#!/bin/bash
# LEGACY / NOT PART OF THE NEW MAINLINE
# =============================================================================
# 【历史 / 演示脚本】非推荐主入口
#
# 本脚本为早期「单目录下顺序执行」示例：不设置 sh/run_step3_optimized.sh 等约定的
# D4C_CHECKPOINT_* / 嵌套 step5 目录，也不做 Step 4 的 --step3-subdir。
#
# DDP_NPROC：与 torchrun --nproc_per_node 一致；=1 为单卡 DDP smoke，仍为 DDP 主路径。
#
# 正式训练与评估请使用（在项目根）：
#   python code/d4c.py step3|step4|step5|pipeline …（MAINLINE ENTRY）
#   或 bash scripts/train_ddp.sh … / bash sh/run_step3_optimized.sh …（Shell 编排，内部 torchrun INTERNAL EXECUTOR）
#   见 docs/D4C_Scripts_and_Runtime_Guide.md
#   bash sh/run_step4_optimized.sh --step3-subdir <与 Step3 一致> …
#   bash sh/run_step5_optimized.sh --task N --step3-subdir <同上> …
# =============================================================================
#
# 脚本会将工作目录切换到 code/（与主线 Python 模块同级）；离线 HF
set -e
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DDP_NPROC="${DDP_NPROC:-1}"
if command -v torchrun >/dev/null 2>&1; then
    _TORCHRUN=(torchrun)
else
    _TORCHRUN=(python -m torch.distributed.run)
fi

echo "=== Step 1: 数据预处理 ==="
python preprocess_data.py
python split_data.py
python combine_data.py

echo "=== Step 2: 嵌入与域语义 ==="
python compute_embeddings.py
python infer_domain_semantics.py

echo "=== Step 3: 域对抗预训练 (8 个任务，torchrun DDP nproc=$DDP_NPROC) ==="
run_step3_pair() {
    "${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py train --auxiliary "$1" --target "$2" --epochs "$3"
    "${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py eval --auxiliary "$1" --target "$2"
}
run_step3_pair AM_Electronics AM_CDs 50
run_step3_pair AM_Movies AM_CDs 50
run_step3_pair AM_CDs AM_Electronics 50
run_step3_pair AM_Movies AM_Electronics 50
run_step3_pair AM_CDs AM_Movies 50
run_step3_pair AM_Electronics AM_Movies 50
run_step3_pair Yelp TripAdvisor 50
run_step3_pair TripAdvisor Yelp 50

echo "=== Step 4: 生成反事实数据 (torchrun DDP nproc=$DDP_NPROC) ==="
"${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" generate_counterfactual.py

echo "=== Step 5: 主训练 (以任务1为例, DDP nproc=$DDP_NPROC) ==="
"${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" run-d4c.py train --auxiliary AM_Electronics --target AM_CDs --epochs 50

echo "=== 完成 ==="
