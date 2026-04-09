#!/bin/bash
# LEGACY / NOT PART OF THE NEW MAINLINE
# =============================================================================
# 【历史 / 演示脚本】非推荐主入口
#
# 本脚本为早期「单目录下顺序执行」示例：不设置 scripts/entrypoints/step3.sh 等约定的
# 历史演示：不经过新版 runs/ 与 d4c.py 主链路约定。
#
# DDP_NPROC：与 torchrun --nproc_per_node 一致；=1 为单卡 DDP smoke，仍为 DDP 主路径。
#
# 正式训练与评估请使用（在项目根）：
#   python code/d4c.py step3|step4|step5|pipeline …（MAINLINE ENTRY）
#   或 bash scripts/entrypoints/train_ddp.sh … / bash scripts/entrypoints/step3.sh …（Shell 编排，内部 torchrun INTERNAL EXECUTOR）
#   见 docs/D4C_Scripts_and_Runtime_Guide.md
#   bash scripts/entrypoints/step4.sh --iter v1 --from-run <run> …
#   bash scripts/entrypoints/step5.sh --task N --iter v1 --from-run <run> …
# =============================================================================
#
# 脚本会将工作目录切换到 code/（与主线 Python 模块同级）；本文件位于 legacy/code/（考古）
set -e
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../code" && pwd)"
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
    "${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" executors/step3_entry.py train --auxiliary "$1" --target "$2" --epochs "$3"
    "${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" executors/step3_entry.py eval --auxiliary "$1" --target "$2"
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
"${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" executors/step4_entry.py

echo "=== Step 5: 主训练 (以任务1为例, DDP nproc=$DDP_NPROC) ==="
"${_TORCHRUN[@]}" --standalone --nproc_per_node="$DDP_NPROC" executors/step5_entry.py train --auxiliary AM_Electronics --target AM_CDs --epochs 50

echo "=== 完成 ==="
