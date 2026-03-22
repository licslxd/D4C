#!/bin/bash
# D4C 离线完整复现脚本
# 在 D4C-main 或 D4C-main/code 下运行
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DDP_NPROC="${DDP_NPROC:-1}"

echo "=== Step 1: 数据预处理 ==="
python preprocess_data.py
python split_data.py
python combine_data.py

echo "=== Step 2: 嵌入与域语义 ==="
python compute_embeddings.py
python infer_domain_semantics.py

echo "=== Step 3: 域对抗预训练 (8 个任务，DDP nproc=$DDP_NPROC) ==="
run_step3_pair() {
    torchrun --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py train --auxiliary "$1" --target "$2" --epochs "$3"
    torchrun --standalone --nproc_per_node="$DDP_NPROC" AdvTrain.py eval --auxiliary "$1" --target "$2"
}
run_step3_pair AM_Electronics AM_CDs 50
run_step3_pair AM_Movies AM_CDs 50
run_step3_pair AM_CDs AM_Electronics 50
run_step3_pair AM_Movies AM_Electronics 50
run_step3_pair AM_CDs AM_Movies 50
run_step3_pair AM_Electronics AM_Movies 50
run_step3_pair Yelp TripAdvisor 50
run_step3_pair TripAdvisor Yelp 50

echo "=== Step 4: 生成反事实数据 ==="
python generate_counterfactual.py

echo "=== Step 5: 主训练 (以任务1为例, DDP nproc=$DDP_NPROC) ==="
torchrun --standalone --nproc_per_node="$DDP_NPROC" run-d4c.py --auxiliary AM_Electronics --target AM_CDs --epochs 50

echo "=== 完成 ==="
