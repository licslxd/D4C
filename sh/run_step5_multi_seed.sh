#!/usr/bin/env bash
# Step5 多 seed：Shell 编排（内部反复调用 run_step5_optimized.sh → d4c.py 分发 step5 runner）。
# 单 seed 日常: python code/d4c.py step5 …（MAINLINE）。见 docs/D4C_Scripts_and_Runtime_Guide.md
#
# run_step5_multi_seed.sh — 多随机种子串行跑 Step 5（每次独立嵌套目录 + --seed）
#
# 内部反复调用 run_step5_optimized.sh；与直接 d4c.py step5 等价，但沿用 checkpoint 软链、
# 日志根目录（log/<task>/step5_optimized/）及 eval 汇总（eval_runs.jsonl 等）。
#
# 用法（在项目根执行）:
#   bash sh/run_step5_multi_seed.sh --task N --step3-subdir step3_opt_YYYYMMDD_HHMM [其它 Step5 选项…]
#   bash scripts/run_step5_multi_seed.sh …   # 与上一行等价（scripts 薄封装）
#
# 环境变量:
#   D4C_MULTI_SEEDS       空格分隔的种子列表，默认: 42 1024 2026 3407 8888
#   D4C_MULTI_SEED_LOGROOT  可选；shell tee 汇总目录，默认 log/multi_seed_runs/<本批RUN_ID>/
#   D4C_MULTI_SEED_TEE=0    关闭按 seed 的 tee（仅 Python train.log）
#   D4C_MULTI_SEED_RUN_ID   可选；固定嵌套目录后缀，便于 --eval-only 复跑同一批 checkpoint。
#                             嵌套名: step5_opt_ms_<RUN_ID>_seed<seed>
#
# 不支持在本包装器上传 --daemon / --bg（多 job 并行会乱）；需要后台请对每个 seed 单独调 run_step5_optimized.sh。
#
# 论文表格: 各 seed 跑完后，train.log 或 log/<task>/step5_optimized/eval/eval_runs.jsonl 中有
#   MAE、RMSE、BLEU-4、ROUGE-L。汇总:
#   python scripts/multi_seed_paper_stats.py --logs log/multi_seed_runs/<RUN_ID>/train_seed_*.log
#   或 python scripts/multi_seed_paper_stats.py --jsonl log/<task>/step5_optimized/eval/eval_runs.jsonl --last-n 5
#
# 重复惩罚（解码，logit 域）: 可调整 get_underlying_model(m).repetition_penalty（>1.0，如 1.1～1.25）做 BLEU 网格搜；
# 另有 generate_temperature / generate_top_p（Step5 模型解码侧）。

set -euo pipefail

SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"
STEP5_SCRIPT="$SH_DIR/run_step5_optimized.sh"

for a in "$@"; do
    if [ "$a" = "--daemon" ] || [ "$a" = "--bg" ]; then
        echo "错误: run_step5_multi_seed.sh 不支持 --daemon/--bg。请前台串行，或对每个 seed 单独调用 run_step5_optimized.sh。" >&2
        exit 2
    fi
done

if [ $# -lt 1 ]; then
    echo "用法: bash sh/run_step5_multi_seed.sh --task N --step3-subdir step3_opt_… [其它 run_step5_optimized 选项…]" >&2
    echo "      或 bash scripts/run_step5_multi_seed.sh …" >&2
    exit 1
fi

# shellcheck disable=SC2086
SEEDS_STR="${D4C_MULTI_SEEDS:-42 1024 2026 3407 8888}"
_RUN_ID="${D4C_MULTI_SEED_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOGROOT="${D4C_MULTI_SEED_LOGROOT:-$D4C_ROOT/log/multi_seed_runs/$_RUN_ID}"
mkdir -p "$LOGROOT"

echo "run_step5_multi_seed: RUN_ID=$_RUN_ID （eval-only 复跑请 export D4C_MULTI_SEED_RUN_ID=$_RUN_ID）"
echo "run_step5_multi_seed: seeds=$SEEDS_STR"
echo "run_step5_multi_seed: shell tee 目录: $LOGROOT"

for s in $SEEDS_STR; do
    _inner="step5_opt_ms_${_RUN_ID}_seed${s}"
    echo "========== seed=$s  nested-subdir=$_inner =========="
    if [ "${D4C_MULTI_SEED_TEE:-1}" != "0" ]; then
        bash "$STEP5_SCRIPT" "$@" --nested-subdir "$_inner" --seed "$s" 2>&1 | tee "$LOGROOT/train_seed_${s}.log"
    else
        bash "$STEP5_SCRIPT" "$@" --nested-subdir "$_inner" --seed "$s"
    fi
done

echo "========== 全部 seed 完成 =========="
echo "shell 日志: $LOGROOT/train_seed_*.log"
echo "论文 mean±std: python $D4C_ROOT/scripts/multi_seed_paper_stats.py --logs $LOGROOT/train_seed_*.log"
