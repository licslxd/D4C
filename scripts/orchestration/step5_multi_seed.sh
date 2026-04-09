#!/usr/bin/env bash
# Step5 多 seed：Shell 编排（内部反复调用 scripts/entrypoints/step5.sh → d4c.py）。
# 单 seed 日常: python code/d4c.py step5 …（MAINLINE）。见 docs/D4C_Scripts_and_Runtime_Guide.md
#
# 内部反复调用 entrypoints/step5.sh；每次 **自动新 Step5 目录**，与 `d4c.py --step5-run auto` 一致。
#
# 用法（在项目根执行）:
#   bash scripts/orchestration/step5_multi_seed.sh --task N --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu [其它 Step5 选项…]
#   bash scripts/orchestration/step5_multi_seed.sh … --multi-seed-run 3
#
# 环境变量:
#   D4C_MULTI_SEEDS       空格分隔的种子列表，默认: 42 1024 2026 3407 8888
#   D4C_MULTI_SEED_LOGROOT  可选；覆盖 shell tee 根目录；默认 runs/task{N}/vN/meta/multi_seed/<run>/（与 train/step5/<run>/logs 独立）
#   D4C_MULTI_SEED_TEE=0    关闭按 seed 的 tee（仅 Python train.log）
#   D4C_MULTI_SEED_RUN_ID   可选；**仅显式 override**：须为 run slug（如 5、2_1），用作 meta/multi_seed 子目录名
#
# 路径边界：multi_seed 属单任务产物，**仅**落在 runs/task{T}/vN/meta/multi_seed/<run>/；禁止写入 runs/global/。
#
# 不支持在本包装器上传 --daemon / --bg；需要后台请对每个 seed 单独调 entrypoints/step5.sh。
#
# 论文表格: 各 seed 对应不同 `runs/.../train/step5/<run>/logs/train.log`；shell tee 与 stats 亦可指向 meta/multi_seed/<run>/:
#   python scripts/multi_seed_paper_stats.py --logs 'runs/task4/v1/meta/multi_seed/1/train_seed_*.log'
#
# 重复惩罚（解码，logit 域）: 可调整 get_underlying_model(m).repetition_penalty（>1.0，如 1.1～1.25）做 BLEU 网格搜；
# 另有 generate_temperature / generate_top_p（Step5 模型解码侧）。

set -euo pipefail

SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "$SH_DIR/../.." && pwd)"
STEP5_SCRIPT="$D4C_ROOT/scripts/entrypoints/step5.sh"

# 从 "$@" 去掉 --multi-seed-run <name>，其余原样转给 scripts/entrypoints/step5.sh
FORWARD_ARGS=()
MS_SEED_RUN_CLI=""
_prev_ms=""
for _a in "$@"; do
    if [ "$_prev_ms" = "--multi-seed-run" ]; then
        MS_SEED_RUN_CLI="$_a"
        _prev_ms=""
        continue
    fi
    if [ "$_a" = "--multi-seed-run" ]; then
        _prev_ms="--multi-seed-run"
        continue
    fi
    FORWARD_ARGS+=("$_a")
done

MS_TASK=""
MS_ITER="${D4C_ITER:-v1}"
_prev_ms=""
for _a in "${FORWARD_ARGS[@]}"; do
    if [ "$_prev_ms" = "--task" ] && [[ "$_a" =~ ^[1-8]$ ]]; then
        MS_TASK="$_a"
        _prev_ms=""
    elif [ "$_a" = "--task" ]; then
        _prev_ms="--task"
    elif [ "$_prev_ms" = "--iter" ]; then
        MS_ITER="$_a"
        export D4C_ITER="$_a"
        _prev_ms=""
    elif [ "$_a" = "--iter" ]; then
        _prev_ms="--iter"
    else
        _prev_ms=""
    fi
done

for a in "${FORWARD_ARGS[@]}"; do
    if [ "$a" = "--daemon" ] || [ "$a" = "--bg" ]; then
        echo "错误: scripts/orchestration/step5_multi_seed.sh 不支持 --daemon/--bg。请前台串行，或对每个 seed 单独调用 scripts/entrypoints/step5.sh。" >&2
        exit 2
    fi
done

if [ ${#FORWARD_ARGS[@]} -lt 1 ]; then
    echo "用法: bash scripts/orchestration/step5_multi_seed.sh --task N --iter v1 --from-run <run> --eval-profile <stem> [--multi-seed-run 3] …" >&2
    exit 1
fi

# shellcheck disable=SC2086
SEEDS_STR="${D4C_MULTI_SEEDS:-42 1024 2026 3407 8888}"

if [ -n "${D4C_MULTI_SEED_LOGROOT:-}" ]; then
    LOGROOT="$D4C_MULTI_SEED_LOGROOT"
elif [[ "$MS_TASK" =~ ^[1-8]$ ]]; then
    _explicit="${MS_SEED_RUN_CLI:-}"
    if [ -z "$_explicit" ] && [ -n "${D4C_MULTI_SEED_RUN_ID:-}" ]; then
        _explicit="${D4C_MULTI_SEED_RUN_ID}"
    fi
    _RUN_ID="$(cd "${D4C_ROOT}/code" && PYTHONPATH=. \
        D4C_ROOT_VAL="$D4C_ROOT" D4C_TID_VAL="$MS_TASK" D4C_ITER_VAL="$MS_ITER" \
        D4C_MS_EXPLICIT="${_explicit}" python -c "
import os
from pathlib import Path
from d4c_core import path_layout, run_naming
root = Path(os.environ['D4C_ROOT_VAL']).resolve()
tid = int(os.environ['D4C_TID_VAL'])
it = os.environ['D4C_ITER_VAL']
ex = os.environ.get('D4C_MS_EXPLICIT', '').strip()
parent = path_layout.get_task_meta_dir(root, tid, it) / 'multi_seed'
if ex:
    rid = run_naming.parse_run_id(ex)
else:
    parent.mkdir(parents=True, exist_ok=True)
    rid = run_naming.next_run_id(parent)
print(rid)
")"
    LOGROOT="${D4C_ROOT}/runs/task${MS_TASK}/${MS_ITER}/meta/multi_seed/${_RUN_ID}"
else
    echo "错误: 未设置 D4C_MULTI_SEED_LOGROOT 时必须在参数中提供 --task N（1–8）。" >&2
    exit 2
fi
mkdir -p "$LOGROOT"

echo "run_step5_multi_seed: shell tee 根目录: $LOGROOT （默认子目录为 1、2、…；复用可 --multi-seed-run / D4C_MULTI_SEED_RUN_ID）"
echo "run_step5_multi_seed: seeds=$SEEDS_STR"
echo "run_step5_multi_seed: shell tee 目录: $LOGROOT"

for s in $SEEDS_STR; do
    echo "========== seed=$s（Step5 自动分配下一目录名）=========="
    if [ "${D4C_MULTI_SEED_TEE:-1}" != "0" ]; then
        bash "$STEP5_SCRIPT" "${FORWARD_ARGS[@]}" --seed "$s" 2>&1 | tee "$LOGROOT/train_seed_${s}.log"
    else
        bash "$STEP5_SCRIPT" "${FORWARD_ARGS[@]}" --seed "$s"
    fi
done

echo "========== 全部 seed 完成 =========="
echo "shell 日志: $LOGROOT/train_seed_*.log"
echo "论文 mean±std: python $D4C_ROOT/scripts/multi_seed_paper_stats.py --logs $LOGROOT/train_seed_*.log"
