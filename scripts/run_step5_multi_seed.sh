#!/usr/bin/env bash
# 与 sh/run_step5_multi_seed.sh 等价；便于与 scripts/train_ddp.sh 同一入口习惯。
# 见 sh/run_step5_multi_seed.sh 文件头；论文汇总: python scripts/multi_seed_paper_stats.py --help
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
exec bash "${D4C_ROOT}/sh/run_step5_multi_seed.sh" "$@"
