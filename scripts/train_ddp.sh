#!/usr/bin/env bash
# D4C — Bash 包装层：fail-fast 校验后派发 sh/run_step*_optimized.sh（非 Python 主入口）
# Python MAINLINE ENTRY: python code/d4c.py step3|step4|step5|eval|pipeline …（项目根）
# 实现见 scripts/train_lib.sh；规范见 docs/D4C_Scripts_and_Runtime_Guide.md
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
export D4C_ROOT

# shellcheck source=train_lib.sh
source "${_SCRIPT_DIR}/train_lib.sh"

d4c_train_main "$D4C_ROOT" "$@"
