#!/usr/bin/env bash
# D4C — 统一 DDP 训练入口（第二阶段：可执行 + fail-fast 校验）
# 实现见 scripts/train_lib.sh；规范见 docs/D4C_RUNTIME_SPEC.md
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
export D4C_ROOT

# shellcheck source=train_lib.sh
source "${_SCRIPT_DIR}/train_lib.sh"

d4c_train_main "$D4C_ROOT" "$@"
