#!/usr/bin/env bash
# D4C — Shell 编排：GPU/CVD/DDP 校验后调用 python code/d4c.py
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
export D4C_ROOT

# shellcheck source=../lib/train_lib.sh
source "${_SCRIPT_DIR}/../lib/train_lib.sh"

d4c_train_main "$D4C_ROOT" "$@"
