#!/usr/bin/env bash
# Step5 权重评测：透传至 python code/d4c.py eval（业务语义仅来自 CLI / preset）。
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4C_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
export D4C_ROOT
cd "$D4C_ROOT"
exec python "$D4C_ROOT/code/d4c.py" eval "$@"
