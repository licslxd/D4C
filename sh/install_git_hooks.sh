#!/usr/bin/env bash
# 将 Git hooks 目录指向仓库内 .githooks（含 pre-commit 文档同步检查）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git config core.hooksPath .githooks
echo "已设置 core.hooksPath=.githooks（仓库根: $ROOT）"
