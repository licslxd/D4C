#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LEGACY — 迁仓前 CSV 表头修补；主线产物在 runs/.../meta/，见 legacy/README.md。
"""
将 eval 注册表 CSV（eval_registry.csv / eval_registry_all.csv）首行表头替换为
train_logging._EVAL_REGISTRY_CSV_FIELDS（最后一列应为 eval_elapsed_s）。

用法（在项目根目录）:
  python legacy/tools/fix_eval_csv_header.py
  python legacy/tools/fix_eval_csv_header.py runs/task4/v1/meta/eval_registry.csv
  python legacy/tools/fix_eval_csv_header.py --dry-run

默认递归扫描 <项目根>/runs/ 下文件名匹配的 CSV；仅修补迁仓库前旧数据时用 --legacy-only
扫描 log/ 下的 eval_runs*.csv（legacy_only，不再作为主线默认）。
"""
from __future__ import annotations

import argparse
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TOOLS_DIR, "..", ".."))
_CODE_DIR = os.path.join(_REPO_ROOT, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from train_logging import _EVAL_REGISTRY_CSV_FIELDS  # noqa: E402

from paths_config import get_d4c_root  # noqa: E402

_LEGACY_NAMES = frozenset({"eval_runs.csv", "eval_runs_all.csv"})
_REGISTRY_NAMES = frozenset({"eval_registry.csv", "eval_registry_all.csv"})


def _scan_runs_registry_csvs() -> list[str]:
    runs_root = os.path.join(get_d4c_root(), "runs")
    out: list[str] = []
    if not os.path.isdir(runs_root):
        return out
    for dirpath, _dirnames, filenames in os.walk(runs_root):
        for name in filenames:
            if name in _REGISTRY_NAMES:
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _scan_log_legacy_csvs() -> list[str]:
    log_root = os.path.join(get_d4c_root(), "log")
    out: list[str] = []
    if not os.path.isdir(log_root):
        return out
    for dirpath, _dirnames, filenames in os.walk(log_root):
        for name in filenames:
            if name in _LEGACY_NAMES:
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def fix_file(path: str, *, dry_run: bool) -> bool:
    header_line = ",".join(_EVAL_REGISTRY_CSV_FIELDS) + "\n"
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        print(f"跳过（空文件）: {path}")
        return False
    old = lines[0].rstrip("\r\n")
    new = header_line.rstrip("\n")
    if old == new.rstrip("\n"):
        print(f"无需修改: {path}")
        return False
    if dry_run:
        print(f"[dry-run] 将修改: {path}")
        print(f"  旧表头: {old[:120]}{'...' if len(old) > 120 else ''}")
        print(f"  新表头: {new}")
        return True
    lines[0] = header_line
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
    print(f"已更新表头: {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="修正 eval 注册表 CSV 首行表头（legacy_only：仅扫 log/ 下旧 eval_runs*.csv）")
    ap.add_argument(
        "paths",
        nargs="*",
        help="CSV 路径；省略则默认扫描 runs/ 下 eval_registry*.csv",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印将要修改的文件，不写回")
    ap.add_argument(
        "--legacy-only",
        action="store_true",
        help="legacy_only：只扫描 <项目根>/log/ 下 eval_runs.csv / eval_runs_all.csv（迁仓库前旧数据）",
    )
    args = ap.parse_args()
    if args.paths:
        paths = args.paths
    elif args.legacy_only:
        paths = _scan_log_legacy_csvs()
    else:
        paths = _scan_runs_registry_csvs()
    if not paths:
        src = "log/（--legacy-only）" if args.legacy_only else "runs/"
        print(f"未找到任何待处理 CSV（已查 {src}）", file=sys.stderr)
        return 1
    n = 0
    for p in paths:
        if not os.path.isfile(p):
            print(f"跳过（不存在）: {p}", file=sys.stderr)
            continue
        if fix_file(p, dry_run=args.dry_run):
            n += 1
    if args.dry_run:
        print(f"共 {n} 个文件需要修改（未写回）")
    else:
        print(f"完成，已修改 {n} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
