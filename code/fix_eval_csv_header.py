#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 eval_runs*.csv / eval_runs_all.csv 的第一行表头替换为 train_logging._EVAL_SUMMARY_CSV_FIELDS
（最后一列应为 eval_elapsed_s，而非历史误写的 bert）。

用法（在项目根目录）:
  python code/fix_eval_csv_header.py
  python code/fix_eval_csv_header.py log/step3/eval/eval_runs_all.csv log/step5/eval/eval_runs_all.csv
  python code/fix_eval_csv_header.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from train_logging import _EVAL_SUMMARY_CSV_FIELDS  # noqa: E402

from paths_config import D4C_ROOT  # noqa: E402


def _default_paths() -> list[str]:
    log_root = os.path.join(D4C_ROOT, "log")
    out: list[str] = []
    if not os.path.isdir(log_root):
        return out
    for dirpath, _dirnames, filenames in os.walk(log_root):
        for name in filenames:
            if name == "eval_runs.csv" or name == "eval_runs_all.csv":
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def fix_file(path: str, *, dry_run: bool) -> bool:
    header_line = ",".join(_EVAL_SUMMARY_CSV_FIELDS) + "\n"
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
    ap = argparse.ArgumentParser(description="修正 eval 汇总 CSV 首行表头")
    ap.add_argument(
        "paths",
        nargs="*",
        help="CSV 路径；省略则扫描 <项目根>/log 下所有 eval_runs.csv / eval_runs_all.csv",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印将要修改的文件，不写回")
    args = ap.parse_args()
    paths = args.paths if args.paths else _default_paths()
    if not paths:
        print("未找到任何 eval_runs*.csv", file=sys.stderr)
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
