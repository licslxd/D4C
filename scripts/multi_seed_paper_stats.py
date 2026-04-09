#!/usr/bin/env python3
"""从 Step 5 的 train.log（FINAL RESULTS）或 eval 汇总 JSONL 汇总 MAE、RMSE、BLEU-4、ROUGE-L。

论文常用：n 次独立试验 (x_1,…,x_n)，报告均值 bar{x} 与样本标准差
s = sqrt(1/(n-1) * sum (x_i - bar{x})^2)，表中写作 mean ± s。
n=5 时可注明「5 independent runs」；更稳可增加 seed 数或汇报 bootstrap 区间。

示例:
  python scripts/multi_seed_paper_stats.py --logs 'runs/task4/v1/train/step5/run*/logs/train.log'
  python scripts/multi_seed_paper_stats.py --logs 'runs/task4/v1/meta/multi_seed/1/train_seed_*.log'
  python scripts/multi_seed_paper_stats.py --jsonl runs/task4/v1/meta/eval_registry.jsonl --last-n 5
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MAE_RMSE_RE = re.compile(
    r"MAE\s*=\s*([0-9.+-eE]+)\s*\|\s*RMSE\s*=\s*([0-9.+-eE]+)"
)
ROUGE_RE = re.compile(r"ROUGE:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)")
BLEU_RE = re.compile(
    r"BLEU:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
)

METRIC_KEYS = ("mae", "rmse", "bleu_4", "rouge_l")


def _last_float_pair(text: str, pattern: re.Pattern[str], groups: Tuple[int, ...]) -> Tuple[float, ...]:
    ms = list(pattern.finditer(text))
    if not ms:
        raise ValueError(f"未匹配到模式 {pattern.pattern!r}")
    m = ms[-1]
    return tuple(float(m.group(i)) for i in groups)


def parse_train_log(path: str) -> Dict[str, float]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    mae, rmse = _last_float_pair(text, MAE_RMSE_RE, (1, 2))
    _r1, _r2, rouge_l = _last_float_pair(text, ROUGE_RE, (1, 2, 3))
    _b1, _b2, _b3, bleu_4 = _last_float_pair(text, BLEU_RE, (1, 2, 3, 4))
    return {
        "mae": mae,
        "rmse": rmse,
        "bleu_4": bleu_4,
        "rouge_l": rouge_l,
    }


def load_jsonl_metrics(
    path: str,
    *,
    save_file_contains: Optional[str] = None,
    last_n: Optional[int] = None,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if save_file_contains and save_file_contains not in str(o.get("save_file", "")):
            continue
        rows.append(o)
    if last_n is not None:
        rows = rows[-last_n:]
    out: List[Dict[str, float]] = []
    for o in rows:
        out.append(
            {
                "mae": float(o["mae"]),
                "rmse": float(o["rmse"]),
                "bleu_4": float(o["bleu_4"]),
                "rouge_l": float(o["rouge_l"]),
            }
        )
    return out


def mean_pm_stdev(values: Sequence[float]) -> str:
    if len(values) == 0:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    m = statistics.mean(values)
    s = statistics.stdev(values)
    return f"{m:.4f} ± {s:.4f}"


def format_latex_pm(values: Sequence[float]) -> str:
    if len(values) == 0:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    m = statistics.mean(values)
    s = statistics.stdev(values)
    return f"${m:.4f} \\pm {s:.4f}$"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--logs",
        nargs="*",
        default=[],
        help="train.log 路径或 glob（取各文件内最后一次 FINAL RESULTS）",
    )
    ap.add_argument("--jsonl", type=str, default=None, help="eval 注册表 JSONL（如 meta/eval_registry.jsonl）路径")
    ap.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="仅使用 JSONL 末尾 n 行（常用于同一任务最近 n 次 eval）",
    )
    ap.add_argument(
        "--save-file-contains",
        type=str,
        default=None,
        help="JSONL 模式下过滤 save_file 子串（如某次 step5 run 路径片段）",
    )
    ap.add_argument(
        "--latex",
        action="store_true",
        help="输出 LaTeX 形式的 $\\pm$",
    )
    args = ap.parse_args()

    series: List[Dict[str, float]] = []
    sources: List[str] = []

    for pattern in args.logs:
        for p in sorted(glob(pattern)) if any(c in pattern for c in "*?[]") else [pattern]:
            if not p:
                continue
            if not Path(p).is_file():
                print(f"跳过（非文件）: {p}", file=sys.stderr)
                continue
            try:
                series.append(parse_train_log(p))
                sources.append(p)
            except ValueError as e:
                print(f"{p}: {e}", file=sys.stderr)

    if args.jsonl:
        try:
            jm = load_jsonl_metrics(
                args.jsonl,
                save_file_contains=args.save_file_contains,
                last_n=args.last_n,
            )
            for i, row in enumerate(jm):
                series.append(row)
                sources.append(f"{args.jsonl}#[{i}]")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"JSONL 读取失败: {e}", file=sys.stderr)

    if not series:
        print("错误: 未得到任何有效指标行。", file=sys.stderr)
        sys.exit(1)

    n = len(series)
    fmt = format_latex_pm if args.latex else mean_pm_stdev

    print(f"n = {n} runs")
    if n < 5:
        print(
            "提示: 样本标准差在 n<2 时无法计算；审稿人常要求注明 independent runs 次数。",
            file=sys.stderr,
        )
    print()
    for k in METRIC_KEYS:
        vals = [r[k] for r in series]
        print(f"{k:8s}  {fmt(vals)}")
        print(f"         raw: {vals}")
    print()
    print("sources:")
    for s in sources:
        print(f"  {s}")


if __name__ == "__main__":
    main()
