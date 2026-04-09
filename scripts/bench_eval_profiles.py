#!/usr/bin/env python3
"""
顺序运行官方 eval profile，各读 metrics.json 的 eval_performance，汇总为 CSV。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OFFICIAL_EVAL_PROFILES: Tuple[Tuple[str, str], ...] = (
    ("eval_fast_single_gpu", "eval"),
    ("eval_balanced_2gpu", "eval"),
    ("eval_rerank_quality", "eval-rerank"),
)
_SUMMARY_KEYS = (
    "total_eval_time",
    "tokenize_cache_time",
    "dataloader_build_time",
    "decode_time",
    "gather_time",
    "metrics_time",
    "predictions_write_time",
    "rerank_scoring_time",
    "rerank_artifacts_write_time",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_eval_run_dir(combined_output: str) -> Optional[str]:
    m = re.search(r"eval_run_dir=(\S+)", combined_output)
    return m.group(1).rstrip() if m else None


def _load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_from_metrics(profile: str, subcommand: str, metrics_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    ep = data.get("eval_performance") or {}
    summ = ep.get("summary") or {}
    n_samples = int((data.get("collapse_stats") or {}).get("n_samples", 0) or 0)
    tfloat = float(summ.get("total_eval_time") or 0.0)
    sps = (float(n_samples) / tfloat) if n_samples > 0 and tfloat > 0 else None
    row: Dict[str, Any] = {
        "eval_profile": profile,
        "d4c_subcommand": subcommand,
        "metrics_json": str(metrics_path.resolve()),
        "eval_profile_name_in_metrics": data.get("eval_profile_name"),
        "training_semantic_fingerprint": data.get("training_semantic_fingerprint"),
        "generation_semantic_fingerprint": data.get("generation_semantic_fingerprint"),
        "n_samples": n_samples,
        "samples_per_sec": sps,
        "global_eval_batch_size": ep.get("global_eval_batch_size"),
        "per_gpu_eval_batch_size": ep.get("per_gpu_eval_batch_size"),
    }
    for k in _SUMMARY_KEYS:
        row[k] = summ.get(k, "")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="批跑官方 eval profile 并汇总 eval_performance → CSV")
    ap.add_argument("--task", type=int, required=True, choices=range(1, 9))
    ap.add_argument("--preset", type=str, required=True)
    ap.add_argument("--iter", type=str, default="v1", dest="iteration_id")
    ap.add_argument("--from-run", type=str, required=True, dest="from_run")
    ap.add_argument("--step5-run", type=str, required=True, dest="step5_run")
    ap.add_argument("--model-path", type=str, default=None, dest="model_path")
    ap.add_argument("--output", type=str, default="eval_profile_benchmark_summary.csv")
    args = ap.parse_args()

    root = _repo_root()
    d4c = root / "code" / "d4c.py"
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path

    rows: List[Dict[str, Any]] = []
    for profile, sub in OFFICIAL_EVAL_PROFILES:
        cmd: List[str] = [
            sys.executable, str(d4c), sub, "--task", str(args.task), "--preset", args.preset,
            "--iter", args.iteration_id, "--eval-profile", profile, "--run-id", "auto",
        ]
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        else:
            cmd.extend(["--from-run", args.from_run, "--step5-run", args.step5_run])
        r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if r.returncode != 0:
            print((r.stdout or "") + "\n" + (r.stderr or ""), file=sys.stderr)
            return r.returncode
        ev_dir = _parse_eval_run_dir((r.stdout or "") + "\n" + (r.stderr or ""))
        if not ev_dir:
            return 3
        mj = Path(ev_dir) / "metrics.json"
        rows.append(_row_from_metrics(profile, sub, mj, _load_metrics(mj)))

    rows.sort(key=lambda row: float(row.get("total_eval_time") or math.inf))
    fieldnames = [
        "eval_profile", "d4c_subcommand", "metrics_json", "eval_profile_name_in_metrics",
        "training_semantic_fingerprint", "generation_semantic_fingerprint", "n_samples", "samples_per_sec",
        "global_eval_batch_size", "per_gpu_eval_batch_size", *_SUMMARY_KEYS,
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
