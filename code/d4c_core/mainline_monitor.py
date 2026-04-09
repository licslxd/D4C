"""
Step5 主路径监控：full valid 上文本指标 + 评分门控 + 复合选模分。

与训练期 greedy mainline decode 对齐；不参与 nucleus 候选路径。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from base_utils import evaluate_text
from d4c_eval_dirty_text import compute_dirty_text_stats


def ratings_rmse_mae(
    pred: Sequence[float], gt: Sequence[float]
) -> Tuple[float, float]:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.size == 0:
        return 0.0, 0.0
    d = p - g
    rmse = float(math.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    return rmse, mae


def build_mainline_monitor_bundle_from_merged_rows(
    merged_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """merged_rows 须按 sample_id 0..N-1 排序且含 pred_text/ref_text；可选 pred_rating/gt_rating。"""
    preds = [str(r.get("pred_text", "") or "") for r in merged_rows]
    refs = [str(r.get("ref_text", "") or "") for r in merged_rows]
    txt = evaluate_text(preds, refs)
    mean_ref = sum(len(x.split()) for x in refs) / max(len(refs), 1)
    dirty = compute_dirty_text_stats(preds, ref_mean_len_words=mean_ref)
    dirty_rate = float(dirty.get("hit_rate", 0.0))

    rmse = 0.0
    mae = 0.0
    if merged_rows and "pred_rating" in merged_rows[0] and "gt_rating" in merged_rows[0]:
        pr = [float(r["pred_rating"]) for r in merged_rows]
        gr = [float(r["gt_rating"]) for r in merged_rows]
        rmse, mae = ratings_rmse_mae(pr, gr)

    bleu = txt.get("bleu") or {}
    rouge = txt.get("rouge") or {}
    meteor = float(txt.get("meteor") or 0.0)
    b4 = float(bleu.get("4", 0.0))
    rl = float(rouge.get("l", 0.0))
    composite = 0.40 * (b4 / 100.0) + 0.35 * (rl / 100.0) + 0.25 * (meteor / 100.0)

    return {
        "bleu": bleu,
        "rouge": rouge,
        "meteor": meteor,
        "dist": txt.get("dist") or {},
        "dirty_hit_rate": dirty_rate,
        "dirty_stats": dirty,
        "rmse_rating": rmse,
        "mae_rating": mae,
        "mainline_composite_score": float(composite),
    }


def mainline_selection_gate(
    current: Mapping[str, Any],
    best_prev: Optional[Mapping[str, Any]],
    *,
    dirty_relax: float = 0.04,
    rating_relax_ratio: float = 1.10,
) -> Tuple[bool, Dict[str, Any]]:
    """
    门控：相对上一轮 best，dirty 命中率与 RMSE/MAE 不得明显变差。
    无 best 时直接通过。
    """
    if best_prev is None:
        return True, {"reason": "no_previous_best", "passed": True}

    cur_d = float(current.get("dirty_hit_rate", 0.0))
    prev_d = float(best_prev.get("dirty_hit_rate", 0.0))
    ok_dirty = cur_d <= prev_d + dirty_relax

    cur_rmse = float(current.get("rmse_rating", 0.0))
    prev_rmse = float(best_prev.get("rmse_rating", 0.0))
    ok_rmse = prev_rmse <= 1e-8 or cur_rmse <= prev_rmse * rating_relax_ratio + 1e-8

    cur_mae = float(current.get("mae_rating", 0.0))
    prev_mae = float(best_prev.get("mae_rating", 0.0))
    ok_mae = prev_mae <= 1e-8 or cur_mae <= prev_mae * rating_relax_ratio + 1e-8

    passed = bool(ok_dirty and ok_rmse and ok_mae)
    detail = {
        "passed": passed,
        "ok_dirty": ok_dirty,
        "ok_rmse": ok_rmse,
        "ok_mae": ok_mae,
        "cur_dirty_hit_rate": cur_d,
        "prev_dirty_hit_rate": prev_d,
        "cur_rmse": cur_rmse,
        "prev_rmse": prev_rmse,
        "cur_mae": cur_mae,
        "prev_mae": prev_mae,
    }
    return passed, detail


def should_update_best_mainline(
    current: Mapping[str, Any],
    best_prev: Optional[Mapping[str, Any]],
    *,
    best_composite: float,
) -> Tuple[bool, Dict[str, Any]]:
    """复合分严格上升且通过门控时更新 best_mainline。"""
    score = float(current.get("mainline_composite_score", 0.0))
    gate_ok, gate_detail = mainline_selection_gate(current, best_prev)
    improved = score > best_composite + 1e-9
    return bool(improved and gate_ok), {
        "improved": improved,
        "gate_ok": gate_ok,
        "score": score,
        "best_composite_before": best_composite,
        "gate": gate_detail,
    }
