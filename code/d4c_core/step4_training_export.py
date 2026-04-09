"""
Step4 最终训练表：三类来源（aux_gold / target_gold / aux_cf）+ 清洗与质量字段 + manifest。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from d4c_core.text_cleaning import (
    build_sample_quality_flags,
    build_template_stats,
    clean_explanation_text,
    merge_flags_into_row,
)

DEFAULT_ORIGIN_WEIGHTS: Dict[str, float] = {
    "target_gold": 1.0,
    "aux_gold": 0.9,
    "aux_cf": 0.5,
}


def assemble_step4_training_table(
    train_df: pd.DataFrame,
    filtered_cf_df: pd.DataFrame,
    *,
    origin_weights: Mapping[str, float] | None = None,
    template_min_count: int = 3,
    template_hard_drop_min_count: int = 0,
) -> pd.DataFrame:
    """
    train_df: aug_train 中有 explanation 的子集（与 step4_engine 一致）。
    filtered_cf_df: 熵过滤后的反事实行（domain 已为 auxiliary，explanation 为生成）。
    """
    ow = dict(origin_weights or DEFAULT_ORIGIN_WEIGHTS)

    aux_gold = train_df[train_df["domain"] == "auxiliary"].copy()
    target_gold = train_df[train_df["domain"] == "target"].copy()
    aux_cf = filtered_cf_df.copy()

    aux_gold["_export_role"] = "aux_gold"
    target_gold["_export_role"] = "target_gold"
    aux_cf["_export_role"] = "aux_cf"

    combined = pd.concat([target_gold, aux_gold, aux_cf], ignore_index=True)

    # 第一轮：清洗，准备模板统计
    cleans: List[str] = []
    clean_results = []
    raws: List[str] = []
    for _, row in combined.iterrows():
        raw = row.get("explanation", "")
        raws.append(str(raw) if raw is not None else "")
        cr = clean_explanation_text(raw)
        clean_results.append(cr)
        cleans.append(cr.clean_text)

    tmpl_stats = build_template_stats(cleans)

    rows_out: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(combined.iterrows()):
        d = row.to_dict()
        d["template_hard_drop_min_count"] = int(template_hard_drop_min_count)
        role = str(d.pop("_export_role", ""))
        is_cf = 1 if role == "aux_cf" else 0
        cr = clean_results[i]
        flags = build_sample_quality_flags(
            raw_explanation=raws[i],
            clean_result=cr,
            template_stats=tmpl_stats,
            template_min_count=template_min_count,
        )
        merge_flags_into_row(d, flags, sample_origin=role, is_cf=is_cf, origin_weights=ow)
        rows_out.append(d)

    out_df = pd.DataFrame(rows_out)
    out_df["sample_id"] = np.arange(len(out_df), dtype=np.int64)
    return out_df


def build_step4_train_manifest(
    df: pd.DataFrame,
    *,
    n_cf_entropy_input: int,
    n_cf_entropy_kept: int,
    origin_weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """rank0 写入 JSON；供 runs/ 下审计与外部脚本读取。"""
    ow = dict(origin_weights or DEFAULT_ORIGIN_WEIGHTS)
    manifest: Dict[str, Any] = {
        "schema_version": "d4c_step4_train_table/1.0",
        "origin_weights": ow,
        "entropy_filter": {
            "n_target_rows_for_cf": int(n_cf_entropy_input),
            "n_cf_kept": int(n_cf_entropy_kept),
            "n_cf_dropped": int(max(0, n_cf_entropy_input - n_cf_entropy_kept)),
        },
        "row_counts": {
            "total_rows": int(len(df)),
            "by_sample_origin": {},
        },
        "cleaning": {
            "n_clean_changed": int(df["clean_changed"].sum()) if "clean_changed" in df.columns else 0,
            "n_nonempty_clean_text": int((df["clean_text"].fillna("").astype(str).str.strip() != "").sum()),
        },
        "quality_flag_counts": {},
        "template_hit_audit": {
            "template_hit_total": 0,
            "template_hit_kept": 0,
            "template_hit_downweighted": 0,
            "template_hit_dropped": 0,
        },
        "train_keep": {
            "n_keep_1": int((df["train_keep"] == 1).sum()) if "train_keep" in df.columns else 0,
            "n_keep_0": int((df["train_keep"] == 0).sum()) if "train_keep" in df.columns else 0,
        },
        "drop_reason_counts": {},
    }

    if "sample_origin" in df.columns:
        vc = df["sample_origin"].value_counts()
        manifest["row_counts"]["by_sample_origin"] = {str(k): int(v) for k, v in vc.items()}

    for col in (
        "html_entity_hit",
        "bad_tail_hit",
        "template_hit",
        "short_fragment_hit",
        "repeat_tail_hit",
        "clean_changed",
    ):
        if col in df.columns:
            manifest["quality_flag_counts"][col] = int(df[col].sum())

    if "train_drop_reason" in df.columns:
        dr = df[df["train_keep"] == 0]["train_drop_reason"].fillna("").astype(str)
        manifest["drop_reason_counts"] = dict(Counter([x for x in dr if x]))
    if "template_hit" in df.columns:
        tmask = df["template_hit"] == 1
        manifest["template_hit_audit"]["template_hit_total"] = int(tmask.sum())
        if "train_keep" in df.columns:
            manifest["template_hit_audit"]["template_hit_kept"] = int(
                ((df["template_hit"] == 1) & (df["train_keep"] == 1)).sum()
            )
            manifest["template_hit_audit"]["template_hit_dropped"] = int(
                ((df["template_hit"] == 1) & (df["train_keep"] == 0)).sum()
            )
        if "template_downweighted" in df.columns:
            manifest["template_hit_audit"]["template_hit_downweighted"] = int(
                ((df["template_hit"] == 1) & (df["template_downweighted"] == 1)).sum()
            )

    return manifest


def write_step4_training_artifacts(
    df: pd.DataFrame,
    manifest: Dict[str, Any],
    out_dir: str,
    *,
    csv_name: str = "factuals_counterfactuals.csv",
    manifest_name: str = "step4_train_table_manifest.json",
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)
    man_path = os.path.join(out_dir, manifest_name)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return csv_path, man_path
