# -*- coding: utf-8 -*-
"""验证集 explanation BLEU-4：DDP 下全量分片推理 + rank0 按 sample_id 聚合后算分。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from base_utils import compute_bleu1234_only, get_underlying_model
from d4c_eval_metrics import merge_eval_rows_by_sample_id


def bleu4_explanation_full_valid_ddp(
    model: torch.nn.Module,
    valid_dataset,
    *,
    tokenizer: Any,
    device: int,
    rank: int,
    world_size: int,
    batch_size: int = 32,
    dataloader_num_workers: int = 2,
    dataloader_prefetch_factor: Optional[int] = 4,
) -> float:
    """
    在完整 valid 上计算 explanation BLEU-4（词级，与 compute_bleu1234_only 一致）。
    各 rank 处理连续分片，all_gather_object 后 rank0 按 sample_id 排序再算分。
    """
    _m = get_underlying_model(model)
    n = len(valid_dataset)
    if n <= 0:
        return 0.0

    if world_size <= 1:
        rows = _bleu_rows_subset(
            _m,
            valid_dataset,
            list(range(n)),
            device,
            tokenizer,
            batch_size,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
        )
        merged = merge_eval_rows_by_sample_id([rows], n)
        preds = [r["pred_text"] for r in merged]
        refs = [r["ref_text"] for r in merged]
        return float(compute_bleu1234_only(preds, refs).get("4", 0.0))

    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(n, start + chunk)
    indices: List[int] = list(range(start, end)) if start < n else []

    rows = _bleu_rows_subset(
        _m,
        valid_dataset,
        indices,
        device,
        tokenizer,
        batch_size,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
    )
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, rows)

    score = 0.0
    if rank == 0:
        merged = merge_eval_rows_by_sample_id(gathered, n)
        preds = [r["pred_text"] for r in merged]
        refs = [r["ref_text"] for r in merged]
        score = float(compute_bleu1234_only(preds, refs).get("4", 0.0))

    t = torch.zeros(1, dtype=torch.float32, device=device)
    if rank == 0:
        t[0] = score
    dist.broadcast(t, src=0)
    return float(t[0].item())


def _bleu_rows_subset(
    _m,
    valid_dataset,
    indices: List[int],
    device: int,
    tokenizer,
    batch_size: int,
    *,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: Optional[int],
) -> List[Dict[str, Any]]:
    if not indices:
        return []
    subset = Subset(valid_dataset, indices)
    n = len(subset)
    bs = max(1, min(int(batch_size), n))
    _vn = max(0, int(dataloader_num_workers))
    _pf = dataloader_prefetch_factor if _vn > 0 else None
    dl = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=_vn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=_vn > 0,
        prefetch_factor=_pf,
    )
    out: List[Dict[str, Any]] = []
    _m.eval()
    with torch.inference_mode():
        for batch in dl:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx, sample_id = _m.gather(
                batch, device
            )
            gen_ids, *_ = _m.generate(user_idx, item_idx, domain_idx)
            pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(tgt_output, skip_special_tokens=True)
            for i in range(user_idx.size(0)):
                out.append(
                    {
                        "sample_id": int(sample_id[i].item()),
                        "pred_text": pred_texts[i],
                        "ref_text": ref_texts[i],
                    }
                )
    return out
