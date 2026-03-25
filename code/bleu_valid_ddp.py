# -*- coding: utf-8 -*-
"""验证集 explanation BLEU-4：DDP 下全量分片推理 + rank0 聚合算分。"""
from __future__ import annotations

from typing import Any, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from base_utils import compute_bleu1234_only, get_underlying_model
from config import get_dataloader_num_workers, get_dataloader_prefetch_factor


def bleu4_explanation_full_valid_ddp(
    model: torch.nn.Module,
    valid_dataset,
    *,
    tokenizer: Any,
    device: int,
    rank: int,
    world_size: int,
    batch_size: int = 32,
) -> float:
    """
    在完整 valid 上计算 explanation BLEU-4（词级，与 compute_bleu1234_only 一致）。
    各 rank 处理连续分片，all_gather_object 后在 rank0 拼接并按全局顺序算分；
    最后 broadcast 标量，所有 rank 得到相同 float。
    """
    _m = get_underlying_model(model)
    n = len(valid_dataset)
    if n <= 0:
        return 0.0

    if world_size <= 1:
        return _bleu4_subset_local(_m, valid_dataset, list(range(n)), device, tokenizer, batch_size)

    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(n, start + chunk)
    indices: List[int] = list(range(start, end)) if start < n else []

    pred_texts, ref_texts = _bleu4_collect_shard(_m, valid_dataset, indices, device, tokenizer, batch_size)
    payload = {"preds": pred_texts, "refs": ref_texts}
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, payload)

    score = 0.0
    if rank == 0:
        all_pred: List[str] = []
        all_ref: List[str] = []
        for r in range(world_size):
            all_pred.extend(gathered[r]["preds"])
            all_ref.extend(gathered[r]["refs"])
        if len(all_pred) != n:
            raise RuntimeError(
                f"BLEU DDP gather 样本数不一致: 期望 {n}, 实际 {len(all_pred)} (world_size={world_size})"
            )
        score = float(compute_bleu1234_only(all_pred, all_ref).get("4", 0.0))

    t = torch.zeros(1, dtype=torch.float32, device=device)
    if rank == 0:
        t[0] = score
    dist.broadcast(t, src=0)
    return float(t[0].item())


def _bleu4_subset_local(_m, valid_dataset, indices: List[int], device: int, tokenizer, batch_size: int) -> float:
    if not indices:
        return 0.0
    subset = Subset(valid_dataset, indices)
    n = len(subset)
    bs = max(1, min(int(batch_size), n))
    _vn = min(2, get_dataloader_num_workers("valid"))
    _pf = get_dataloader_prefetch_factor(_vn)
    dl = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=_vn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=_vn > 0,
        prefetch_factor=_pf,
    )
    pred_texts: List[str] = []
    ref_texts: List[str] = []
    _m.eval()
    with torch.inference_mode():
        for batch in dl:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _m.gather(batch, device)
            gen_ids, *_ = _m.generate(user_idx, item_idx, domain_idx)
            pred_texts.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            ref_texts.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
    return float(compute_bleu1234_only(pred_texts, ref_texts).get("4", 0.0))


def _bleu4_collect_shard(_m, valid_dataset, indices: List[int], device: int, tokenizer, batch_size: int):
    if not indices:
        return [], []
    subset = Subset(valid_dataset, indices)
    n = len(subset)
    bs = max(1, min(int(batch_size), n))
    _vn = min(2, get_dataloader_num_workers("valid"))
    _pf = get_dataloader_prefetch_factor(_vn)
    dl = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=_vn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=_vn > 0,
        prefetch_factor=_pf,
    )
    pred_texts: List[str] = []
    ref_texts: List[str] = []
    _m.eval()
    with torch.inference_mode():
        for batch in dl:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _m.gather(batch, device)
            gen_ids, *_ = _m.generate(user_idx, item_idx, domain_idx)
            pred_texts.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            ref_texts.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
    return pred_texts, ref_texts
