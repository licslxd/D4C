# -*- coding: utf-8 -*-
"""主线 batch gather 命名协议：取代 tuple 位置语义，供训练 / 验证 / BLEU 共用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class GatheredBatch:
    """Model.gather(batch, device) 的统一返回值（主线唯一协议）。"""

    user_idx: torch.Tensor
    item_idx: torch.Tensor
    rating: torch.Tensor
    tgt_input: torch.Tensor
    tgt_output: torch.Tensor
    domain_idx: torch.Tensor
    sample_id: torch.Tensor
    exp_sample_weight: Optional[torch.Tensor] = None

    def assert_uniform_batch_dim(self) -> None:
        """校验各张量首维（batch）一致；失败时指出字段名。"""
        b = int(self.user_idx.shape[0])
        pairs = [
            ("user_idx", self.user_idx),
            ("item_idx", self.item_idx),
            ("rating", self.rating),
            ("tgt_input", self.tgt_input),
            ("tgt_output", self.tgt_output),
            ("domain_idx", self.domain_idx),
            ("sample_id", self.sample_id),
        ]
        for name, t in pairs:
            if int(t.shape[0]) != b:
                raise ValueError(
                    f"GatheredBatch 批量维不一致：基准 batch_size={b}（来自 user_idx），"
                    f"字段 {name!r} 的首维为 {int(t.shape[0])}。"
                )
        w = self.exp_sample_weight
        if w is not None and int(w.shape[0]) != b:
            raise ValueError(
                f"GatheredBatch 批量维不一致：基准 batch_size={b}（来自 user_idx），"
                f"字段 'exp_sample_weight' 的首维为 {int(w.shape[0])}。"
            )


def require_gathered_batch(obj: Any) -> GatheredBatch:
    """将 gather 返回值约束为 GatheredBatch；否则抛出清晰 TypeError。"""
    if not isinstance(obj, GatheredBatch):
        raise TypeError(
            "model.gather(batch, device) 必须返回 d4c_core.gather_schema.GatheredBatch；"
            f"实际为 {type(obj).__name__!r}。主线已移除 tuple 位置协议，请统一改为 GatheredBatch。"
        )
    obj.assert_uniform_batch_dim()
    return obj
