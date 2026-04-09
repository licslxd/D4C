"""
Step5 执行体核心（ENGINE）：主模型 train/eval/test。

由 ``executors.step5_entry`` 在 torchrun 下调用（code/ 下历史薄壳名保持不变）。
用户入口请使用 ``python code/d4c.py step5|eval|pipeline …``。
"""
import os
import sys
import time
import hashlib
import logging
# 离线模式：禁止从网络加载
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
from base_utils import *
from paths_config import get_data_dir, get_hf_cache_root, get_stage_run_dir, get_t5_small_dir
from config import (
    FinalTrainingConfig,
    apply_ddp_fast_torch_backends,
    build_full_bleu_monitor_cfg_override,
    build_resolved_training_config,
    format_full_bleu_eval_epoch_decision_log_line,
    format_full_bleu_eval_resolved_log_line,
    format_full_bleu_monitor_log_line,
    hf_datasets_progress_bar,
    resolve_task_idx_from_aux_target,
    should_run_full_bleu_eval_epoch,
)
from training_hardware_inputs import collect_training_hardware_overrides_from_args
from d4c_core.runtime_env_pack import runtime_env_dict_for_config_resolved
from d4c_core.training_diagnostics import training_diagnostics_snapshot
from lr_schedule_utils import resolve_warmup_steps, warmup_cosine_multiplier_lambda
from bleu_valid_ddp import bleu4_explanation_full_valid_ddp
from d4c_core.bleu_runtime import explanation_bleu4_quick_score, mainline_monitor_full_valid_ddp
from d4c_core.mainline_monitor import should_update_best_mainline
from d4c_core.gather_schema import GatheredBatch, require_gathered_batch
import torch

# transformers 在 modeling_utils.load_state_dict 里用 torch.load 未传 weights_only，
# PyTorch 2.4+ 会 FutureWarning；在 from_pretrained 前默认 weights_only=True（与 AdvTrain 一致）
def _patch_torch_load_default_weights_only() -> None:
    _orig = torch.load

    def _wrapped(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = True
        try:
            return _orig(*args, **kwargs)
        except TypeError as e:
            if "weights_only" not in str(e).lower():
                raise
            kwargs.pop("weights_only", None)
            return _orig(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[assignment]


_patch_torch_load_default_weights_only()

import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from transformers import T5Tokenizer
from torch import nn, optim
from torch.nn.modules.transformer import _get_activation_fn
import argparse
import gzip
import json
import contextlib
import math
from functools import partial
from dataclasses import replace
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from torch.optim import lr_scheduler as lr_sched
from datasets import Dataset, DatasetDict, load_from_disk
from perf_monitor import PerfMonitor, gather_ddp_gpu_stats_for_epoch_log
import numpy as np
import copy
import torch.nn.functional as F
from train_logging import (
    append_train_epoch_metrics_jsonl,
    create_run_paths,
    setup_train_logging,
    log_run_snapshot,
    flush_preset_load_events,
    format_epoch_summary_lines,
    format_epoch_training_block,
    log_epoch_training_block,
    log_epoch_summary_compact,
    broadcast_run_paths_ddp,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    flush_d4c_file_handlers,
    LOGGER_NAME,
    logger_has_file_handler,
    log_route_extra,
    ROUTE_DETAIL,
    ROUTE_SUMMARY,
)
from d4c_eval_dirty_text import compute_dirty_text_stats
from d4c_core.generation_semantics import build_generation_semantic_resolved_and_fingerprint
from d4c_core.rerank import (
    build_rerank_weights_dict,
    compute_lp_norm,
    extract_rerank_features,
    extract_rerank_features_for_v3,
    keywords_from_source_text,
    merge_rerank_v3_profile,
    rouge_l_proxy,
    score_candidates_rule_v1,
    score_candidates_rule_v2,
    score_candidates_rule_v3,
)
from d4c_eval_metrics import (
    compute_collapse_stats,
    eval_decode_tag,
    extended_text_metrics_bundle,
    log_sample_id_alignment_snippet,
    merge_eval_rows_by_sample_id,
    write_predictions_csv,
    write_predictions_jsonl,
    write_eval_digest_log,
)
from d4c_core.step5_word_losses import (
    d4c_anti_repeat_unlikelihood_loss_from_logp,
    per_sample_mean_ce_from_logp,
)
from executors.decode_controller import (
    GenerateConfig,
    coerce_generate_cfg_override,
    build_candidate_generation_specs,
    apply_eos_boost,
    apply_min_len_eos_mask,
    apply_no_repeat_ngram_logits,
    apply_repetition_penalty_logits,
    apply_sampling_schedule,
    apply_token_repeat_suppression,
    apply_unbalanced_delimiter_eos_mask,
    build_generate_kwargs_effective_v2,
    forbid_eos_if_bad_tail_token,
    prepare_logits,
    sample_next_token,
)
from train_diagnostics import (
    collect_distributed_env_for_meta,
    d4c_cuda_bf16_autocast,
    d4c_cuda_bf16_autocast_enabled,
    d4c_grad_topk,
    d4c_log_grad_interval,
    d4c_log_step_interval,
    d4c_log_step_loss_parts,
    d4c_save_checkpoint,
    d4c_timing_phase,
    ddp_heartbeat,
    grad_norm_total,
    grad_topk_param_norms,
    log_bf16_amp_note,
    log_step_sample,
    log_training_crash,
    maybe_log_grad_norm_diff_ddp,
    parse_d4c_finite_check_mode,
    run_training_finite_checks,
    warn_empty_batch,
)

# 离线加载：优先使用本地 T5 目录
_t5_base = get_t5_small_dir()
_t5_path = _t5_base if os.path.exists(_t5_base) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)

# HuggingFace tokenize 磁盘缓存：与 AdvTrain（step3）在「共享 tokenize 语义」变更时同步递增
D4C_TOKENIZE_CACHE_VERSION = "v3"

tasks = [
    ("AM_Electronics", "AM_CDs"),
    ("AM_Movies", "AM_CDs"),
    ("AM_CDs", "AM_Electronics"),
    ("AM_Movies", "AM_Electronics"),
    ("AM_CDs", "AM_Movies"),
    ("AM_Electronics", "AM_Movies"),
    ("Yelp", "TripAdvisor"),
    ("TripAdvisor", "Yelp")
]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask, src_key_padding_mask):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attns = []
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)  
        if self.norm is not None:
            output = self.norm(output)
        return output, attns

class Processor():
    def __init__(self, auxiliary, target, max_length: int = 25):
        self.max_length = int(max_length)
        self.auxiliary = auxiliary
        self.target = target

    def __call__(self, sample):
        user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
        item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
        raitng = torch.tensor(sample["rating"], dtype=torch.float)
        if "clean_text" not in sample:
            raise KeyError(
                "训练样本缺少 clean_text（须使用新版 Step4 CSV）；禁止静默回退到 explanation 原文。"
            )
        explanation = sample["clean_text"]
        explanation_idx = tokenizer(
            explanation, padding=False, max_length=self.max_length, truncation=True
        )["input_ids"]
        explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)

        if sample["domain"] == "auxiliary":
            domain_val = 0  # Auxiliary domain
        elif sample["domain"] == "target":
            domain_val = 1  # Target domain
        else:
            raise ValueError("Unknown domain!")

        domain_idx = torch.tensor(domain_val, dtype=torch.long)
        sample_id = torch.tensor(int(sample["sample_id"]), dtype=torch.long)
        sw = sample.get("sample_weight_hint", 1.0)
        exp_sample_weight = torch.tensor(float(sw), dtype=torch.float32)
        return {
            "user_idx": user_idx,
            "item_idx": item_idx,
            "rating": raitng,
            "explanation_idx": explanation_idx,
            "domain_idx": domain_idx,
            "sample_id": sample_id,
            "exp_sample_weight": exp_sample_weight,
        }


def _step5_collate_dynamic(
    batch: List[Dict[str, torch.Tensor]],
    *,
    dynamic_padding: bool,
    fixed_max_length: int,
):
    if not batch:
        raise ValueError("step5 collate 收到空 batch。")
    user_idx = torch.stack([torch.as_tensor(x["user_idx"], dtype=torch.long) for x in batch], dim=0)
    item_idx = torch.stack([torch.as_tensor(x["item_idx"], dtype=torch.long) for x in batch], dim=0)
    rating = torch.stack([torch.as_tensor(x["rating"], dtype=torch.float32) for x in batch], dim=0)
    domain_idx = torch.stack([torch.as_tensor(x["domain_idx"], dtype=torch.long) for x in batch], dim=0)
    sample_id = torch.stack([torch.as_tensor(x["sample_id"], dtype=torch.long) for x in batch], dim=0)
    exp_sample_weight = torch.stack(
        [torch.as_tensor(x["exp_sample_weight"], dtype=torch.float32) for x in batch], dim=0
    )
    seqs = [torch.as_tensor(x["explanation_idx"], dtype=torch.long).view(-1) for x in batch]
    if dynamic_padding:
        max_len = max(int(s.numel()) for s in seqs)
    else:
        max_len = max(1, int(fixed_max_length))
    padded = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = min(max_len, int(s.numel()))
        padded[i, :L] = s[:L]
    return (
        user_idx,
        item_idx,
        rating,
        padded,
        domain_idx,
        sample_id,
        exp_sample_weight,
    )

class PETER_MLP(nn.Module):
    def __init__(self, emsize=512):
        super().__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear1(hidden))  # (batch_size, emsize)
        rating = self.linear2(mlp_vector).view(-1)  # (batch_size,)
        return rating


def _domain_fusion_causal_mask(tgt_len: int, device: torch.device, prefix_len: int = 2) -> torch.Tensor:
    """前缀 prefix_len 个 token 全互见，其后为因果掩码（语义对齐旧 generate_domain_mask，前缀长度改为 2）。"""
    total_len = prefix_len + tgt_len
    mask = torch.triu(torch.ones((total_len, total_len), device=device, dtype=torch.bool), diagonal=1)
    mask[:prefix_len, :prefix_len] = False
    return mask


class Model(nn.Module):
    """域提示交叉注意力融合 user/item 上下文 + 主 Transformer；解码为 logit 重复惩罚 + Top-p 采样。"""

    PREFIX_LEN = 2  # [domain_cross_out, domain_raw]

    def __init__(
        self,
        nuser,
        nitem,
        ntoken,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout,
        user_profiles,
        item_profiles,
        domain_profiles,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.domain_profiles = nn.Parameter(domain_profiles)
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.user_profiles = nn.Parameter(user_profiles)  # user_profiles
        self.item_profiles = nn.Parameter(item_profiles)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.recommender = PETER_MLP(emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.ntoken = int(ntoken)
        self.domain_cross_attn = nn.MultiheadAttention(
            embed_dim=emsize, num_heads=nhead, dropout=dropout, batch_first=True
        )
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layers, nlayers)
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.emsize = emsize
        self.repetition_penalty = 1.15
        self.generate_temperature = 0.8
        self.generate_top_p = 0.9
        self.max_explanation_length = 25
        self.decode_strategy = "greedy"
        self.decode_seed = None  # type: Optional[int]
        self.no_repeat_ngram_size = None  # type: Optional[int]
        self.min_len = None  # type: Optional[int]
        self.soft_max_len = None  # type: Optional[int]
        self.hard_max_len = None  # type: Optional[int]
        self.eos_boost_start = 9999
        self.eos_boost_value = 0.0
        self.tail_temperature = -1.0
        self.tail_top_p = -1.0
        self.forbid_eos_after_open_quote = False
        self.forbid_eos_after_open_bracket = False
        self.forbid_bad_terminal_tokens = True
        self.bad_terminal_token_ids_resolved: Tuple[int, ...] = ()
        self.decode_token_repeat_window = 4
        self.decode_token_repeat_max = 2
        self.candidate_family = "balanced"
        self.candidate_mixed_include_diverse = True
        self.loss_weight_repeat_ul = 0.0
        self.loss_weight_terminal_clean = 0.0
        self.terminal_clean_span = 3
        self.decoder_eos_id = -1
        self.rating_loss_fn = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=float(label_smoothing))
        self.init_weights()

    def init_weights(self):
        # 仅初始化自有 Linear / Embedding / Parameter；跳过 TransformerEncoder 内 self_attn 子树，
        # 避免 uniform_ 覆盖 nn.MultiheadAttention 的 in_proj（PyTorch 默认 xavier 更合理）。
        initrange = 0.1

        def _init_linear(m: nn.Linear) -> None:
            nn.init.uniform_(m.weight.data, -initrange, initrange)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        _init_linear(self.hidden2token)
        nn.init.uniform_(self.user_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.item_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.word_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.domain_profiles.data, -initrange, initrange)
        nn.init.uniform_(self.user_profiles.data, -initrange, initrange)
        nn.init.uniform_(self.item_profiles.data, -initrange, initrange)
        self.recommender.init_weights()
        for enc_layer in self.transformer_encoder.layers:
            for name, mod in enc_layer.named_modules():
                if ".self_attn" in name or name.endswith("self_attn"):
                    continue
                if isinstance(mod, nn.Linear):
                    _init_linear(mod)

    def apply_runtime_config(self, cfg: FinalTrainingConfig, tok) -> None:
        """从 FinalTrainingConfig + tokenizer 同步解码与解释头超参（训练/评测共用）。"""
        self.repetition_penalty = float(cfg.repetition_penalty)
        self.generate_temperature = max(float(cfg.generate_temperature), 1e-8)
        self.generate_top_p = float(cfg.generate_top_p)
        self.max_explanation_length = int(cfg.max_explanation_length)
        self.decode_strategy = str(cfg.decode_strategy).strip().lower()
        self.decode_seed = int(cfg.decode_seed) if cfg.decode_seed is not None else None
        _nr = getattr(cfg, "no_repeat_ngram_size", None)
        self.no_repeat_ngram_size = int(_nr) if _nr is not None and int(_nr) > 0 else None
        _mn = getattr(cfg, "min_len", None)
        self.min_len = int(_mn) if _mn is not None and int(_mn) > 0 else None
        _sm = getattr(cfg, "soft_max_len", None)
        self.soft_max_len = int(_sm) if _sm is not None and int(_sm) > 0 else None
        _hm = getattr(cfg, "hard_max_len", None)
        self.hard_max_len = int(_hm) if _hm is not None and int(_hm) > 0 else None
        self.eos_boost_start = int(getattr(cfg, "eos_boost_start", 9999))
        self.eos_boost_value = float(getattr(cfg, "eos_boost_value", 0.0))
        self.tail_temperature = float(getattr(cfg, "tail_temperature", -1.0))
        self.tail_top_p = float(getattr(cfg, "tail_top_p", -1.0))
        self.forbid_eos_after_open_quote = bool(getattr(cfg, "forbid_eos_after_open_quote", True))
        self.forbid_eos_after_open_bracket = bool(getattr(cfg, "forbid_eos_after_open_bracket", True))
        self.forbid_bad_terminal_tokens = bool(getattr(cfg, "forbid_bad_terminal_tokens", True))
        self.decode_token_repeat_window = int(getattr(cfg, "decode_token_repeat_window", 4))
        self.decode_token_repeat_max = int(getattr(cfg, "decode_token_repeat_max", 2))
        self.candidate_family = str(getattr(cfg, "candidate_family", "balanced")).strip().lower()
        self.candidate_mixed_include_diverse = bool(getattr(cfg, "candidate_mixed_include_diverse", True))
        self.loss_weight_repeat_ul = float(getattr(cfg, "loss_weight_repeat_ul", 0.0))
        self.loss_weight_terminal_clean = float(getattr(cfg, "loss_weight_terminal_clean", 0.0))
        self.terminal_clean_span = int(getattr(cfg, "terminal_clean_span", 3))
        eid = getattr(tok, "eos_token_id", None)
        self.decoder_eos_id = int(eid) if eid is not None else -1
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=float(cfg.label_smoothing))
        _bt = getattr(cfg, "bad_terminal_token_ids", None)
        if _bt is not None and len(_bt) > 0:
            self.bad_terminal_token_ids_resolved = tuple(int(x) for x in _bt)
        else:
            self.bad_terminal_token_ids_resolved = Model._default_bad_terminal_token_ids(tok)

    @staticmethod
    def _default_bad_terminal_token_ids(tok) -> Tuple[int, ...]:
        """未配置时的坏尾 token：常见未闭合起笔符号的首子词 id。"""
        ids: List[int] = []
        for piece in ("(", "[", "{", "<", "``", "''"):
            try:
                enc = tok.encode(piece, add_special_tokens=False)
            except Exception:
                enc = []
            if enc:
                ids.append(int(enc[0]))
        return tuple(sorted(set(ids)))

    def _context_kv(self, user, item):
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        return torch.cat([user_profile, item_profile, user_embeddings, item_embeddings], dim=1)

    def _domain_prompted_prefix(self, domain_idx, user, item):
        domain_embedding = self.domain_profiles[domain_idx].unsqueeze(dim=1)
        context_kv = self._context_kv(user, item)
        domain_enhanced, _ = self.domain_cross_attn(
            domain_embedding, context_kv, context_kv, need_weights=False
        )
        return domain_enhanced, domain_embedding

    def _make_generate_config(self) -> GenerateConfig:
        hard = int(self.hard_max_len) if getattr(self, "hard_max_len", None) else int(self.max_explanation_length)
        soft = int(self.soft_max_len) if getattr(self, "soft_max_len", None) not in (None, 0) else 0
        _nr = self.no_repeat_ngram_size
        nrs = int(_nr) if _nr is not None and int(_nr) > 0 else 0
        _mn = self.min_len
        min_l = int(_mn) if _mn is not None and int(_mn) > 0 else 0
        bad = tuple(getattr(self, "bad_terminal_token_ids_resolved", ()) or ())
        return GenerateConfig(
            strategy=str(self.decode_strategy).lower(),
            temperature=float(self.generate_temperature),
            top_p=float(self.generate_top_p),
            repetition_penalty=float(self.repetition_penalty),
            no_repeat_ngram_size=nrs,
            min_len=min_l,
            soft_max_len=soft,
            hard_max_len=max(1, hard),
            eos_boost_start=int(getattr(self, "eos_boost_start", 9999)),
            eos_boost_value=float(getattr(self, "eos_boost_value", 0.0)),
            tail_temperature=float(getattr(self, "tail_temperature", -1.0)),
            tail_top_p=float(getattr(self, "tail_top_p", -1.0)),
            forbid_eos_after_open_quote=bool(getattr(self, "forbid_eos_after_open_quote", True)),
            forbid_eos_after_open_bracket=bool(getattr(self, "forbid_eos_after_open_bracket", True)),
            forbid_bad_terminal_tokens=bool(getattr(self, "forbid_bad_terminal_tokens", True)),
            bad_terminal_token_ids=bad,
            token_repeat_window=int(getattr(self, "decode_token_repeat_window", 4)),
            token_repeat_max=int(getattr(self, "decode_token_repeat_max", 2)),
            decode_seed=self.decode_seed,
        )

    def get_generate_kwargs_effective(self) -> Dict[str, Any]:
        """实际参与本模型手写 decode 循环的参数字典（供 metrics / 日志核对）。"""
        out: Dict[str, Any] = {
            "decode_strategy": self.decode_strategy,
            "max_explanation_length": self.max_explanation_length,
            "repetition_penalty": self.repetition_penalty,
            "generate_temperature": self.generate_temperature,
            "generate_top_p": self.generate_top_p,
        }
        if self.decode_seed is not None:
            out["decode_seed"] = self.decode_seed
        if self.decoder_eos_id >= 0:
            out["eos_token_id"] = self.decoder_eos_id
        if self.no_repeat_ngram_size is not None:
            out["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if self.min_len is not None:
            out["min_length"] = self.min_len
        return out

    def get_generate_kwargs_effective_v2(self) -> Dict[str, Any]:
        return build_generate_kwargs_effective_v2(
            self._make_generate_config(),
            eos_token_id=int(self.decoder_eos_id),
        )

    def _decode_with_controller(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator],
        *,
        track_logprobs: bool,
        cfg_override: Optional[GenerateConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
        gc = cfg_override if cfg_override is not None else self._make_generate_config()
        bos_idx = 0
        device = user.device
        batch_size = int(user.shape[0])
        domain_enhanced, domain_embedding = self._domain_prompted_prefix(domain, user, item)
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device).fill_(bos_idx)
        eos_id = int(self.decoder_eos_id)
        gen = generator
        if gen is None and self.decode_seed is not None and gc.strategy == "nucleus":
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.decode_seed))
        total_entropies: List[torch.Tensor] = []
        max_steps = int(gc.hard_max_len)
        token_logprob_sum = torch.zeros(batch_size, device=device, dtype=torch.float32)
        token_count = torch.zeros(batch_size, device=device, dtype=torch.float32)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        recent: List[List[int]] = [[] for _ in range(batch_size)]
        attention_scores = None
        for _step in range(max_steps):
            if not bool(active.any()):
                break
            gen_so_far = int(decoder_input_ids.shape[1]) - 1
            word_feature = self.word_embeddings(decoder_input_ids)
            src = torch.cat([domain_enhanced, domain_embedding, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)
            attn_mask = _domain_fusion_causal_mask(decoder_input_ids.shape[1], device, prefix_len=self.PREFIX_LEN)
            hidden, attention_scores = self.transformer_encoder(src=src, mask=attn_mask)
            logits = prepare_logits(hidden[:, -1, :], self.hidden2token)
            logits = apply_repetition_penalty_logits(logits, decoder_input_ids, float(gc.repetition_penalty))
            if gc.no_repeat_ngram_size > 0:
                apply_no_repeat_ngram_logits(logits, decoder_input_ids, int(gc.no_repeat_ngram_size))
            apply_token_repeat_suppression(
                logits,
                recent,
                window=int(gc.token_repeat_window),
                max_same=int(gc.token_repeat_max),
            )
            apply_min_len_eos_mask(logits, eos_id=eos_id, gen_so_far=gen_so_far, min_len=int(gc.min_len))
            if gc.forbid_eos_after_open_quote or gc.forbid_eos_after_open_bracket:
                texts = tokenizer.batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)
                apply_unbalanced_delimiter_eos_mask(logits, eos_id=eos_id, decoded_texts=texts, cfg=gc)
            tail_ids = decoder_input_ids[:, -1]
            if gc.forbid_bad_terminal_tokens:
                forbid_eos_if_bad_tail_token(
                    logits,
                    eos_id=eos_id,
                    tail_token_ids=tail_ids,
                    bad_ids=gc.bad_terminal_token_ids,
                )
            eff_t, eff_p = apply_sampling_schedule(gc, gen_so_far)
            apply_eos_boost(logits, eos_id=eos_id, step=gen_so_far, cfg=gc)
            output_id, ent_step, log_probs = sample_next_token(
                logits,
                strategy=gc.strategy,
                temperature=eff_t,
                top_p=eff_p,
                generator=gen,
            )
            total_entropies.append(ent_step)
            if track_logprobs:
                chosen_lp = log_probs.gather(1, output_id).squeeze(-1)
                token_logprob_sum = token_logprob_sum + chosen_lp * active.to(dtype=torch.float32)
                token_count = token_count + active.to(dtype=torch.float32)
            for b in range(batch_size):
                recent[b].append(int(output_id[b, 0].item()))
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
            if eos_id >= 0:
                active = active & (output_id.squeeze(-1) != eos_id)
                if not bool(active.any()):
                    break
        stacked = (
            torch.stack(total_entropies).mean(dim=0)
            if total_entropies
            else torch.zeros(batch_size, device=device)
        )
        avg_lp: Optional[torch.Tensor]
        if track_logprobs:
            avg_lp = token_logprob_sum / token_count.clamp(min=1.0)
        else:
            avg_lp = None
        return decoder_input_ids[:, 1:], stacked, attention_scores, avg_lp

    def forward(self, user, item, tgt_input, domain_idx):
        device = user.device
        domain_enhanced, domain_embedding = self._domain_prompted_prefix(domain_idx, user, item)
        word_feature = self.word_embeddings(tgt_input)
        src = torch.cat([domain_enhanced, domain_embedding, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = _domain_fusion_causal_mask(tgt_input.shape[1], device, prefix_len=self.PREFIX_LEN)
        hidden, _ = self.transformer_encoder(src=src, mask=attn_mask)
        rating = self.recommender(hidden[:, 0])
        # 仅保留 (B,V) context logits；loss 端使用广播避免物化 (B,T,V) repeat 大张量。
        context_dist = self.hidden2token(hidden[:, 1])
        word_dist = self.hidden2token(hidden[:, 2:])
        return rating, context_dist, word_dist
    
    def gather(self, batch, device):
        if len(batch) != 7:
            raise ValueError(
                f"batch 须含 7 张量（含 exp_sample_weight），当前 {len(batch)}。"
                "请确认 DataLoader 与新版 Processor 一致。"
            )
        user_idx, item_idx, rating, tgt_output, domain_idx, sample_id, exp_sample_weight = batch
        # 配合 DataLoader(pin_memory=True) 使用 non_blocking=True，减少同步拷贝等待
        user_idx = user_idx.to(device, non_blocking=True)
        item_idx = item_idx.to(device, non_blocking=True)
        domain_idx = domain_idx.to(device, non_blocking=True)
        rating = rating.to(device, non_blocking=True).float()
        tgt_output = tgt_output.to(device, non_blocking=True)
        sample_id = sample_id.to(device, non_blocking=True)
        exp_sample_weight = exp_sample_weight.to(device, non_blocking=True).float()
        tgt_input = T5_shift_right(tgt_output)
        return GatheredBatch(
            user_idx=user_idx,
            item_idx=item_idx,
            rating=rating,
            tgt_input=tgt_input,
            tgt_output=tgt_output,
            domain_idx=domain_idx,
            sample_id=sample_id,
            exp_sample_weight=exp_sample_weight,
        )

    def recommend(self, user, item, domain):
        domain_enhanced, domain_embedding = self._domain_prompted_prefix(domain, user, item)
        src = torch.cat([domain_enhanced, domain_embedding], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, _ = self.transformer_encoder(src=src, mask=None)
        rating = self.recommender(hidden[:, 0])
        return rating

    def generate(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator] = None,
        *,
        cfg_override: Optional[Mapping[str, Any]] = None,
    ):
        """单次前向解码；cfg_override 仅影响本调用，不回写模型默认 decode 配置。"""
        gc = coerce_generate_cfg_override(self._make_generate_config(), cfg_override)
        out, ent, attn, _ = self._decode_with_controller(
            user, item, domain, generator, track_logprobs=False, cfg_override=gc
        )
        return out, ent, attn

    def generate_with_token_logprobs(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator] = None,
        *,
        cfg_override: Optional[Union[GenerateConfig, Mapping[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        解码 + 每 token 选中类 logprob；平均得 avg_logprob（长度归一见 rerank v3 lp_norm）。
        cfg_override 可为 GenerateConfig 或 dict，仅本调用有效。
        """
        gc = coerce_generate_cfg_override(self._make_generate_config(), cfg_override)
        out, ent, attn, avg_lp = self._decode_with_controller(
            user, item, domain, generator, track_logprobs=True, cfg_override=gc
        )
        assert avg_lp is not None
        return out, ent, attn, avg_lp


def _per_sample_mean_ce(
    logits_bt: torch.Tensor,
    tgt: torch.Tensor,
    *,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    """(B,T,V) 与 (B,T) → 每样本对非 padding 位置的平均 CE（与全局 CE 的 label_smoothing 语义一致）。"""
    B, T, V = logits_bt.shape
    ce = F.cross_entropy(
        logits_bt.reshape(-1, V),
        tgt.reshape(-1).long(),
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing),
        reduction="none",
    ).view(B, T)
    mask = (tgt != ignore_index).to(dtype=ce.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (ce * mask).sum(dim=1) / denom


def _per_sample_mean_ce_bv_to_bt(
    logits_bv: torch.Tensor,
    tgt: torch.Tensor,
    *,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    """
    context 分支 CE：输入 (B,V) + target (B,T)。
    通过广播 gather 计算每个 token 位点的 CE，并按非 pad 位点求样本均值；
    避免构造 (B,T,V) repeat 张量。
    """
    B, V = logits_bv.shape
    T = int(tgt.shape[1])
    logp = F.log_softmax(logits_bv, dim=-1)
    tg = tgt.long()
    gather_idx = tg.clamp(min=0).unsqueeze(-1)
    nll = -logp.unsqueeze(1).expand(B, T, V).gather(-1, gather_idx).squeeze(-1)
    if float(label_smoothing) > 0.0:
        smooth = -logp.mean(dim=-1, keepdim=True).expand(B, T)
        ce = (1.0 - float(label_smoothing)) * nll + float(label_smoothing) * smooth
    else:
        ce = nll
    ce = torch.where(tg == int(ignore_index), torch.zeros_like(ce), ce)
    mask = (tg != int(ignore_index)).to(dtype=ce.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (ce * mask).sum(dim=1) / denom


@contextlib.contextmanager
def _ddp_no_sync_model(model, world_size: int, sync_gradients: bool):
    """梯度累积非边界微批上使用 DDP no_sync。"""
    if world_size <= 1 or sync_gradients:
        yield
    else:
        with model.no_sync():
            yield


def d4c_profile_step_components_enabled() -> bool:
    """D4C_PROFILE_STEP_COMPONENTS=1 时开启 Step5 微批热点计时（仅用于验证优化，默认关闭）。"""
    v = os.environ.get("D4C_PROFILE_STEP_COMPONENTS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class _StepComponentCudaTimer:
    """rank0 调试：可选 CUDA Event 计时；关闭时方法为空操作，无额外同步。"""

    def __init__(self, active: bool, *, use_cuda_events: bool):
        self.active = active
        self.use_cuda = bool(active and use_cuda_events)
        self._starts: Dict[str, Any] = {}
        self._ends: Dict[str, Any] = {}

    def start(self, name: str) -> None:
        if not self.active:
            return
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._starts[name] = ev
        else:
            self._starts[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not self.active:
            return
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._ends[name] = ev
        else:
            self._ends[name] = time.perf_counter()

    def ms(self, name: str) -> float:
        if not self.active:
            return 0.0
        s, e = self._starts.get(name), self._ends.get(name)
        if s is None or e is None:
            return 0.0
        if self.use_cuda:
            e.synchronize()
            return float(s.elapsed_time(e))
        return float((e - s) * 1000.0)


def trainModel_ddp(
    model,
    train_dataloader,
    valid_dataloader,
    sampler,
    valid_sampler,
    final_cfg: FinalTrainingConfig,
    rank,
    world_size,
    step5_collate_fn=None,
):
    epochs = final_cfg.epochs
    G = int(final_cfg.train_batch_size)
    P = int(final_cfg.per_device_train_batch_size)
    A = max(1, int(final_cfg.gradient_accumulation_steps))
    eff = int(final_cfg.effective_global_batch_size)
    initial_lr = float(final_cfg.scheduler_initial_lr)
    learning_rate = initial_lr
    coef = float(final_cfg.coef)
    eta = float(final_cfg.eta)
    _model = get_underlying_model(model)
    device = final_cfg.device
    use_bf16 = d4c_cuda_bf16_autocast_enabled()
    n_micro = len(train_dataloader)
    n_steps = max(1, n_micro // A)
    train_info = (
        f"[Train] global_batch_size={G} effective_global_batch_size={eff} "
        f"per_device_batch_size={P} gradient_accumulation_steps={A} world_size={world_size} "
        f"micro_batches_per_epoch={n_micro} optimizer_steps_per_epoch={n_steps} epochs={epochs}"
    )
    _lg = final_cfg.logger
    min_epochs = int(final_cfg.min_epochs)
    early_stop_patience = int(final_cfg.early_stop_patience)
    early_stop_patience_full = int(final_cfg.early_stop_patience_full)
    early_stop_patience_loss = int(final_cfg.early_stop_patience_loss)
    checkpoint_metric = str(final_cfg.checkpoint_metric)
    bleu4_max_samples = int(final_cfg.bleu4_max_samples)
    quick_eval_max_samples = int(final_cfg.quick_eval_max_samples)
    valid_dataset_for_bleu = final_cfg.valid_dataset
    lr_scheduler_type = str(final_cfg.lr_scheduler)
    warmup_epochs = float(final_cfg.warmup_epochs)
    fe_sched = final_cfg.full_bleu_eval_resolved
    full_bleu_monitor_cfg_override = build_full_bleu_monitor_cfg_override(final_cfg)
    dual_bleu = bool(final_cfg.dual_bleu_eval)
    min_lr_ratio = float(final_cfg.min_lr_ratio)
    warmup_steps_env = final_cfg.d4c_warmup_steps
    warmup_ratio_env = final_cfg.d4c_warmup_ratio
    total_steps_plan = max(1, int(epochs * n_steps))
    best_full_bleu4 = -1.0
    best_mainline_composite = -1.0
    best_mainline_gate_baseline: Optional[Dict[str, Any]] = None
    enduration = 0
    full_eval_stall = 0
    valid_loss_stall = 0
    prev_valid_loss = float("inf")
    if rank == 0:
        if _lg:
            _lg.info(train_info, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(train_info, flush=True)
        if use_bf16:
            _bf16_msg = "[Train] bf16 autocast: ON (default, CUDA bf16 supported; set D4C_BF16=0 to disable)"
        elif os.environ.get("D4C_BF16", "").strip().lower() in ("0", "false", "no", "off"):
            _bf16_msg = "[Train] bf16 autocast: OFF (D4C_BF16 disables bf16)"
        elif not torch.cuda.is_available():
            _bf16_msg = "[Train] bf16 autocast: OFF (CUDA not available)"
        else:
            _bf16_msg = "[Train] bf16 autocast: OFF (torch.cuda.is_bf16_supported() is False)"
        if _lg:
            _lg.info(_bf16_msg, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_bf16_msg, flush=True)
        log_bf16_amp_note(_lg, use_bf16, has_grad_scaler=False)
        _fbe_line = format_full_bleu_eval_resolved_log_line(fe_sched)
        if _lg:
            _lg.info(_fbe_line, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_fbe_line, flush=True)
        _fb_mon_line = format_full_bleu_monitor_log_line(final_cfg)
        if _lg:
            _lg.info(_fb_mon_line, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_fb_mon_line, flush=True)
        _es = (
            f"Early stop: min_epochs={min_epochs}, patience={early_stop_patience} (非 dual_bleu 时 valid 变差), "
            f"early_stop_patience_full={early_stop_patience_full} (dual_bleu: full BLEU 未刷新 best), "
            f"early_stop_patience_loss={early_stop_patience_loss} (dual_bleu: valid_loss 连续变差), "
            f"checkpoint_metric={checkpoint_metric}, quick_eval_max_samples={quick_eval_max_samples}, "
            f"dual_bleu_eval={dual_bleu}"
        )
        if _lg:
            _lg.info(_es, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_es, flush=True)
        if _lg:
            _lg.info(
                "Train profile: lr_scheduler=%s warmup_epochs=%g %s",
                lr_scheduler_type,
                warmup_epochs,
                _fbe_line,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
            _lg.info(
                "[CheckpointPolicy] checkpoint_metric=%s | quick_bleu=trend_only | "
                "full_bleu_monitor=primary | dual_bleu=%s",
                checkpoint_metric,
                dual_bleu,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
            _lg.info(
                "[ValidWeighting] aligned_with_train_main_loss=True (sample_weight/exp_sample_weight enabled)",
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    sched = None
    ws_resolved = None
    warmup_ratio_logged = 0.0
    min_lr_effective = initial_lr * min_lr_ratio
    if lr_scheduler_type == "warmup_cosine":
        ws_resolved, warmup_ratio_logged = resolve_warmup_steps(
            total_steps_plan,
            n_steps,
            explicit_steps=warmup_steps_env,
            explicit_ratio=warmup_ratio_env,
            warmup_epochs_fallback=warmup_epochs,
        )
        lr_lambda = warmup_cosine_multiplier_lambda(ws_resolved, total_steps_plan, min_lr_ratio)
        sched = lr_sched.LambdaLR(optimizer, lr_lambda)
        if rank == 0 and _lg:
            _lg.info(
                "LR schedule resolved: scheduler_type=warmup_cosine "
                "initial_lr=%s current_lr=%s (equals initial before first step) min_lr=%s min_lr_ratio=%s "
                "warmup_steps=%d total_steps=%d warmup_ratio=%s | "
                "LambdaLR: one scheduler.step() immediately after each optimizer.step() (global_step aligned)",
                initial_lr,
                initial_lr,
                min_lr_effective,
                min_lr_ratio,
                ws_resolved,
                total_steps_plan,
                warmup_ratio_logged,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
    device_ids = list(final_cfg.device_ids) if final_cfg.device_ids else [device]
    train_nw = getattr(train_dataloader, "num_workers", 0)
    valid_nw = getattr(valid_dataloader, "num_workers", 0) if valid_dataloader is not None else None
    perf = None
    if rank == 0:
        perf = PerfMonitor(
            device=final_cfg.device,
            log_file=final_cfg.log_file,
            num_proc=final_cfg.num_proc,
            device_ids=device_ids,
            train_num_workers=train_nw,
            valid_num_workers=valid_nw,
            training_logger=_lg,
        )
        perf.start()
    scheduler_steps = 0
    global_step = 0
    step_iv = max(1, d4c_log_step_interval())
    profile_step_iv = max(100, step_iv * 20)
    grad_iv = max(1, d4c_log_grad_interval())
    _finite_mode, _finite_warn = parse_d4c_finite_check_mode()
    if rank == 0 and _lg:
        if _finite_warn:
            _lg.warning(
                "[Diag] %s",
                _finite_warn,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
        _lg.info(
            "[Diag] finite_check_mode=%s（环境变量 D4C_FINITE_CHECK_MODE；默认 loss_only）",
            _finite_mode,
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[DDP] ddp_find_unused_parameters=%s（来自 FinalTrainingConfig，未在 engine 层覆盖）",
            bool(final_cfg.ddp_find_unused_parameters),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[Diag] D4C_GRAD_TOPK=%d（仅 >0 时在 GradClip 路径打印 top 参数 grad norm）",
            d4c_grad_topk(),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
    micro_step_count = 0
    save_last_checkpoint = False
    last_epoch_completed = 0
    last_ckpt_meta: Dict[str, Any] = {
        "valid_loss_total": float("nan"),
        "valid_loss_r": float("nan"),
        "valid_loss_c": float("nan"),
        "valid_loss_e": float("nan"),
        "quick_bleu4": None,
        "full_bleu4": None,
        "mainline_composite": None,
    }
    try:
        for epoch in range(epochs):
            epoch_1 = epoch + 1
            last_epoch_completed = epoch_1
            sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            if rank == 0:
                perf.epoch_start()
            model.train()
            loss_sum = torch.zeros((), dtype=torch.double, device=device)
            loss_r_sum = torch.zeros((), dtype=torch.double, device=device)
            loss_c_sum = torch.zeros((), dtype=torch.double, device=device)
            loss_e_sum = torch.zeros((), dtype=torch.double, device=device)
            n_samples = torch.zeros((), dtype=torch.double, device=device)
            micro_step_epoch = 0
            optimizer.zero_grad(set_to_none=True)
            inv_accum = 1.0 / float(A)
            iterator = train_dataloader
            if rank == 0:
                iterator = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in iterator:
                micro_step_epoch += 1
                micro_step_count += 1
                sync = micro_step_epoch % A == 0
                sync_ctx = _ddp_no_sync_model(model, world_size, sync)
                gb = require_gathered_batch(_model.gather(batch, device))
                user_idx = gb.user_idx
                item_idx = gb.item_idx
                rating = gb.rating
                tgt_input = gb.tgt_input
                tgt_output = gb.tgt_output
                domain_idx = gb.domain_idx
                exp_w = gb.exp_sample_weight
                if exp_w is None:
                    raise RuntimeError(
                        "Step5 训练 batch 缺少 exp_sample_weight：请确认 DataLoader 与 Processor 输出 7 张量含权重。"
                    )
                bsz = int(user_idx.size(0))
                warn_empty_batch(_lg, global_step=global_step, epoch=epoch_1, bsz=bsz)
                _do_step_profile = (
                    rank == 0
                    and d4c_profile_step_components_enabled()
                    and (micro_step_count % profile_step_iv == 0)
                )
                _use_cuda_prof = _do_step_profile and torch.cuda.is_available()
                _step_timer = _StepComponentCudaTimer(_do_step_profile, use_cuda_events=_use_cuda_prof)
                with sync_ctx:
                    with d4c_cuda_bf16_autocast():
                        _step_timer.start("forward")
                        pred_rating, context_dist, word_dist = model(
                            user_idx, item_idx, tgt_input, domain_idx
                        )
                        _step_timer.end("forward")
                        word_logp = F.log_softmax(word_dist, dim=-1)
                        _step_timer.start("exp_ce")
                        ls = float(final_cfg.label_smoothing)
                        loss_r_ps = F.mse_loss(pred_rating, rating, reduction="none")
                        loss_c_ps = _per_sample_mean_ce_bv_to_bt(
                            context_dist, tgt_output, ignore_index=0, label_smoothing=ls
                        )
                        loss_e_ps = per_sample_mean_ce_from_logp(
                            word_logp, tgt_output, ignore_index=0, label_smoothing=ls
                        )
                        per_sample = coef * loss_r_ps + coef * loss_c_ps + loss_e_ps

                        dom = domain_idx.view(-1)
                        w = exp_w.view(-1)
                        f_mask = (dom == 1).to(dtype=per_sample.dtype)
                        c_mask = (dom == 0).to(dtype=per_sample.dtype)
                        wf_sum = (w * f_mask).sum().clamp(min=1e-8)
                        wc_sum = (w * c_mask).sum().clamp(min=1e-8)
                        loss_factual = (per_sample * w * f_mask).sum() / wf_sum
                        loss_counterfactual = eta * ((per_sample * w * c_mask).sum() / wc_sum)
                        _step_timer.end("exp_ce")
                        w_ul = float(final_cfg.loss_weight_repeat_ul)
                        w_tc = float(final_cfg.loss_weight_terminal_clean)
                        loss_ul = word_dist.new_zeros(())
                        loss_tc = word_dist.new_zeros(())
                        _step_timer.start("repeat_ul")
                        if w_ul > 0:
                            loss_ul = d4c_anti_repeat_unlikelihood_loss_from_logp(
                                word_logp, tgt_output
                            )
                        _step_timer.end("repeat_ul")
                        _step_timer.start("terminal_clean")
                        if w_tc > 0:
                            loss_tc = d4c_terminal_cleanliness_loss(
                                word_dist,
                                tgt_output,
                                list(_model.bad_terminal_token_ids_resolved),
                                int(final_cfg.terminal_clean_span),
                            )
                        _step_timer.end("terminal_clean")
                        loss = (
                            loss_factual
                            + loss_counterfactual
                            + w_ul * loss_ul
                            + w_tc * loss_tc
                        )
                    with torch.no_grad():
                        wsum = w.sum().clamp(min=1e-8)
                        _tr = (loss_r_ps * w).sum() / wsum
                        _tc = (loss_c_ps * w).sum() / wsum
                        _te = (loss_e_ps * w).sum() / wsum
                    _step_timer.start("backward")
                    (loss * inv_accum).backward()
                    _step_timer.end("backward")
                if sync:
                    _step_timer.start("optim")
                    _log_grad = rank == 0 and _lg is not None and (global_step + 1) % grad_iv == 0
                    _pre_gn = None
                    _tops = None
                    if _log_grad:
                        _pre_gn = grad_norm_total(model.parameters())
                        _tops = grad_topk_param_norms(model, d4c_grad_topk())
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    if _log_grad:
                        _post_gn = grad_norm_total(model.parameters())
                        _tp = (
                            " top_params=" + json.dumps(_tops, ensure_ascii=False)
                            if _tops
                            else ""
                        )
                        _lg.info(
                            "[GradClip] global_step=%d epoch=%d grad_norm_pre_clip=%.6g grad_norm_post_clip=%.6g%s",
                            global_step + 1,
                            epoch_1,
                            float(_pre_gn),
                            float(_post_gn),
                            _tp,
                            extra=log_route_extra(_lg, ROUTE_DETAIL),
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # LambdaLR：必须在 optimizer.step() 之后调用，使内部 step 与全局优化步一致
                    if sched is not None:
                        sched.step()
                        scheduler_steps += 1
                    global_step += 1
                    maybe_log_grad_norm_diff_ddp(
                        model,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        global_step=global_step,
                        logger=_lg,
                        route_detail=ROUTE_DETAIL,
                    )
                    if rank == 0 and _lg and global_step > 0 and global_step % step_iv == 0:
                        _lr_now = optimizer.param_groups[0]["lr"]
                        _extra = None
                        with torch.no_grad():
                            _wsum = w.sum().clamp(min=1e-8)
                            _lr_h = (loss_r_ps * w).sum() / _wsum
                            _lc_h = (loss_c_ps * w).sum() / _wsum
                            _le_h = (loss_e_ps * w).sum() / _wsum
                        w_ul = float(final_cfg.loss_weight_repeat_ul)
                        w_tc = float(final_cfg.loss_weight_terminal_clean)
                        _lul = float(loss_ul.detach().item()) if w_ul > 0 else 0.0
                        _ltc = float(loss_tc.detach().item()) if w_tc > 0 else 0.0
                        _brk = {
                            "main": float((loss_factual + loss_counterfactual).detach().item()),
                            "weighted_repeat_ul": w_ul * _lul,
                            "weighted_terminal_clean": w_tc * _ltc,
                        }
                        if d4c_log_step_loss_parts():
                            _extra = {
                                "loss_factual": float(loss_factual.detach().item()),
                                "loss_counterfactual": float(loss_counterfactual.detach().item()),
                                "loss_r": float(_lr_h.item()),
                                "loss_c": float(_lc_h.item()),
                                "loss_e": float(_le_h.item()),
                                "loss_repeat_ul": _lul,
                                "loss_terminal_clean": _ltc,
                                "total_loss_breakdown": _brk,
                            }
                        else:
                            _extra = {
                                "loss_r": float(_lr_h.item()),
                                "loss_c": float(_lc_h.item()),
                                "loss_e": float(_le_h.item()),
                                "loss_repeat_ul": _lul,
                                "loss_terminal_clean": _ltc,
                                "total_loss_breakdown": _brk,
                            }
                        log_step_sample(
                            _lg,
                            global_step=global_step,
                            epoch=epoch_1,
                            lr=float(_lr_now),
                            train_loss_batch=float(loss.detach().item()),
                            extra=_extra,
                        )
                    _step_timer.end("optim")
                if _do_step_profile:
                    _pf = (
                        "[StepProfile] micro_step=%d sync=%s forward_ms=%.3f exp_ce_ms=%.3f "
                        "repeat_ul_ms=%.3f terminal_clean_ms=%.3f backward_ms=%.3f optim_ms=%.3f"
                    ) % (
                        micro_step_count,
                        str(bool(sync)),
                        _step_timer.ms("forward"),
                        _step_timer.ms("exp_ce"),
                        _step_timer.ms("repeat_ul"),
                        _step_timer.ms("terminal_clean"),
                        _step_timer.ms("backward"),
                        _step_timer.ms("optim"),
                    )
                    if _lg:
                        _lg.info(_pf, extra=log_route_extra(_lg, ROUTE_SUMMARY))
                    else:
                        print(_pf, flush=True)
                loss_sum = loss_sum + loss.detach().double() * bsz
                loss_r_sum = loss_r_sum + _tr.double() * bsz
                loss_c_sum = loss_c_sum + _tc.double() * bsz
                loss_e_sum = loss_e_sum + _te.double() * bsz
                n_samples += bsz
                if micro_step_count % step_iv == 0 and rank == 0:
                    run_training_finite_checks(
                        _finite_mode,
                        loss,
                        word_dist,
                        _lg,
                        global_step=global_step,
                        epoch=epoch_1,
                        route_detail=ROUTE_DETAIL,
                    )
            ddp_heartbeat(_lg, "before_train_loss_allreduce", rank=rank, epoch=epoch_1)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_r_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_c_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_e_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
            ddp_heartbeat(_lg, "after_train_loss_allreduce", rank=rank, epoch=epoch_1)
            _ns = float(n_samples.item()) if n_samples.numel() else 0.0
            avg_loss = (loss_sum / n_samples).item() if _ns > 0 else 0.0
            train_loss_r_epoch = (loss_r_sum / n_samples).item() if _ns > 0 else 0.0
            train_loss_c_epoch = (loss_c_sum / n_samples).item() if _ns > 0 else 0.0
            train_loss_e_epoch = (loss_e_sum / n_samples).item() if _ns > 0 else 0.0
            lr_epoch = optimizer.param_groups[0]["lr"]
            ddp_heartbeat(_lg, "before_gpu_stats_allgather", rank=rank, epoch=epoch_1)
            _gu, _gm, _gpeak = gather_ddp_gpu_stats_for_epoch_log(rank, world_size, int(device))
            ddp_heartbeat(_lg, "after_gpu_stats_allgather", rank=rank, epoch=epoch_1)
            if rank == 0:
                rec = perf.epoch_end(epoch + 1, len(train_dataloader), emit_log=False)
                rec["gpu_util"] = _gu
                rec["gpu_mem"] = _gm
                if _gpeak is not None:
                    rec["gpu_mem_bytes"] = _gpeak
            else:
                rec = None

            # valid：各 rank 只跑本地分片（DistributedSampler），样本加权 sum + all_reduce 得全局 avg（与 train loss 聚合方式一致）
            _t_valid0 = time.perf_counter()
            v_loss_sum, v_n, v_lr_sum, v_lc_sum, v_le_sum = validModel(
                model, valid_dataloader, device, coef=coef
            )
            if rank == 0 and _lg:
                _lg.info(
                    "[Timing] valid_loss_forward end epoch=%d elapsed_s=%.3f",
                    epoch_1,
                    time.perf_counter() - _t_valid0,
                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                )
            v_stat = torch.tensor(
                [v_loss_sum, float(v_n), v_lr_sum, v_lc_sum, v_le_sum],
                dtype=torch.double,
                device=device,
            )
            dist.all_reduce(v_stat, op=dist.ReduceOp.SUM)
            _vn = float(v_stat[1].item())
            current_valid_loss = float(v_stat[0] / v_stat[1]) if _vn > 0 else 0.0
            valid_loss_r_epoch = float(v_stat[2] / v_stat[1]) if _vn > 0 else 0.0
            valid_loss_c_epoch = float(v_stat[3] / v_stat[1]) if _vn > 0 else 0.0
            valid_loss_e_epoch = float(v_stat[4] / v_stat[1]) if _vn > 0 else 0.0
            quick_bleu4 = None
            full_bleu4_val = None
            mainline_composite_val = None
            monitor_bundle_rank0: Optional[Dict[str, Any]] = None
            sync_monitor_bundle: Optional[Dict[str, Any]] = None
            use_full_monitor = checkpoint_metric in ("bleu4", "mainline_composite")
            is_full_eval_epoch = (
                use_full_monitor
                and valid_dataset_for_bleu is not None
                and should_run_full_bleu_eval_epoch(epoch + 1, fe_sched)
            )
            if rank == 0 and _lg and use_full_monitor and valid_dataset_for_bleu is not None:
                _lg.info(
                    format_full_bleu_eval_epoch_decision_log_line(epoch_1, is_full_eval_epoch),
                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                )
            if use_full_monitor and valid_dataset_for_bleu is not None:
                if rank == 0:
                    with d4c_timing_phase(
                        _lg,
                        f"bleu_quick_epoch_{epoch_1}",
                        route=ROUTE_SUMMARY,
                        rank=0,
                    ):
                        _quick_bleu_nw = min(2, final_cfg.dataloader_num_workers_valid)
                        _qbw_env = os.environ.get("D4C_QUICK_BLEU_NUM_WORKERS", "").strip()
                        if _qbw_env:
                            try:
                                _quick_bleu_nw = max(0, int(_qbw_env))
                            except Exception:
                                if _lg:
                                    _lg.warning(
                                        "[BLEU quick] 忽略非法 D4C_QUICK_BLEU_NUM_WORKERS=%r",
                                        _qbw_env,
                                        extra=log_route_extra(_lg, ROUTE_SUMMARY),
                                    )
                        quick_bleu4 = explanation_bleu4_quick_score(
                            model,
                            tokenizer,
                            valid_dataset_for_bleu,
                            device,
                            quick_eval_max_samples,
                            rank=0,
                            logger=_lg,
                            dataloader_num_workers=_quick_bleu_nw,
                            dataloader_prefetch_factor=final_cfg.dataloader_prefetch_factor_valid,
                            collate_fn=step5_collate_fn,
                        )
                    # quick BLEU 仅用于趋势观测，不参与 checkpoint 竞争。
                if world_size > 1:
                    dist.barrier()
                if is_full_eval_epoch:
                    _t_full_bleu = time.perf_counter()
                    if checkpoint_metric == "mainline_composite":
                        mainline_composite_val, monitor_bundle_rank0 = mainline_monitor_full_valid_ddp(
                            model,
                            valid_dataset_for_bleu,
                            tokenizer=tokenizer,
                            device=device,
                            rank=rank,
                            world_size=world_size,
                            batch_size=32,
                            dataloader_num_workers=min(2, final_cfg.dataloader_num_workers_valid),
                            dataloader_prefetch_factor=final_cfg.dataloader_prefetch_factor_valid,
                            logger=_lg if rank == 0 else None,
                            collate_fn=step5_collate_fn,
                            cfg_override=full_bleu_monitor_cfg_override,
                        )
                        if monitor_bundle_rank0 is not None:
                            full_bleu4_val = float(monitor_bundle_rank0.get("bleu", {}).get("4", 0.0))
                    else:
                        full_bleu4_val = bleu4_explanation_full_valid_ddp(
                            model,
                            valid_dataset_for_bleu,
                            tokenizer=tokenizer,
                            device=device,
                            rank=rank,
                            world_size=world_size,
                            batch_size=32,
                            dataloader_num_workers=min(2, final_cfg.dataloader_num_workers_valid),
                            dataloader_prefetch_factor=final_cfg.dataloader_prefetch_factor_valid,
                            logger=_lg if rank == 0 else None,
                            collate_fn=step5_collate_fn,
                            cfg_override=full_bleu_monitor_cfg_override,
                        )
                    if rank == 0 and _lg:
                        _lg.info(
                            "[Timing] mainline_full_monitor_ddp end epoch=%d elapsed_s=%.3f",
                            epoch_1,
                            time.perf_counter() - _t_full_bleu,
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    if rank == 0 and _lg and is_full_eval_epoch:
                        _fp_mon = json.dumps(
                            full_bleu_monitor_cfg_override,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        )
                        _lg.info(
                            "[CheckpointSemantics] kind=best_mainline_candidate "
                            "checkpoint_metric=%s selection_score=%s "
                            "decode_monitor_fingerprint_sha1=%s",
                            checkpoint_metric,
                            (
                                f"{mainline_composite_val:.6f}"
                                if mainline_composite_val is not None
                                else (
                                    f"{full_bleu4_val:.6f}" if full_bleu4_val is not None else "na"
                                )
                            ),
                            hashlib.sha1(_fp_mon.encode("utf-8")).hexdigest()[:16],
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    if is_full_eval_epoch and checkpoint_metric == "mainline_composite":
                        sync_monitor_bundle = monitor_bundle_rank0
                        if world_size > 1:
                            _gathered_b: List[Any] = [None] * world_size
                            # all_gather_object 期望每个 rank 传入“单个对象”；
                            # 这里不要把对象再包一层 list，否则下游会拿到 list 而不是 dict。
                            dist.all_gather_object(
                                _gathered_b,
                                monitor_bundle_rank0 if rank == 0 else None,
                            )
                            sync_monitor_bundle = next((x for x in _gathered_b if x is not None), None)
            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bleu_line = None
                if use_full_monitor and valid_dataset_for_bleu is not None and quick_bleu4 is not None:
                    if dual_bleu:
                        fstr = f"{full_bleu4_val:.4f}" if full_bleu4_val is not None else "na"
                        ckpt_src = (
                            "full_bleu_monitor"
                            if is_full_eval_epoch
                            else "trend_quick_bleu4"
                        )
                        bleu_line = (
                            f"quick_bleu4={quick_bleu4:.4f} | full_bleu_monitor={fstr} | "
                            f"checkpoint_metric_source={ckpt_src}"
                        )
                    else:
                        bleu_line = (
                            f"quick_bleu4={quick_bleu4:.4f} | full_bleu_monitor="
                            + (
                                f"{full_bleu4_val:.4f}"
                                if full_bleu4_val is not None
                                else "na"
                            )
                            + " | checkpoint_metric_source=full_bleu_monitor (quick trend only)"
                        )
                lr_sched_line = None
                if sched is not None and ws_resolved is not None:
                    lr_sched_line = (
                        f"scheduler_type=warmup_cosine "
                        f"initial_lr={initial_lr:.6g} current_lr={lr_epoch:.6g} min_lr={min_lr_effective:.6g} "
                        f"min_lr_ratio={min_lr_ratio:.6g} warmup_steps={ws_resolved} total_steps={total_steps_plan} "
                        f"scheduler_steps_cumulative={scheduler_steps} warmup_ratio={warmup_ratio_logged:.6g}"
                    )
                block = format_epoch_training_block(
                    time_str=current_time,
                    epoch=epoch + 1,
                    epoch_time_s=rec["epoch_time"],
                    total_time_s=rec["total_time"],
                    step_time_s=rec["step_time"],
                    gpu_util=rec["gpu_util"],
                    gpu_mem=rec["gpu_mem"],
                    cpu_used=rec["cpu_used"],
                    cpu_total=rec["cpu_total"],
                    cpu_util=rec["cpu_util"],
                    lr=lr_epoch,
                    train_loss=avg_loss,
                    valid_loss=current_valid_loss,
                    adv_loss=None,
                    bleu_line=bleu_line,
                    lr_schedule_detail=lr_sched_line,
                )
                log_epoch_training_block(_lg, block)
                summ = format_epoch_summary_lines(
                    epoch=epoch + 1,
                    train_loss_total_epoch=avg_loss,
                    train_loss_r_epoch=train_loss_r_epoch,
                    train_loss_c_epoch=train_loss_c_epoch,
                    train_loss_e_epoch=train_loss_e_epoch,
                    valid_loss_total_epoch=current_valid_loss,
                    valid_loss_r_epoch=valid_loss_r_epoch,
                    valid_loss_c_epoch=valid_loss_c_epoch,
                    valid_loss_e_epoch=valid_loss_e_epoch,
                    lr=lr_epoch,
                    quick_bleu4=quick_bleu4,
                    full_bleu_monitor_bleu4=full_bleu4_val,
                    meteor=None,
                )
                log_epoch_summary_compact(_lg, summ)
                append_train_epoch_metrics_jsonl(
                    log_file=final_cfg.log_file,
                    row={
                        "epoch": epoch + 1,
                        "train_loss_total_epoch": avg_loss,
                        "train_loss_r_epoch": train_loss_r_epoch,
                        "train_loss_c_epoch": train_loss_c_epoch,
                        "train_loss_e_epoch": train_loss_e_epoch,
                        "valid_loss_total_epoch": current_valid_loss,
                        "valid_loss_r_epoch": valid_loss_r_epoch,
                        "valid_loss_c_epoch": valid_loss_c_epoch,
                        "valid_loss_e_epoch": valid_loss_e_epoch,
                        "lr": lr_epoch,
                        "quick_bleu4": quick_bleu4,
                        "full_bleu_monitor_bleu4": full_bleu4_val,
                        "mainline_composite_score": mainline_composite_val,
                    },
                )
            last_ckpt_meta["valid_loss_total"] = current_valid_loss
            last_ckpt_meta["valid_loss_r"] = valid_loss_r_epoch
            last_ckpt_meta["valid_loss_c"] = valid_loss_c_epoch
            last_ckpt_meta["valid_loss_e"] = valid_loss_e_epoch
            if rank == 0:
                last_ckpt_meta["quick_bleu4"] = quick_bleu4
                last_ckpt_meta["full_bleu4"] = full_bleu4_val
                last_ckpt_meta["mainline_composite"] = mainline_composite_val
            if dual_bleu and checkpoint_metric in ("bleu4", "mainline_composite"):
                if current_valid_loss > prev_valid_loss:
                    valid_loss_stall += 1
                    if lr_scheduler_type != "warmup_cosine":
                        learning_rate /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = learning_rate
                else:
                    valid_loss_stall = 0
                    if checkpoint_metric == "loss" and rank == 0:
                        d4c_save_checkpoint(
                            get_underlying_model(model).state_dict(),
                            str(final_cfg.save_file),
                            epoch=epoch_1,
                            reason="checkpoint_metric_loss_improved",
                            logger=_lg,
                            metadata=_step5_checkpoint_metadata(
                                final_cfg,
                                valid_loss_total=current_valid_loss,
                                valid_loss_r=valid_loss_r_epoch,
                                valid_loss_c=valid_loss_c_epoch,
                                valid_loss_e=valid_loss_e_epoch,
                                quick_bleu4=quick_bleu4,
                                full_bleu4=full_bleu4_val,
                            ),
                        )
            else:
                if current_valid_loss > prev_valid_loss:
                    enduration += 1
                    if lr_scheduler_type != "warmup_cosine":
                        learning_rate /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = learning_rate
                else:
                    enduration = 0
                    if checkpoint_metric == "loss" and rank == 0:
                        d4c_save_checkpoint(
                            get_underlying_model(model).state_dict(),
                            str(final_cfg.save_file),
                            epoch=epoch_1,
                            reason="checkpoint_metric_loss_improved",
                            logger=_lg,
                            metadata=_step5_checkpoint_metadata(
                                final_cfg,
                                valid_loss_total=current_valid_loss,
                                valid_loss_r=valid_loss_r_epoch,
                                valid_loss_c=valid_loss_c_epoch,
                                valid_loss_e=valid_loss_e_epoch,
                                quick_bleu4=quick_bleu4,
                                full_bleu4=full_bleu4_val,
                            ),
                        )
            prev_valid_loss = current_valid_loss

            if checkpoint_metric == "bleu4" and is_full_eval_epoch and full_bleu4_val is not None:
                if full_bleu4_val > best_full_bleu4:
                    best_full_bleu4 = full_bleu4_val
                    full_eval_stall = 0
                    if rank == 0:
                        d4c_save_checkpoint(
                            get_underlying_model(model).state_dict(),
                            str(final_cfg.save_file),
                            epoch=epoch_1,
                            reason="full_bleu_monitor_improved",
                            logger=_lg,
                            metadata=_step5_checkpoint_metadata(
                                final_cfg,
                                valid_loss_total=current_valid_loss,
                                valid_loss_r=valid_loss_r_epoch,
                                valid_loss_c=valid_loss_c_epoch,
                                valid_loss_e=valid_loss_e_epoch,
                                quick_bleu4=quick_bleu4,
                                full_bleu4=full_bleu4_val,
                            ),
                        )
                else:
                    if epoch + 1 >= min_epochs:
                        full_eval_stall += 1

            if (
                checkpoint_metric == "mainline_composite"
                and is_full_eval_epoch
                and sync_monitor_bundle is not None
            ):
                _upd, _sel = should_update_best_mainline(
                    sync_monitor_bundle,
                    best_mainline_gate_baseline,
                    best_composite=best_mainline_composite,
                )
                if _upd:
                    best_mainline_composite = float(sync_monitor_bundle["mainline_composite_score"])
                    best_mainline_gate_baseline = {
                        "dirty_hit_rate": float(sync_monitor_bundle["dirty_hit_rate"]),
                        "rmse_rating": float(sync_monitor_bundle["rmse_rating"]),
                        "mae_rating": float(sync_monitor_bundle["mae_rating"]),
                    }
                    full_eval_stall = 0
                    if rank == 0:
                        d4c_save_checkpoint(
                            get_underlying_model(model).state_dict(),
                            str(final_cfg.save_file),
                            epoch=epoch_1,
                            reason="mainline_composite_improved",
                            logger=_lg,
                            metadata=_step5_checkpoint_metadata(
                                final_cfg,
                                valid_loss_total=current_valid_loss,
                                valid_loss_r=valid_loss_r_epoch,
                                valid_loss_c=valid_loss_c_epoch,
                                valid_loss_e=valid_loss_e_epoch,
                                quick_bleu4=quick_bleu4,
                                full_bleu4=full_bleu4_val,
                                mainline_bundle=sync_monitor_bundle,
                            ),
                        )
                else:
                    if epoch + 1 >= min_epochs:
                        full_eval_stall += 1

            if rank == 0 and dual_bleu and checkpoint_metric in ("bleu4", "mainline_composite"):
                _cur_full = (
                    f"{full_bleu4_val:.4f}"
                    if (is_full_eval_epoch and full_bleu4_val is not None)
                    else "na"
                )
                _should_stop = (epoch + 1) >= min_epochs and (
                    full_eval_stall >= early_stop_patience_full
                )
                _es_line = (
                    f"early_stop_dual: current_full_bleu_monitor={_cur_full} "
                    f"best_full_bleu_monitor={best_full_bleu4:.4f} "
                    f"full_stall={full_eval_stall}/{early_stop_patience_full} "
                    f"valid_loss_stall={valid_loss_stall}/{early_stop_patience_loss} "
                    f"epoch={epoch + 1} min_epochs={min_epochs} should_stop={_should_stop}"
                )
                if _lg:
                    _lg.info(_es_line, extra=log_route_extra(_lg, ROUTE_SUMMARY))
                else:
                    print(_es_line, flush=True)

            ddp_heartbeat(_lg, "before_epoch_end_barrier", rank=rank, epoch=epoch_1)
            dist.barrier()
            ddp_heartbeat(_lg, "after_epoch_end_barrier", rank=rank, epoch=epoch_1)

            if dual_bleu and checkpoint_metric in ("bleu4", "mainline_composite"):
                if (epoch + 1) >= min_epochs and (
                    full_eval_stall >= early_stop_patience_full
                ):
                    save_last_checkpoint = True
                    break
            elif checkpoint_metric in ("bleu4", "mainline_composite"):
                if (epoch + 1) >= min_epochs and (
                    full_eval_stall >= early_stop_patience_full
                ):
                    save_last_checkpoint = True
                    break
            elif epoch + 1 >= min_epochs and enduration >= early_stop_patience:
                save_last_checkpoint = True
                break
        else:
            save_last_checkpoint = True
    finally:
        if rank == 0 and perf is not None:
            perf.finish()
        if save_last_checkpoint and rank == 0:
            _last_path = (final_cfg.last_checkpoint_path or "").strip()
            if not _last_path:
                _last_path = os.path.join(
                    os.path.dirname(os.path.abspath(str(final_cfg.save_file))),
                    "last.pth",
                )
            _resolved_last = os.path.abspath(os.path.expanduser(_last_path))
            d4c_save_checkpoint(
                get_underlying_model(model).state_dict(),
                _resolved_last,
                epoch=last_epoch_completed,
                reason="training_finished_last",
                logger=_lg,
                is_last=True,
                metadata=_step5_checkpoint_metadata(
                    final_cfg,
                    valid_loss_total=float(last_ckpt_meta["valid_loss_total"]),
                    valid_loss_r=float(last_ckpt_meta["valid_loss_r"]),
                    valid_loss_c=float(last_ckpt_meta["valid_loss_c"]),
                    valid_loss_e=float(last_ckpt_meta["valid_loss_e"]),
                    quick_bleu4=last_ckpt_meta.get("quick_bleu4"),
                    full_bleu4=last_ckpt_meta.get("full_bleu4"),
                    mainline_composite=last_ckpt_meta.get("mainline_composite"),
                    checkpoint_kind="last",
                ),
            )


def _step5_checkpoint_metadata(
    final_cfg: FinalTrainingConfig,
    *,
    valid_loss_total: float,
    valid_loss_r: float,
    valid_loss_c: float,
    valid_loss_e: float,
    quick_bleu4: Optional[float] = None,
    full_bleu4: Optional[float] = None,
    mainline_composite: Optional[float] = None,
    mainline_bundle: Optional[Dict[str, Any]] = None,
    checkpoint_kind: str = "best_mainline",
) -> Dict[str, Any]:
    """与权重同 stem 的 .meta.json（主路径 best_mainline / 收尾 last）。"""
    _mon = json.dumps(
        build_full_bleu_monitor_cfg_override(final_cfg),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    out: Dict[str, Any] = {
        "checkpoint_kind": str(checkpoint_kind),
        "checkpoint_selection_metric": str(final_cfg.checkpoint_metric),
        "checkpoint_selection_decode_semantics": "mainline_greedy_alignment_monitor",
        "decode_monitor_fingerprint_sha1": hashlib.sha1(_mon.encode("utf-8")).hexdigest()[:16],
        "valid_loss_total_epoch": float(valid_loss_total),
        "valid_loss_r_epoch": float(valid_loss_r),
        "valid_loss_c_epoch": float(valid_loss_c),
        "valid_loss_e_epoch": float(valid_loss_e),
        "quick_bleu4": None if quick_bleu4 is None else float(quick_bleu4),
        "full_bleu_monitor_bleu4": None if full_bleu4 is None else float(full_bleu4),
        "mainline_composite_score": None if mainline_composite is None else float(mainline_composite),
        "full_bleu_decode_strategy": str(final_cfg.full_bleu_decode_strategy),
        "decode_strategy": str(final_cfg.decode_strategy),
        "generate_temperature": float(final_cfg.generate_temperature),
        "generate_top_p": float(final_cfg.generate_top_p),
        "repetition_penalty": float(final_cfg.repetition_penalty),
        "decode_seed": final_cfg.decode_seed,
        "max_explanation_length": int(final_cfg.max_explanation_length),
        "train_label_max_length": int(getattr(final_cfg, "train_label_max_length", 128)),
        "hard_max_len": getattr(final_cfg, "hard_max_len", None),
        "soft_max_len": getattr(final_cfg, "soft_max_len", None),
    }
    if mainline_bundle is not None:
        out["mainline_monitor_snapshot"] = {
            "bleu": mainline_bundle.get("bleu"),
            "rouge": mainline_bundle.get("rouge"),
            "meteor": mainline_bundle.get("meteor"),
            "dirty_hit_rate": mainline_bundle.get("dirty_hit_rate"),
            "rmse_rating": mainline_bundle.get("rmse_rating"),
            "mae_rating": mainline_bundle.get("mae_rating"),
            "mainline_composite_score": mainline_bundle.get("mainline_composite_score"),
        }
    return out


def d4c_terminal_cleanliness_loss(
    word_logits: torch.Tensor,
    tgt: torch.Tensor,
    bad_ids: Sequence[int],
    span: int,
    *,
    pad_id: int = 0,
) -> torch.Tensor:
    """尾部若干步上压低「坏尾」词 id 的 softmax 质量（轻量）。"""
    if not bad_ids:
        return word_logits.new_zeros(())
    B, T, _V = word_logits.shape
    span = max(1, min(int(span), T))
    t0 = max(0, T - span)
    sl = word_logits[:, t0:, :]
    m = (tgt[:, t0:] != pad_id).float()
    probs = F.softmax(sl, dim=-1)
    bid = torch.tensor(list(bad_ids), device=probs.device, dtype=torch.long)
    mass = probs.index_select(-1, bid).sum(-1)
    den = m.sum().clamp(min=1.0)
    return (mass * m).sum() / den


def validModel(model, valid_dataloader, device, *, coef: float):
    _model = get_underlying_model(model)
    # forward 必须用底层 _model，避免 DDP 包装后的 model(...) 在局部执行时触发额外 NCCL collective。
    _model.eval()
    loss_sum = 0.0
    loss_r_sum = 0.0
    loss_c_sum = 0.0
    loss_e_sum = 0.0
    n_samples = 0
    c = float(coef)
    with torch.no_grad():
        for batch in valid_dataloader:
            gb = require_gathered_batch(_model.gather(batch, device))
            user_idx = gb.user_idx
            item_idx = gb.item_idx
            rating = gb.rating
            tgt_input = gb.tgt_input
            tgt_output = gb.tgt_output
            domain_idx = gb.domain_idx
            exp_w = gb.exp_sample_weight
            if exp_w is None:
                raise RuntimeError("validModel 缺少 exp_sample_weight，无法与训练损失口径对齐。")
            bsz = int(user_idx.size(0))
            with d4c_cuda_bf16_autocast():
                pred_rating, context_dist, word_dist = _model(user_idx, item_idx, tgt_input, domain_idx)
            word_logp = F.log_softmax(word_dist, dim=-1)
            ls = float(getattr(_model.exp_loss_fn, "label_smoothing", 0.0) or 0.0)
            loss_r_ps = F.mse_loss(pred_rating, rating, reduction="none")
            loss_c_ps = _per_sample_mean_ce_bv_to_bt(
                context_dist, tgt_output, ignore_index=0, label_smoothing=ls
            )
            loss_e_ps = per_sample_mean_ce_from_logp(
                word_logp, tgt_output, ignore_index=0, label_smoothing=ls
            )
            per_sample = c * loss_r_ps + c * loss_c_ps + loss_e_ps
            dom = domain_idx.view(-1)
            w = exp_w.view(-1)
            f_mask = (dom == 1).to(dtype=per_sample.dtype)
            c_mask = (dom == 0).to(dtype=per_sample.dtype)
            wf_sum = (w * f_mask).sum().clamp(min=1e-8)
            wc_sum = (w * c_mask).sum().clamp(min=1e-8)
            loss = ((per_sample * w * f_mask).sum() / wf_sum) + 0.0 * ((per_sample * w * c_mask).sum() / wc_sum)
            wsum = w.sum().clamp(min=1e-8)
            loss_r = (loss_r_ps * w).sum() / wsum
            loss_c = (loss_c_ps * w).sum() / wsum
            loss_e = (loss_e_ps * w).sum() / wsum
            loss_sum += float(loss.detach().item()) * bsz
            loss_r_sum += float(loss_r.detach().item()) * bsz
            loss_c_sum += float(loss_c.detach().item()) * bsz
            loss_e_sum += float(loss_e.detach().item()) * bsz
            n_samples += bsz
    return loss_sum, n_samples, loss_r_sum, loss_c_sum, loss_e_sum


def evalModel(model, test_dataloader, device):
    """逐 batch 推理，返回带 sample_id 的行列表（用于 DDP gather 后按 id 排序）。"""
    import time as _time_perf

    _model = get_underlying_model(model).to(device)
    _model.eval()
    rows: List[dict] = []
    decode_wall = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            _t0 = _time_perf.perf_counter()
            gb = require_gathered_batch(_model.gather(batch, device))
            user_idx = gb.user_idx
            item_idx = gb.item_idx
            rating = gb.rating
            tgt_output = gb.tgt_output
            domain_idx = gb.domain_idx
            sample_id = gb.sample_id
            with d4c_cuda_bf16_autocast():
                pred_ratings = _model.recommend(user_idx, item_idx, domain_idx)
                pred_exps, *_ = _model.generate(user_idx, item_idx, domain_idx)
            pred_texts = tokenizer.batch_decode(pred_exps, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(tgt_output, skip_special_tokens=True)
            decode_wall += _time_perf.perf_counter() - _t0
            pr = pred_ratings.detach().cpu().tolist()
            gr = rating.detach().cpu().tolist()
            sids = sample_id.detach().cpu().tolist()
            for i in range(len(sids)):
                rows.append(
                    {
                        "sample_id": int(sids[i]),
                        "pred_rating": float(pr[i]),
                        "gt_rating": float(gr[i]),
                        "pred_text": pred_texts[i],
                        "ref_text": ref_texts[i],
                        "pred_token_ids": pred_exps[i].detach().cpu().tolist(),
                        "ref_token_ids": tgt_output[i].detach().cpu().tolist(),
                    }
                )
    return {"rows": rows, "timings": {"decode_time": float(decode_wall)}}


def _load_review_by_sample_id(csv_path: str) -> Tuple[List[str], Dict[str, Any]]:
    if not (csv_path or "").strip() or not os.path.isfile(csv_path):
        return [], {"fast_path": "missing_path", "review_rows_count": 0}
    try:
        df = pd.read_csv(csv_path, usecols=["review"])
    except Exception:
        return [], {"fast_path": "read_failed", "review_rows_count": 0}
    if "review" not in df.columns:
        return [], {"fast_path": "missing_review_col", "review_rows_count": 0}
    vals = df["review"].fillna("").astype(str).tolist()
    return vals, {"fast_path": "loaded", "review_rows_count": int(len(vals))}


def _count_tokens_before_eos(ids_list: List[int], eos_id: int) -> int:
    if eos_id is not None and eos_id >= 0 and eos_id in ids_list:
        return int(ids_list.index(eos_id))
    return len(ids_list)


def evalModelWithRerank(
    model,
    test_dataloader,
    device,
    *,
    num_return_sequences: int,
    rerank_method: str,
    rerank_top_k: int,
    rerank_weight_logprob: float,
    rerank_weight_length: float,
    rerank_weight_repeat: float,
    rerank_weight_dirty: float,
    rerank_target_len_ratio: float,
    rerank_malformed_tail_penalty: float,
    rerank_malformed_token_penalty: float,
    cli_seed: int,
    review_by_sample_id: Optional[Sequence[str]] = None,
    rerank_v3_profile: Optional[Dict[str, Any]] = None,
):
    """混合候选池 + rule_v1/v2/v3 rerank；最终 pred_text 与 evalModel 对齐。"""
    import time as _time_perf

    rm = (rerank_method or "").strip().lower().replace("-", "_")
    if rm not in ("rule_v1", "rule_v2", "rule_v3"):
        raise ValueError(
            f"不支持的 rerank_method={rerank_method!r}（支持 rule_v1 / rule_v2 / rule_v3）"
        )
    v3_prof = merge_rerank_v3_profile(rerank_v3_profile)
    _m = get_underlying_model(model).to(device)
    _m.eval()
    K = max(1, int(num_return_sequences))
    strategy = str(_m.decode_strategy).lower()
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
    base_gc = _m._make_generate_config()
    fam = str(getattr(_m, "candidate_family", "balanced")).strip().lower()
    specs = build_candidate_generation_specs(
        base_gc,
        fam,
        k_cli=K,
        include_diverse=bool(getattr(_m, "candidate_mixed_include_diverse", True)),
    )
    cand_slot = 0
    rows: List[dict] = []
    batch_idx = 0
    decode_wall = 0.0
    feature_wall = 0.0
    score_wall = 0.0
    base_decode_seed = _m.decode_seed if _m.decode_seed is not None else int(cli_seed)
    review_rows = list(review_by_sample_id or [])
    with torch.no_grad():
        for batch in test_dataloader:
            gb = require_gathered_batch(_m.gather(batch, device))
            user_idx = gb.user_idx
            item_idx = gb.item_idx
            rating = gb.rating
            tgt_output = gb.tgt_output
            domain_idx = gb.domain_idx
            sample_id = gb.sample_id
            ref_texts = tokenizer.batch_decode(tgt_output, skip_special_tokens=True)
            B = int(user_idx.size(0))
            candidates_per_row: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
            _tdecode0 = _time_perf.perf_counter()
            with d4c_cuda_bf16_autocast():
                pred_ratings = _m.recommend(user_idx, item_idx, domain_idx)
                for fam_tag, cfg_ov in specs:
                    gen = None
                    if strategy == "nucleus":
                        gen = torch.Generator(device=device)
                        gen.manual_seed(
                            int(base_decode_seed) + cand_slot * 1_000_003 + batch_idx * 97
                        )
                    gen_ids, _, _, avg_lp = _m.generate_with_token_logprobs(
                        user_idx, item_idx, domain_idx, generator=gen, cfg_override=cfg_ov
                    )
                    pred_texts_k = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    eff_cfg = cfg_ov if cfg_ov is not None else base_gc
                    gen_ids_cpu = gen_ids.detach().cpu()
                    for i in range(B):
                        ids_i = gen_ids_cpu[i].tolist()
                        n_tok = _count_tokens_before_eos(ids_i, eos_id)
                        alp = float(avg_lp[i].detach().item())
                        candidates_per_row[i].append(
                            {
                                "text": pred_texts_k[i],
                                "avg_logprob": alp,
                                "lp_norm": float(compute_lp_norm(alp, n_tok)),
                                "token_len": int(n_tok),
                                "candidate_family": fam_tag,
                                "effective_temperature": float(eff_cfg.temperature),
                                "effective_top_p": float(eff_cfg.top_p),
                            }
                        )
                    cand_slot += 1
            decode_wall += _time_perf.perf_counter() - _tdecode0
            pr = pred_ratings.detach().cpu().tolist()
            gr = rating.detach().cpu().tolist()
            sids = sample_id.detach().cpu().tolist()
            _tfeat0 = _time_perf.perf_counter()
            feat_cache: List[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = [[] for _ in range(B)]
            for i in range(B):
                ref = ref_texts[i]
                ref_w = max(len(ref.split()), 1)
                ref_mean_dirty = float(ref_w)
                sid = int(sids[i])
                review_txt = review_rows[sid] if 0 <= sid < len(review_rows) else ""
                review_kw = keywords_from_source_text(review_txt)
                for rank_before, c in enumerate(candidates_per_row[i]):
                    if rm == "rule_v3":
                        feats = extract_rerank_features_for_v3(
                            c["text"],
                            avg_logprob=c["avg_logprob"],
                            token_len=int(c["token_len"]),
                            ref_mean_len_words=18.0,
                        )
                        feats_out = dict(feats)
                        ok_hard, hard_rs, sc, bkd = score_candidates_rule_v3(
                            c["text"],
                            feats,
                            review_keywords=review_kw,
                            lp_norm=float(c["lp_norm"]),
                            profile=v3_prof,
                        )
                        feats_out["rerank_score"] = round(sc, 6)
                        feats_out["v3_hard_pass"] = bool(ok_hard)
                        feats_out["v3_hard_filter_reasons"] = list(hard_rs)
                        feats_out["v3_score_breakdown"] = {k: round(float(v), 6) for k, v in bkd.items()}
                        feats_out["length_deviation_penalty"] = float(
                            bkd.get("length_penalty_v2", 0.0) or 0.0
                        )
                        feats_out["malformed_tail_penalty"] = float(
                            bkd.get("malformed_tail_penalty", 0.0) or 0.0
                        )
                        feats_out["malformed_token_penalty"] = float(
                            bkd.get("malformed_token_penalty", 0.0) or 0.0
                        )
                        len_pen = float(feats_out["length_deviation_penalty"])
                    else:
                        feats = extract_rerank_features(
                            c["text"],
                            ref,
                            avg_logprob=c["avg_logprob"],
                            ref_mean_len_words=ref_mean_dirty,
                        )
                        feats_out = dict(feats)
                        if rm == "rule_v2":
                            sc, len_pen, bkd = score_candidates_rule_v2(
                                feats,
                                weight_logprob=rerank_weight_logprob,
                                weight_length=rerank_weight_length,
                                weight_repeat=rerank_weight_repeat,
                                weight_dirty=rerank_weight_dirty,
                                target_len_ratio=rerank_target_len_ratio,
                                coef_malformed_tail=float(rerank_malformed_tail_penalty),
                                coef_malformed_token=float(rerank_malformed_token_penalty),
                            )
                            tail_ap = (
                                float(rerank_malformed_tail_penalty)
                                if feats_out.get("malformed_tail_hit")
                                else 0.0
                            )
                            tok_ap = (
                                float(rerank_malformed_token_penalty)
                                if feats_out.get("malformed_token_hit")
                                else 0.0
                            )
                            feats_out["malformed_tail_penalty"] = round(tail_ap, 6)
                            feats_out["malformed_token_penalty"] = round(tok_ap, 6)
                            feats_out["rule_v2_score_breakdown"] = {
                                k: round(float(v), 6) for k, v in bkd.items()
                            }
                        else:
                            sc, len_pen = score_candidates_rule_v1(
                                feats,
                                weight_logprob=rerank_weight_logprob,
                                weight_length=rerank_weight_length,
                                weight_repeat=rerank_weight_repeat,
                                weight_dirty=rerank_weight_dirty,
                                target_len_ratio=rerank_target_len_ratio,
                            )
                            feats_out["malformed_tail_penalty"] = 0.0
                            feats_out["malformed_token_penalty"] = 0.0
                        feats_out["length_deviation_penalty"] = round(len_pen, 6)
                        feats_out["rerank_score"] = round(sc, 6)
                    feat_cache[i].append((dict(c), feats_out))
            feature_wall += _time_perf.perf_counter() - _tfeat0
            _tsc0 = _time_perf.perf_counter()
            for i in range(B):
                sid = int(sids[i])
                scored: List[Tuple[int, Dict[str, Any], float, float, Dict[str, Any]]] = []
                for rank_before, (c, feats_out) in enumerate(feat_cache[i]):
                    _rs = feats_out.get("rerank_score", 0.0)
                    sc = float(0.0 if _rs is None else _rs)
                    _lp = feats_out.get("length_deviation_penalty", 0.0)
                    len_pen = float(0.0 if _lp is None else _lp)
                    scored.append((rank_before, feats_out, sc, len_pen, dict(c)))
                scored.sort(key=lambda x: -x[2])
                take_k = max(1, int(rerank_top_k))
                top = scored[:take_k]
                best = top[0]
                best_lp_rank = max(
                    range(len(candidates_per_row[i])),
                    key=lambda j: candidates_per_row[i][j]["avg_logprob"],
                )
                sel_text = best[4]["text"]
                cand_payload = []
                selected_rank = int(best[0])
                for rank_before, feats_out, sc, _len_pen, cdict in scored:
                    entry = {
                        "candidate_rank_before_rerank": rank_before,
                        "candidate_text": cdict["text"] if os.environ.get("D4C_RERANK_DEBUG", "0") == "1" else "",
                        "avg_logprob": cdict["avg_logprob"],
                        "lp_norm": cdict.get("lp_norm"),
                        "candidate_family": cdict.get("candidate_family"),
                        "effective_temperature": cdict.get("effective_temperature"),
                        "effective_top_p": cdict.get("effective_top_p"),
                        "token_len": cdict.get("token_len"),
                        "features": feats_out,
                        "rerank_score": feats_out["rerank_score"],
                        "selected_as_final": rank_before == selected_rank,
                    }
                    if rm == "rule_v3":
                        entry["v3_hard_pass"] = feats_out.get("v3_hard_pass")
                        entry["v3_hard_filter_reasons"] = feats_out.get("v3_hard_filter_reasons")
                        entry["v3_score_breakdown"] = feats_out.get("v3_score_breakdown")
                    cand_payload.append(entry)
                sel = best[1]
                rows.append(
                    {
                        "sample_id": sid,
                        "pred_rating": float(pr[i]),
                        "gt_rating": float(gr[i]),
                        "pred_text": sel_text,
                        "ref_text": ref,
                        "pred_token_ids": [],
                        "ref_token_ids": tgt_output[i].detach().cpu().tolist(),
                        "candidate_family": best[4].get("candidate_family"),
                        "lp_norm": best[4].get("lp_norm"),
                        "completion_ok": bool(
                            (sel.get("v3_score_breakdown") or {}).get("completion_score", 0) >= 0.9
                        )
                        if rm == "rule_v3"
                        else None,
                        "_rerank": {
                            "candidates": cand_payload,
                            "selected_rank_before_rerank": selected_rank,
                            "best_logprob_rank_before": int(best_lp_rank),
                            "rerank_method_effective": rm,
                            "selected": {
                                "text": sel_text,
                                "avg_logprob": candidates_per_row[i][selected_rank]["avg_logprob"],
                                "lp_norm": candidates_per_row[i][selected_rank].get("lp_norm"),
                                "candidate_family": candidates_per_row[i][selected_rank].get(
                                    "candidate_family"
                                ),
                                "rerank_score": sel["rerank_score"],
                                "repeat_penalty": sel.get("repeat_penalty"),
                                "dirty_penalty_diagnostic_only": sel.get(
                                    "dirty_penalty_diagnostic_only", sel.get("dirty_penalty")
                                ),
                                "length_deviation_penalty": sel.get("length_deviation_penalty"),
                                "pred_len_ratio": sel.get("pred_len_ratio"),
                                "malformed_tail_hit": bool(sel.get("malformed_tail_hit")),
                                "malformed_token_hit": bool(sel.get("malformed_token_hit")),
                                "malformed_tail_penalty": float(
                                    sel.get("malformed_tail_penalty", 0.0) or 0.0
                                ),
                                "malformed_token_penalty": float(
                                    sel.get("malformed_token_penalty", 0.0) or 0.0
                                ),
                                "v3_hard_pass": sel.get("v3_hard_pass"),
                                "v3_hard_filter_reasons": sel.get("v3_hard_filter_reasons"),
                                "v3_score_breakdown": sel.get("v3_score_breakdown"),
                            },
                            "best_by_logprob": {
                                "rank_before": int(best_lp_rank),
                                "text": candidates_per_row[i][best_lp_rank]["text"],
                                "avg_logprob": candidates_per_row[i][best_lp_rank]["avg_logprob"],
                            },
                        },
                    }
                )
            score_wall += _time_perf.perf_counter() - _tsc0
            batch_idx += 1
    return {
        "rows": rows,
        "timings": {
            "decode_time": float(decode_wall),
            "rerank_feature_time": float(feature_wall),
            "rerank_scoring_time": float(score_wall),
        },
    }


def _aggregate_rerank_summary(
    merged: List[dict],
    *,
    export_examples_mode: str,
    rerank_method: str,
) -> Dict[str, Any]:
    n = len(merged)
    if n <= 0:
        return {
            "num_samples": 0,
            "avg_candidate_count": 0.0,
            "mean_best_logprob_score": float("nan"),
            "mean_selected_rerank_score": float("nan"),
            "selected_not_best_logprob_rate": float("nan"),
            "mean_selected_len_ratio": float("nan"),
            "mean_selected_repeat_penalty": float("nan"),
            "mean_selected_dirty_penalty": float("nan"),
            "mean_selected_avg_logprob": float("nan"),
            "mean_selected_length_deviation_penalty": float("nan"),
            "mean_candidate_rouge_proxy": float("nan"),
            "mean_selected_malformed_tail_penalty": float("nan"),
            "mean_selected_malformed_token_penalty": float("nan"),
            "selected_malformed_tail_hit_rate": float("nan"),
            "selected_malformed_token_hit_rate": float("nan"),
            "export_examples_mode": export_examples_mode,
            "rerank_method": rerank_method,
            "completion_pass_rate": float("nan"),
            "well_formed_pass_rate": float("nan"),
            "source_coverage_mean": float("nan"),
            "entity_drift_hit_rate": float("nan"),
            "generic_template_hit_rate": float("nan"),
            "hard_filter_drop_rate": float("nan"),
        }

    def _float_field(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
        if key not in d:
            return default
        v = d[key]
        if v is None:
            return default
        return float(v)

    cand_counts: List[int] = []
    best_lps: List[float] = []
    sel_scores: List[float] = []
    not_best_lp = 0
    sel_lr: List[float] = []
    sel_rep: List[float] = []
    sel_dty: List[float] = []
    sel_alp: List[float] = []
    sel_ldp: List[float] = []
    sel_mtail: List[float] = []
    sel_mtok: List[float] = []
    hit_mtail = 0
    hit_mtok = 0
    rouge_proxies: List[float] = []
    rm_lc = (rerank_method or "").strip().lower().replace("-", "_")
    comp_pass = 0
    wf_pass = 0
    src_covs: List[float] = []
    drift_hits = 0
    gen_tmpl_hits = 0
    total_cands_v3 = 0
    hard_drop_cands = 0
    for r in merged:
        rr = r.get("_rerank") or {}
        cands = rr.get("candidates") or []
        cand_counts.append(len(cands))
        if cands:
            best_lps.append(
                max(
                    float(c["avg_logprob"]) if c.get("avg_logprob") is not None else float("-inf")
                    for c in cands
                )
            )
            rx = [rouge_l_proxy(str(c.get("candidate_text", "")), str(r.get("ref_text", ""))) for c in cands]
            rouge_proxies.append(sum(rx) / max(len(rx), 1))
        sel = rr.get("selected") or {}
        if sel:
            sel_scores.append(_float_field(sel, "rerank_score"))
            sel_lr.append(_float_field(sel, "pred_len_ratio"))
            sel_rep.append(_float_field(sel, "repeat_penalty"))
            dty_v = _float_field(sel, "dirty_penalty_diagnostic_only")
            if dty_v != dty_v:
                dty_v = _float_field(sel, "dirty_penalty")
            sel_dty.append(dty_v)
            sel_alp.append(_float_field(sel, "avg_logprob"))
            sel_ldp.append(_float_field(sel, "length_deviation_penalty"))
            mtp = float(sel.get("malformed_tail_penalty", 0.0) or 0.0)
            mkp = float(sel.get("malformed_token_penalty", 0.0) or 0.0)
            sel_mtail.append(mtp)
            sel_mtok.append(mkp)
            if bool(sel.get("malformed_tail_hit")) or mtp > 0:
                hit_mtail += 1
            if bool(sel.get("malformed_token_hit")) or mkp > 0:
                hit_mtok += 1
        br = int(rr.get("best_logprob_rank_before", -1))
        sr = int(rr.get("selected_rank_before_rerank", -2))
        if br >= 0 and sr >= 0 and br != sr:
            not_best_lp += 1
        if rm_lc == "rule_v3" and cands:
            for c in cands:
                total_cands_v3 += 1
                if c.get("v3_hard_pass") is False:
                    hard_drop_cands += 1
        if sel and rm_lc == "rule_v3":
            bd_sel = sel.get("v3_score_breakdown") or {}
            if bd_sel:
                if float(bd_sel.get("completion_score", 0) or 0) >= 0.9:
                    comp_pass += 1
                if float(bd_sel.get("well_formed_score", 0) or 0) >= 0.65:
                    wf_pass += 1
                scv = _float_field(bd_sel, "source_coverage_score")
                if scv == scv:
                    src_covs.append(scv)
                if float(bd_sel.get("entity_drift_penalty", 0) or 0) >= 0.55:
                    drift_hits += 1
                if float(bd_sel.get("generic_template_penalty", 0) or 0) >= 0.99:
                    gen_tmpl_hits += 1

    def _mean(xs: List[float]) -> float:
        v = [x for x in xs if x == x]
        return float(sum(v) / max(len(v), 1)) if v else float("nan")

    return {
        "num_samples": n,
        "avg_candidate_count": float(sum(cand_counts) / max(n, 1)),
        "mean_best_logprob_score": _mean(best_lps),
        "mean_selected_rerank_score": _mean(sel_scores),
        "selected_not_best_logprob_rate": float(not_best_lp) / float(n),
        "mean_selected_len_ratio": _mean(sel_lr),
        "mean_selected_repeat_penalty": _mean(sel_rep),
        "mean_selected_dirty_penalty": _mean(sel_dty),
        "mean_selected_avg_logprob": _mean(sel_alp),
        "mean_selected_length_deviation_penalty": _mean(sel_ldp),
        "mean_candidate_rouge_proxy": _mean(rouge_proxies),
        "mean_selected_malformed_tail_penalty": _mean(sel_mtail),
        "mean_selected_malformed_token_penalty": _mean(sel_mtok),
        "selected_malformed_tail_hit_rate": float(hit_mtail) / float(n),
        "selected_malformed_token_hit_rate": float(hit_mtok) / float(n),
        "export_examples_mode": export_examples_mode,
        "rerank_method": rerank_method,
        "completion_pass_rate": float(comp_pass) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "well_formed_pass_rate": float(wf_pass) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "source_coverage_mean": _mean(src_covs) if rm_lc == "rule_v3" else float("nan"),
        "entity_drift_hit_rate": float(drift_hits) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "generic_template_hit_rate": float(gen_tmpl_hits) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "hard_filter_drop_rate": float(hard_drop_cands) / float(max(total_cands_v3, 1))
        if rm_lc == "rule_v3" and total_cands_v3 > 0
        else float("nan"),
    }


def _rerank_eval_cli_resolved(args: Any) -> Dict[str, Any]:
    """eval-rerank：argparse 默认 None 表示未覆盖，数值基线与 config_loader.eval-rerank 一致。"""
    ns = getattr(args, "num_return_sequences", None)
    num_ret = 4 if ns is None else max(1, int(ns))
    rm = getattr(args, "rerank_method", None)
    rm_s = "rule_v3" if rm is None or not str(rm).strip() else str(rm).strip()
    mtk = getattr(args, "rerank_top_k", None)
    top_k = 1 if mtk is None else max(1, int(mtk))
    w_lp = 0.45 if getattr(args, "rerank_weight_logprob", None) is None else float(args.rerank_weight_logprob)
    w_len = 0.12 if getattr(args, "rerank_weight_length", None) is None else float(args.rerank_weight_length)
    w_rep = 0.18 if getattr(args, "rerank_weight_repeat", None) is None else float(args.rerank_weight_repeat)
    w_drt = 0.25 if getattr(args, "rerank_weight_dirty", None) is None else float(args.rerank_weight_dirty)
    tlr = (
        1.10 if getattr(args, "rerank_target_len_ratio", None) is None else float(args.rerank_target_len_ratio)
    )
    ex = getattr(args, "export_examples_mode", None)
    ex_mode = "head50" if ex is None or not str(ex).strip() else str(ex).strip().lower()
    mtail = (
        0.15 if getattr(args, "rerank_malformed_tail_penalty", None) is None else float(args.rerank_malformed_tail_penalty)
    )
    mtok = (
        0.18
        if getattr(args, "rerank_malformed_token_penalty", None) is None
        else float(args.rerank_malformed_token_penalty)
    )
    return {
        "num_return_sequences": num_ret,
        "rerank_method": rm_s,
        "rerank_top_k": top_k,
        "rerank_weight_logprob": w_lp,
        "rerank_weight_length": w_len,
        "rerank_weight_repeat": w_rep,
        "rerank_weight_dirty": w_drt,
        "rerank_target_len_ratio": tlr,
        "export_examples_mode": ex_mode,
        "rerank_malformed_tail_penalty": mtail,
        "rerank_malformed_token_penalty": mtok,
    }


def _eval_rows_local(
    model,
    dl,
    device,
    args,
    *,
    review_rows: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], Dict[str, float]]:
    if str(args.command) == "eval-rerank":
        v3p = None
        raw_prof = (os.environ.get("D4C_RERANK_PROFILE_JSON") or "").strip()
        if raw_prof:
            try:
                v3p = json.loads(raw_prof)
            except Exception:
                v3p = None
        _rr = _rerank_eval_cli_resolved(args)
        out = evalModelWithRerank(
            model,
            dl,
            device,
            num_return_sequences=int(_rr["num_return_sequences"]),
            rerank_method=str(_rr["rerank_method"]),
            rerank_top_k=int(_rr["rerank_top_k"]),
            rerank_weight_logprob=float(_rr["rerank_weight_logprob"]),
            rerank_weight_length=float(_rr["rerank_weight_length"]),
            rerank_weight_repeat=float(_rr["rerank_weight_repeat"]),
            rerank_weight_dirty=float(_rr["rerank_weight_dirty"]),
            rerank_target_len_ratio=float(_rr["rerank_target_len_ratio"]),
            rerank_malformed_tail_penalty=float(_rr["rerank_malformed_tail_penalty"]),
            rerank_malformed_token_penalty=float(_rr["rerank_malformed_token_penalty"]),
            cli_seed=int(args.seed),
            review_by_sample_id=review_rows,
            rerank_v3_profile=v3p,
        )
        return out["rows"], dict(out.get("timings") or {})
    out_m = evalModel(model, dl, device)
    return out_m["rows"], dict(out_m.get("timings") or {})


_RERANK_TOP2_GAP = 0.025


def _rerank_compact_cand(c: Dict[str, Any], *, text_max: int = 200) -> Dict[str, Any]:
    fe = c.get("features") or {}
    t = str(c.get("candidate_text") or "")
    if len(t) > text_max:
        t = t[:text_max] + "…"
    out: Dict[str, Any] = {
        "candidate_rank_before_rerank": c.get("candidate_rank_before_rerank"),
        "avg_logprob": c.get("avg_logprob"),
        "rerank_score": c.get("rerank_score"),
        "pred_len_ratio": fe.get("pred_len_ratio"),
        "dirty_penalty_diagnostic_only": fe.get(
            "dirty_penalty_diagnostic_only", fe.get("dirty_penalty")
        ),
        "malformed_tail_penalty": fe.get("malformed_tail_penalty", 0),
        "malformed_token_penalty": fe.get("malformed_token_penalty", 0),
        "text_preview": t,
    }
    bd = fe.get("rule_v2_score_breakdown")
    if bd:
        out["rule_v2_score_breakdown"] = bd
    dr = fe.get("dirty_detail_v2")
    if isinstance(dr, dict):
        out["dirty_detail_v2"] = {"active_rules": dr.get("active_rules", [])}
    return out


def _rerank_build_changed_only_row(r: dict) -> Tuple[bool, Dict[str, Any]]:
    rr = r.get("_rerank") or {}
    cands = list(rr.get("candidates") or [])
    sid = int(r["sample_id"])
    sel = rr.get("selected") or {}
    best_lp = rr.get("best_by_logprob") or {}
    br = int(rr.get("best_logprob_rank_before", -1))
    sr = int(rr.get("selected_rank_before_rerank", -2))
    selected_not_best_lp = bool(br >= 0 and sr >= 0 and br != sr)

    scores = sorted(
        [float((c.get("rerank_score") if c.get("rerank_score") is not None else 0.0) or 0.0) for c in cands],
        reverse=True,
    )
    top12_gap = (scores[0] - scores[1]) if len(scores) >= 2 else 1.0
    tight_top2 = bool(len(scores) >= 2 and top12_gap < _RERANK_TOP2_GAP)

    if cands:
        lp_winner = max(
            cands,
            key=lambda x: float(x.get("avg_logprob"))
            if x.get("avg_logprob") is not None
            else float("-inf"),
        )
        lp_rank = int(lp_winner.get("candidate_rank_before_rerank", -1))
    else:
        lp_rank = -1
    rerank_differs_from_lp_winner = bool(lp_rank >= 0 and sr >= 0 and lp_rank != sr)

    dirty = float(sel.get("dirty_penalty_diagnostic_only", sel.get("dirty_penalty", 0.0)) or 0.0)
    ldp = float(sel.get("length_deviation_penalty", 0.0) or 0.0)
    mtail_ap = float(sel.get("malformed_tail_penalty", 0.0) or 0.0)
    mtok_ap = float(sel.get("malformed_token_penalty", 0.0) or 0.0)
    mtail_hit = bool(sel.get("malformed_tail_hit"))
    mtok_hit = bool(sel.get("malformed_token_hit"))

    veto_clean_lp = False
    if dirty >= 0.15 and cands:
        lp_text = str(best_lp.get("text") or "")
        lp_c = next((c for c in cands if str(c.get("candidate_text") or "") == lp_text), None)
        if lp_c:
            lp_d = float(
                (
                    (lp_c.get("features") or {}).get(
                        "dirty_penalty_diagnostic_only",
                        (lp_c.get("features") or {}).get("dirty_penalty", 1.0),
                    )
                )
                or 1.0
            )
            if lp_d < 0.05:
                veto_clean_lp = True

    flags: List[str] = []
    if selected_not_best_lp:
        flags.append("selected_not_best_logprob")
    if dirty > 0:
        flags.append("selected_dirty")
    if mtail_ap > 0 or mtail_hit:
        flags.append("malformed_tail")
    if mtok_ap > 0 or mtok_hit:
        flags.append("malformed_token")
    if ldp > 0.30:
        flags.append("length_dev_high")
    if rerank_differs_from_lp_winner:
        flags.append("rerank_not_logprob_winner")
    if tight_top2:
        flags.append("tight_top2_scores")
    if veto_clean_lp:
        flags.append("veto_dirty_selected_vs_clean_logprob")

    include = bool(
        selected_not_best_lp
        or dirty > 0
        or mtail_ap > 0
        or mtok_ap > 0
        or mtail_hit
        or mtok_hit
        or ldp > 0.30
        or rerank_differs_from_lp_winner
        or tight_top2
        or veto_clean_lp
    )
    scored_c = sorted(
        cands,
        key=lambda x: -float(x["rerank_score"])
        if x.get("rerank_score") is not None
        else float("inf"),
    )
    top_compact = [_rerank_compact_cand(x) for x in scored_c[:3]]
    rec: Dict[str, Any] = {
        "sample_id": sid,
        "reference": r.get("ref_text"),
        "selected": {
            "text": sel.get("text"),
            "avg_logprob": sel.get("avg_logprob"),
            "rerank_score": sel.get("rerank_score"),
            "dirty_penalty_diagnostic_only": sel.get(
                "dirty_penalty_diagnostic_only", sel.get("dirty_penalty")
            ),
            "malformed_tail_penalty": mtail_ap,
            "malformed_token_penalty": mtok_ap,
            "length_deviation_penalty": ldp,
        },
        "best_by_logprob": best_lp,
        "comparison": {
            "rerank_selected_text": sel.get("text"),
            "logprob_winner_text": best_lp.get("text"),
            "selected_not_best_logprob": selected_not_best_lp,
            "top1_top2_rerank_gap": round(top12_gap, 6),
        },
        "top_candidates_compact": top_compact,
        "analysis_flags": flags,
    }
    return include, rec


def _rerank_head50_sort_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    flags = set(rec.get("analysis_flags") or [])
    sel = rec.get("selected") or {}
    _g = (rec.get("comparison") or {}).get("top1_top2_rerank_gap", 1.0)
    gap = float(1.0 if _g is None else _g)
    dirty = float(sel.get("dirty_penalty_diagnostic_only", sel.get("dirty_penalty", 0.0)) or 0.0)
    return (
        0 if "malformed_token" in flags else 1,
        0 if "malformed_tail" in flags else 1,
        0 if "selected_not_best_logprob" in flags else 1,
        -dirty,
        gap,
    )


def _write_rerank_artifacts(
    eval_sub: str,
    merged: List[dict],
    *,
    rerank_cfg: Dict[str, Any],
    rerank_summary: Dict[str, Any],
    export_examples_mode: str,
    export_full_rerank_examples: bool,
) -> None:
    import csv as _csv

    mode = (export_examples_mode or "head50").strip().lower()
    cand_path = os.path.join(eval_sub, "rerank_candidates.csv")
    fieldnames = [
        "sample_id",
        "candidate_rank_before_rerank",
        "candidate_text",
        "avg_logprob",
        "pred_len_words",
        "pred_len_ratio",
        "repeat_penalty",
        "dirty_penalty_diagnostic_only",
        "length_deviation_penalty",
        "malformed_tail_penalty",
        "malformed_token_penalty",
        "rerank_score",
        "selected_as_final",
    ]
    full_samples: List[Dict[str, Any]] = []
    with open(cand_path, "w", newline="", encoding="utf-8") as cf:
        w = _csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in merged:
            sid = int(r["sample_id"])
            rr = r.get("_rerank") or {}
            for c in rr.get("candidates") or []:
                fe = c.get("features") or {}
                w.writerow(
                    {
                        "sample_id": sid,
                        "candidate_rank_before_rerank": c.get("candidate_rank_before_rerank"),
                        "candidate_text": c.get("candidate_text"),
                        "avg_logprob": fe.get("avg_logprob"),
                        "pred_len_words": fe.get("pred_len_words"),
                        "pred_len_ratio": fe.get("pred_len_ratio"),
                        "repeat_penalty": fe.get("repeat_penalty"),
                        "dirty_penalty_diagnostic_only": fe.get(
                            "dirty_penalty_diagnostic_only", fe.get("dirty_penalty")
                        ),
                        "length_deviation_penalty": fe.get("length_deviation_penalty"),
                        "malformed_tail_penalty": fe.get("malformed_tail_penalty", 0),
                        "malformed_token_penalty": fe.get("malformed_token_penalty", 0),
                        "rerank_score": c.get("rerank_score"),
                        "selected_as_final": c.get("selected_as_final"),
                    }
                )
            full_samples.append(
                {
                    "sample_id": sid,
                    "reference": r.get("ref_text"),
                    "candidates": rr.get("candidates"),
                    "selected": rr.get("selected"),
                    "best_by_logprob": rr.get("best_by_logprob"),
                    "comparison": {
                        "rerank_selected_text": (rr.get("selected") or {}).get("text"),
                        "logprob_winner_text": (rr.get("best_by_logprob") or {}).get("text"),
                    },
                }
            )

    want_light = mode in ("changed_only", "head20", "head50")
    changed_records: List[Dict[str, Any]] = []
    if want_light:
        for r in merged:
            inc, rec = _rerank_build_changed_only_row(r)
            if inc:
                changed_records.append(rec)
        jlp = os.path.join(eval_sub, "rerank_examples_changed_only.jsonl")
        with open(jlp, "w", encoding="utf-8") as jlf:
            for rec in changed_records:
                jlf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        idx_p = os.path.join(eval_sub, "rerank_examples_changed_only_index.csv")
        with open(idx_p, "w", newline="", encoding="utf-8") as ixf:
            iw = _csv.DictWriter(
                ixf,
                fieldnames=[
                    "sample_id",
                    "selected_not_best_logprob",
                    "dirty_penalty_diagnostic_only",
                    "malformed_tail_penalty",
                    "malformed_token_penalty",
                    "flags_summary",
                ],
            )
            iw.writeheader()
            for rec in changed_records:
                comp = rec.get("comparison") or {}
                sel = rec.get("selected") or {}
                iw.writerow(
                    {
                        "sample_id": rec.get("sample_id"),
                        "selected_not_best_logprob": comp.get("selected_not_best_logprob"),
                        "dirty_penalty_diagnostic_only": sel.get(
                            "dirty_penalty_diagnostic_only", sel.get("dirty_penalty")
                        ),
                        "malformed_tail_penalty": sel.get("malformed_tail_penalty"),
                        "malformed_token_penalty": sel.get("malformed_token_penalty"),
                        "flags_summary": ";".join(rec.get("analysis_flags") or []),
                    }
                )
        _head_limits = {"head20": 20, "head50": 50}
        if mode in _head_limits:
            n = _head_limits[mode]
            head = sorted(changed_records, key=_rerank_head50_sort_key)[:n]
            with open(
                os.path.join(eval_sub, f"rerank_examples_{mode}.json"),
                "w",
                encoding="utf-8",
            ) as hf:
                json.dump(head, hf, ensure_ascii=False, indent=2, default=str)

    if export_full_rerank_examples:
        gz_path = os.path.join(eval_sub, "rerank_examples.json.gz")
        blob = json.dumps(
            {
                "rerank_cfg": rerank_cfg,
                "rerank_summary": rerank_summary,
                "samples": full_samples,
            },
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        with gzip.open(gz_path, "wb", compresslevel=6) as gzf:
            gzf.write(blob)


def _load_profile_tensors(auxiliary, target, device_idx):
    sdomain = torch.tensor(
        np.load(os.path.join(get_data_dir(), auxiliary, "domain.npy")), dtype=torch.float, device=device_idx,
    )
    tdomain = torch.tensor(
        np.load(os.path.join(get_data_dir(), target, "domain.npy")), dtype=torch.float, device=device_idx,
    )
    domain_profiles = torch.cat([sdomain.unsqueeze(0), tdomain.unsqueeze(0)], dim=0)
    # 与 Step4 / adv_train_core / aug_train 一致：target 行用 [0..Nt) 的 user/item_idx；
    # auxiliary 行在全局索引上接在 target 之后（Nt..Nt+Ns)，故 profile 须 torch.cat([target, auxiliary], dim=0)。
    tuser = torch.tensor(
        np.load(os.path.join(get_data_dir(), target, "user_profiles.npy")), dtype=torch.float, device=device_idx,
    )
    suser = torch.tensor(
        np.load(os.path.join(get_data_dir(), auxiliary, "user_profiles.npy")), dtype=torch.float, device=device_idx,
    )
    titem = torch.tensor(
        np.load(os.path.join(get_data_dir(), target, "item_profiles.npy")), dtype=torch.float, device=device_idx,
    )
    sitem = torch.tensor(
        np.load(os.path.join(get_data_dir(), auxiliary, "item_profiles.npy")), dtype=torch.float, device=device_idx,
    )
    user_profiles = torch.cat([tuser, suser], dim=0)
    item_profiles = torch.cat([titem, sitem], dim=0)
    return domain_profiles, user_profiles, item_profiles


def _make_model(final_cfg: FinalTrainingConfig, args, device_idx):
    domain_profiles, user_profiles, item_profiles = _load_profile_tensors(args.auxiliary, args.target, device_idx)
    m = Model(
        final_cfg.nuser,
        final_cfg.nitem,
        final_cfg.ntoken,
        final_cfg.emsize,
        final_cfg.nhead,
        final_cfg.nhid,
        final_cfg.nlayers,
        final_cfg.dropout,
        user_profiles,
        item_profiles,
        domain_profiles,
        label_smoothing=float(final_cfg.label_smoothing),
    ).to(device_idx)
    m.apply_runtime_config(final_cfg, tokenizer)
    return m


def _safe_file_mtime(path: str) -> str:
    try:
        if not os.path.isfile(path):
            return "missing"
        return str(int(os.path.getmtime(path)))
    except OSError:
        return "na"


def _tokenizer_cache_identity(tok) -> str:
    nop = getattr(tok, "name_or_path", None) or getattr(tok, "name", None)
    if nop:
        return str(nop)
    return type(tok).__name__


def _build_tokenize_cache_fingerprint(
    *,
    train_path: str,
    eval_split_path: str,
    tok,
    max_length: int,
    cache_version: str,
    eval_only: bool,
) -> str:
    if eval_only:
        parts = [
            f"eval_split={os.path.abspath(eval_split_path)}",
            f"eval_mtime={_safe_file_mtime(eval_split_path)}",
            f"tok={_tokenizer_cache_identity(tok)}",
            f"maxlen={int(max_length)}",
            f"ver={cache_version}",
        ]
    else:
        parts = [
            f"train={os.path.abspath(train_path)}",
            f"train_mtime={_safe_file_mtime(train_path)}",
            f"valid={os.path.abspath(eval_split_path)}",
            f"valid_mtime={_safe_file_mtime(eval_split_path)}",
            f"tok={_tokenizer_cache_identity(tok)}",
            f"maxlen={int(max_length)}",
            f"ver={cache_version}",
        ]
    raw = "|".join(parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{cache_version}_{h}"


def _build_step5_cache_dir(
    ckpt_task_dir: str,
    train_path: str,
    eval_split_path: str,
    processor,
    tok,
    *,
    eval_only: bool,
    cache_version: str = D4C_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str]:
    fp = _build_tokenize_cache_fingerprint(
        train_path=train_path,
        eval_split_path=eval_split_path,
        tok=tok,
        max_length=int(getattr(processor, "max_length", 25)),
        cache_version=cache_version,
        eval_only=eval_only,
    )
    prefix = "hf_cache_step5_eval" if eval_only else "hf_cache_step5"
    cache_dir = os.path.join(ckpt_task_dir, f"{prefix}_{fp}")
    return cache_dir, fp


def _dist_barrier_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _hf_dataset_cache_ready(cache_dir: str) -> bool:
    return os.path.isdir(cache_dir) and os.path.isfile(os.path.join(cache_dir, "dataset_dict.json"))


def _log_step5_tokenize_line(msg: str) -> None:
    lg = logging.getLogger(LOGGER_NAME)
    if not logger_has_file_handler(lg):
        print(msg, flush=True)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


def _log_tokenize_map_done_step5(phase: str, nproc: int, elapsed_s: float) -> None:
    msg = f"[Tokenize] {phase} 完成 | num_proc={nproc} | wall_time={elapsed_s:.2f}s"
    _log_step5_tokenize_line(msg)


def _step5_map_or_load_tokenize_cache(
    *,
    datasets: DatasetDict,
    processor,
    nproc: int,
    cache_dir: str,
    cache_fingerprint: str,
    rank: int,
    show_datasets_progress: bool,
    log_tokenize: bool,
    phase: str,
) -> DatasetDict:
    """rank0 map + save；barrier 后各 rank load_from_disk；cache 命中则直接 load。"""
    if _hf_dataset_cache_ready(cache_dir):
        t_hit0 = time.perf_counter()
        encoded_data = load_from_disk(cache_dir)
        elapsed_hit = time.perf_counter() - t_hit0
        if rank == 0 and log_tokenize:
            msg = (
                f"[Tokenize] {phase} cache hit | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"load_wall_time={elapsed_hit:.2f}s"
            )
            _log_step5_tokenize_line(msg)
        return encoded_data

    if rank == 0:
        t0 = time.perf_counter()
        with hf_datasets_progress_bar(show_datasets_progress):
            encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
        encoded_data.save_to_disk(cache_dir)
        elapsed = time.perf_counter() - t0
        if log_tokenize:
            _log_tokenize_map_done_step5(phase, nproc, elapsed)
            msg = (
                f"[Tokenize] {phase} cache miss | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"build_wall_time={elapsed:.2f}s"
            )
            _log_step5_tokenize_line(msg)

    _dist_barrier_if_initialized()
    encoded_data = load_from_disk(cache_dir)
    return encoded_data


def _decode_profile_to_training_patch(blob: Dict[str, Any]) -> Dict[str, Any]:
    """将 decode profile 的解码控制字段映射到 FinalTrainingConfig.replace 补丁。"""
    out: Dict[str, Any] = {}
    if not blob:
        return out

    def _opt_int(k: str) -> Optional[int]:
        v = blob.get(k)
        if v is None or v == "":
            return None
        try:
            i = int(v)
            return i if i > 0 else None
        except Exception:
            return None

    sm = _opt_int("soft_max_len")
    if sm is not None:
        out["soft_max_len"] = sm
    hm = _opt_int("hard_max_len")
    if hm is not None:
        out["hard_max_len"] = hm
    if "eos_boost_start" in blob:
        try:
            out["eos_boost_start"] = int(blob["eos_boost_start"])
        except Exception:
            pass
    if "eos_boost_value" in blob:
        try:
            out["eos_boost_value"] = float(blob["eos_boost_value"])
        except Exception:
            pass
    for fk in ("tail_temperature", "tail_top_p"):
        if fk in blob:
            try:
                out[fk] = float(blob[fk])
            except Exception:
                pass
    for bk in (
        "forbid_eos_after_open_quote",
        "forbid_eos_after_open_bracket",
        "forbid_bad_terminal_tokens",
        "candidate_mixed_include_diverse",
    ):
        if bk in blob:
            out[bk] = bool(blob[bk])
    if "decode_token_repeat_window" in blob:
        try:
            out["decode_token_repeat_window"] = max(1, int(blob["decode_token_repeat_window"]))
        except Exception:
            pass
    if "decode_token_repeat_max" in blob:
        try:
            out["decode_token_repeat_max"] = max(1, int(blob["decode_token_repeat_max"]))
        except Exception:
            pass
    cf = blob.get("candidate_family")
    if cf is not None and str(cf).strip():
        out["candidate_family"] = str(cf).strip().lower()
    bt = blob.get("bad_terminal_token_ids")
    if isinstance(bt, list) and bt:
        try:
            out["bad_terminal_token_ids"] = tuple(int(x) for x in bt)
        except Exception:
            pass
    return out


_STEP5_TRAIN_REQUIRED_COLS = (
    "clean_text",
    "sample_origin",
    "train_keep",
    "sample_weight_hint",
)
_STEP5_IGNORED_FIELDS = (
    "adversarial_coef",
    "adversarial_alpha",
    "adversarial_beta",
    "adversarial_schedule_enabled",
    "adversarial_start_epoch",
    "adversarial_warmup_epochs",
    "adversarial_coef_target",
)


def _require_step5_train_csv_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _STEP5_TRAIN_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Step5 训练 CSV 缺少必需列: "
            + ", ".join(missing)
            + "。请使用新版 Step4 导出的 factuals_counterfactuals.csv（禁止静默回退 explanation）。"
        )


def _rank0_step5_train_data_audit(
    raw_df: pd.DataFrame,
    filt_df: pd.DataFrame,
    *,
    train_label_max_length: int,
    train_dynamic_padding: bool,
    train_padding_strategy: str,
    log_path: Optional[str],
    ddp_find_unused_parameters_effective: Optional[bool] = None,
) -> None:
    """终端 + step5_train_data_audit.json + 简要 csv。"""
    n_raw = len(raw_df)
    n_filt = len(filt_df)
    msg_lines = [
        f"[Step5 数据审计] 过滤后训练行数={n_filt}（过滤前={n_raw}，条件 train_keep==1 且 clean_text 非空）",
        f"  train_label_max_length={train_label_max_length}",
        f"  train_dynamic_padding={1 if train_dynamic_padding else 0}",
        f"  train_padding_strategy={train_padding_strategy}",
    ]
    if "sample_origin" in raw_df.columns and n_raw > 0:
        vc = raw_df["sample_origin"].value_counts()
        for k, v in vc.items():
            msg_lines.append(f"  过滤前 sample_origin {k}: {int(v)} ({100.0 * float(v) / n_raw:.2f}%)")
    if "sample_origin" in filt_df.columns and n_filt > 0:
        vc2 = filt_df["sample_origin"].value_counts()
        msg_lines.append("  过滤后按来源:")
        for k, v in vc2.items():
            msg_lines.append(f"    {k}: {int(v)} ({100.0 * float(v) / n_filt:.2f}%)")
        if "sample_weight_hint" in filt_df.columns:
            for org in sorted(vc2.keys(), key=lambda x: str(x)):
                sub = filt_df[filt_df["sample_origin"] == org]
                mw = float(sub["sample_weight_hint"].astype(float).mean())
                msg_lines.append(f"    {org} sample_weight_hint 均值={mw:.6f}")

    for flag, label in (
        ("html_entity_hit", "HTML 实体命中"),
        ("bad_tail_hit", "bad_tail 命中"),
        ("template_hit", "template 命中"),
    ):
        if flag in raw_df.columns:
            msg_lines.append(f"  过滤前 {label}: {int(raw_df[flag].sum())}")
    if "template_hit" in raw_df.columns:
        t_total = int((raw_df["template_hit"] == 1).sum())
        t_kept = int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 1)).sum())
        t_drop = int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 0)).sum())
        t_dw = int(
            ((raw_df["template_hit"] == 1) & (raw_df.get("template_downweighted", 0) == 1)).sum()
        )
        msg_lines.append(
            "  template_hit 审计: "
            f"total={t_total}, kept={t_kept}, downweighted={t_dw}, dropped={t_drop}"
        )

    if "train_keep" in raw_df.columns:
        msg_lines.append(f"  train_keep==0 行数: {int((raw_df['train_keep'] == 0).sum())}")

    topn: List[tuple[Any, int]] = []
    if "train_drop_reason" in raw_df.columns:
        dr = raw_df.loc[raw_df["train_keep"] == 0, "train_drop_reason"].fillna("").astype(str)
        ctr = Counter([x for x in dr.tolist() if x])
        topn = ctr.most_common(15)
        msg_lines.append(
            "  train_drop_reason Top-N: " + ", ".join(f"{a}:{b}" for a, b in topn) if topn else "  train_drop_reason Top-N: (无)"
        )

    tok_lens: List[int] = []
    truncated = 0
    word_lens: List[int] = []
    # 审计需统计「未按 train_label 截断」的原始 token 数；若超过 T5 默认 model_max_length（512），
    # HF 会对 truncation=False 打告警。训练路径中 Processor 已 truncation=True，不会把超长序列喂进模型。
    _prev_mml = int(getattr(tokenizer, "model_max_length", 512) or 512)
    try:
        tokenizer.model_max_length = 1_000_000
        for _, row in filt_df.iterrows():
            ct = str(row.get("clean_text", "") or "")
            word_lens.append(len(ct.split()))
            ids = tokenizer(ct, add_special_tokens=True, truncation=False)["input_ids"]
            L = len(ids)
            tok_lens.append(L)
            if L > train_label_max_length:
                truncated += 1
    finally:
        tokenizer.model_max_length = _prev_mml

    if tok_lens:
        arr = np.asarray(tok_lens, dtype=np.int64)
        warr = np.asarray(word_lens, dtype=np.int64)
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99]).astype(int)
        msg_lines.append(f"  clean_text token 长度 p50={p50} p90={p90} p95={p95} p99={p99}")
        msg_lines.append(
            f"  超过 train_label_max_length 将被截断: {truncated} / {len(tok_lens)} "
            f"({100.0 * truncated / max(len(tok_lens), 1):.2f}%)"
        )
        w50, w90 = np.percentile(warr, [50, 90]).astype(int)
        msg_lines.append(f"  clean_text 词数 p50={w50} p90={w90}")

    text_block = "\n".join(msg_lines)
    print(text_block, flush=True)
    _lg_a = logging.getLogger(LOGGER_NAME)
    if _lg_a.handlers:
        _lg_a.info(text_block)
    else:
        logging.info(text_block)

    _tg_raw = int((raw_df["sample_origin"] == "target_gold").sum()) if "sample_origin" in raw_df.columns else 0
    _tg_filt = int((filt_df["sample_origin"] == "target_gold").sum()) if "sample_origin" in filt_df.columns else 0
    _cf_raw = int((raw_df["sample_origin"] == "aux_cf").sum()) if "sample_origin" in raw_df.columns else 0
    _cf_filt = int((filt_df["sample_origin"] == "aux_cf").sum()) if "sample_origin" in filt_df.columns else 0
    _t_sev_drop = (
        int((raw_df["train_drop_reason"].astype(str) == "severe_template").sum())
        if "train_drop_reason" in raw_df.columns
        else 0
    )
    _t_med_dw = (
        int(
            ((raw_df.get("template_downweighted", 0) == 1) & (raw_df["train_keep"] == 1)).sum()
        )
        if "train_keep" in raw_df.columns
        else 0
    )
    _nt_dw = (
        int((raw_df.get("noisy_tail_downweighted", 0) == 1).sum())
        if "noisy_tail_downweighted" in raw_df.columns
        else 0
    )
    _nt_drop = (
        int(
            (
                (raw_df.get("repeat_tail_hit", 0) == 1)
                & (raw_df["train_keep"] == 0)
                & (raw_df["train_drop_reason"].astype(str) != "severe_template")
            ).sum()
        )
        if "train_keep" in raw_df.columns and "train_drop_reason" in raw_df.columns
        else 0
    )
    msg_lines.append(
        f"  治理统计 template_severe_dropped={_t_sev_drop} template_medium_downweighted={_t_med_dw} "
        f"noisy_tail_downweighted={_nt_dw} noisy_tail_dropped_other={_nt_drop}"
    )
    msg_lines.append(
        f"  target_gold_kept_ratio={(_tg_filt / max(_tg_raw, 1)):.4f} "
        f"aux_cf_kept_ratio={(_cf_filt / max(_cf_raw, 1)):.4f}"
    )

    audit_obj: Dict[str, Any] = {
        "schema_version": "d4c_step5_train_audit/1.1",
        "n_rows_before_filter": n_raw,
        "n_rows_after_filter": n_filt,
        "train_label_max_length": int(train_label_max_length),
        "train_dynamic_padding": bool(train_dynamic_padding),
        "train_padding_strategy": str(train_padding_strategy),
        "drop_reason_top": {str(k): int(v) for k, v in topn},
        "template_severe_dropped": _t_sev_drop,
        "template_medium_downweighted": _t_med_dw,
        "noisy_tail_downweighted": _nt_dw,
        "noisy_tail_dropped": _nt_drop,
        "target_gold_kept_ratio": float(_tg_filt / max(_tg_raw, 1)),
        "aux_cf_kept_ratio": float(_cf_filt / max(_cf_raw, 1)),
        "training_diagnostics": training_diagnostics_snapshot(
            diagnostics_scope="child",
            effective_training_payload_json=os.environ.get("D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""),
            ddp_find_unused_parameters_effective=ddp_find_unused_parameters_effective,
        ),
    }
    if "template_hit" in raw_df.columns:
        audit_obj["template_hit_audit"] = {
            "template_hit_total": int((raw_df["template_hit"] == 1).sum()),
            "template_hit_kept": int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 1)).sum()),
            "template_hit_dropped": int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 0)).sum()),
            "template_hit_downweighted": int(
                ((raw_df["template_hit"] == 1) & (raw_df.get("template_downweighted", 0) == 1)).sum()
            ),
        }
    if tok_lens:
        arr = np.asarray(tok_lens, dtype=np.int64)
        warr = np.asarray(word_lens, dtype=np.int64)
        audit_obj["truncation_over_max"] = {
            "count": int(truncated),
            "frac": float(truncated / max(len(tok_lens), 1)),
        }
        audit_obj["token_len_quantiles"] = {
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
        }
        audit_obj["word_len_quantiles"] = {
            "p50": int(np.percentile(warr, 50)),
            "p90": int(np.percentile(warr, 90)),
        }

    if log_path:
        log_dir = os.path.dirname(os.path.abspath(os.path.expanduser(log_path)))
        os.makedirs(log_dir, exist_ok=True)
        jpath = os.path.join(log_dir, "step5_train_data_audit.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(audit_obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        csv_path = os.path.join(log_dir, "step5_train_data_audit_summary.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write(f"n_rows_before_filter,{n_raw}\n")
            f.write(f"n_rows_after_filter,{n_filt}\n")
            f.write(f"truncation_over_max_count,{truncated}\n")
            f.write(f"template_severe_dropped,{_t_sev_drop}\n")
            f.write(f"template_medium_downweighted,{_t_med_dw}\n")
            f.write(f"noisy_tail_downweighted,{_nt_dw}\n")
            f.write(f"noisy_tail_dropped,{_nt_drop}\n")
            f.write(f"target_gold_kept_ratio,{_tg_filt / max(_tg_raw, 1):.6f}\n")
            f.write(f"aux_cf_kept_ratio,{_cf_filt / max(_cf_raw, 1):.6f}\n")


def build_d4c_ddp_artefacts(
    args,
    world_size,
    local_rank,
    rank,
    *,
    command: str,
    show_datasets_progress: bool = True,
):
    _dataset_build_t0 = time.perf_counter()
    task_idx = resolve_task_idx_from_aux_target(args.auxiliary, args.target)
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")
    eval_only = command != "train"
    _ro = collect_training_hardware_overrides_from_args(args)
    resolved = build_resolved_training_config(
        args,
        task_idx=task_idx,
        world_size=world_size,
        hardware_overrides=_ro,
    )
    path = os.path.join(get_data_dir(), args.target)
    _ckpt_task = get_stage_run_dir(task_idx)
    os.makedirs(_ckpt_task, exist_ok=True)
    train_path = os.path.join(_ckpt_task, "factuals_counterfactuals.csv")
    valid_path = os.path.join(path, "valid.csv")
    test_path = os.path.join(path, "test.csv")
    if command == "test":
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"缺少测试集 CSV（--command test）: {test_path}")
        eval_data_path = test_path
        split_label = "test"
    else:
        eval_data_path = valid_path
        split_label = "valid"

    train_df = pd.read_csv(train_path)
    if command == "train":
        _require_step5_train_csv_columns(train_df)
    nuser = int(train_df["user_idx"].max()) + 1
    nitem = int(train_df["item_idx"].max()) + 1
    _model_dir = os.path.join(_ckpt_task, "model")
    os.makedirs(_model_dir, exist_ok=True)
    _best = os.path.join(_model_dir, "best_mainline.pth")
    _last = os.path.join(_model_dir, "last.pth")
    save_file = args.save_file or _best
    save_file = os.path.abspath(os.path.expanduser(save_file))
    nproc = int(resolved.num_proc)
    _raw_dp = (os.environ.get("D4C_DECODE_PROFILE_JSON") or "").strip()
    if not _raw_dp:
        raise RuntimeError(
            "缺少 D4C_DECODE_PROFILE_JSON：须由父进程 `python code/d4c.py …` 经 torchrun 注入完整 decode 预设 JSON；"
            "勿裸调 executors/step5_entry。"
        )
    try:
        _dp_full = json.loads(_raw_dp)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"D4C_DECODE_PROFILE_JSON 非法 JSON: {e}") from e
    if not isinstance(_dp_full, dict):
        raise TypeError("D4C_DECODE_PROFILE_JSON 根须为 object")
    _prof_patch = _decode_profile_to_training_patch(_dp_full)
    _ds = str(_dp_full.get("decode_strategy", "greedy")).strip().lower()
    _dseed = _dp_full.get("decode_seed")
    if _dseed in ("", "null", "None", None):
        _dseed_p = None
    else:
        _dseed_p = int(_dseed)
    _nr = _dp_full.get("no_repeat_ngram_size")
    _nr_p = None if _nr in (None, "", "null", "None") else int(_nr)
    _mn = _dp_full.get("min_len")
    _mn_p = None if _mn in (None, "", "null", "None") else int(_mn)
    ml_eff = int(_dp_full.get("max_explanation_length", int(args.max_explanation_length)))
    tlm = int(resolved.train_label_max_length)
    processor = Processor(args.auxiliary, args.target, max_length=tlm)
    base_final = replace(
        resolved,
        **_prof_patch,
        nuser=nuser,
        nitem=nitem,
        ntoken=len(tokenizer),
        save_file=save_file,
        last_checkpoint_path=_last,
        device=local_rank,
        device_ids=tuple(range(world_size)),
        ddp_world_size=world_size,
        nlayers=int(args.nlayers),
        nhead=int(args.nhead),
        nhid=int(args.nhid),
        dropout=float(args.dropout),
        label_smoothing=float(_dp_full["label_smoothing"]),
        repetition_penalty=float(_dp_full["repetition_penalty"]),
        generate_temperature=float(_dp_full["generate_temperature"]),
        generate_top_p=float(_dp_full["generate_top_p"]),
        max_explanation_length=ml_eff,
        decode_strategy=_ds,
        decode_seed=_dseed_p,
        no_repeat_ngram_size=_nr_p,
        min_len=_mn_p,
    )
    setattr(args, "_d4c_eval_split_label", split_label)
    setattr(args, "_d4c_eval_data_path", eval_data_path)

    if eval_only:
        ev_df = pd.read_csv(eval_data_path)
        ev_df["domain"] = "target"
        ev_df = ev_df.reset_index(drop=True)
        ev_df["sample_id"] = np.arange(len(ev_df), dtype=np.int64)
        ev_df["clean_text"] = ev_df["explanation"].fillna("").astype(str)
        ev_df["sample_weight_hint"] = 1.0
        cache_dir, cache_fp = _build_step5_cache_dir(
            get_hf_cache_root(task_idx),
            train_path,
            eval_data_path,
            processor,
            tokenizer,
            eval_only=True,
        )
        if rank == 0:
            _log_step5_tokenize_line(
                f"[Tokenize] step5 cache key | split={split_label} | fingerprint={cache_fp} | cache_dir={cache_dir}",
            )
        datasets = DatasetDict({"valid": Dataset.from_pandas(ev_df)})
        _st5 = logging.getLogger(LOGGER_NAME)
        _tok_wall0 = time.perf_counter()
        with d4c_timing_phase(
            _st5,
            f"tokenize_pipeline_step5_{split_label}",
            route=ROUTE_SUMMARY,
            rank=rank,
        ):
            encoded_data = _step5_map_or_load_tokenize_cache(
                datasets=datasets,
                processor=processor,
                nproc=nproc,
                cache_dir=cache_dir,
                cache_fingerprint=cache_fp,
                rank=rank,
                show_datasets_progress=show_datasets_progress,
                log_tokenize=(rank == 0),
                phase=split_label,
            )
        setattr(args, "_d4c_eval_tokenize_cache_wall_s", float(time.perf_counter() - _tok_wall0))
        train_dataset = None
    else:
        valid_df = pd.read_csv(valid_path)
        valid_df["domain"] = "target"
        valid_df = valid_df.reset_index(drop=True)
        valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
        valid_df["clean_text"] = valid_df["explanation"].fillna("").astype(str)
        valid_df["sample_weight_hint"] = 1.0

        train_raw = train_df.copy()
        train_df = train_df[train_df["train_keep"] == 1]
        train_df = train_df[train_df["clean_text"].fillna("").astype(str).str.strip() != ""]
        train_df = train_df.reset_index(drop=True)
        train_df["sample_id"] = np.arange(len(train_df), dtype=np.int64)
        if len(train_df) == 0:
            raise ValueError(
                "Step5 训练集在 train_keep==1 且 clean_text 非空过滤后行数为 0；请检查 Step4 导出或质量阈值。"
            )
        if rank == 0:
            _rank0_step5_train_data_audit(
                train_raw,
                train_df,
                train_label_max_length=tlm,
                train_dynamic_padding=bool(resolved.train_dynamic_padding),
                train_padding_strategy=str(resolved.train_padding_strategy),
                log_path=getattr(args, "log_file", None),
                ddp_find_unused_parameters_effective=bool(resolved.ddp_find_unused_parameters),
            )
        cache_dir, cache_fp = _build_step5_cache_dir(
            get_hf_cache_root(task_idx),
            train_path,
            valid_path,
            processor,
            tokenizer,
            eval_only=False,
        )
        if rank == 0:
            _log_step5_tokenize_line(
                f"[Tokenize] step5 cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
            )
        datasets = DatasetDict(
            {"train": Dataset.from_pandas(train_df), "valid": Dataset.from_pandas(valid_df)}
        )
        _st5 = logging.getLogger(LOGGER_NAME)
        _tok_wall0 = time.perf_counter()
        with d4c_timing_phase(
            _st5,
            "tokenize_pipeline_step5_train_valid",
            route=ROUTE_SUMMARY,
            rank=rank,
        ):
            encoded_data = _step5_map_or_load_tokenize_cache(
                datasets=datasets,
                processor=processor,
                nproc=nproc,
                cache_dir=cache_dir,
                cache_fingerprint=cache_fp,
                rank=rank,
                show_datasets_progress=show_datasets_progress,
                log_tokenize=(rank == 0),
                phase="train+valid",
            )
        setattr(args, "_d4c_eval_tokenize_cache_wall_s", float(time.perf_counter() - _tok_wall0))
        train_dataset = encoded_data["train"]
    valid_dataset = encoded_data["valid"]
    model = _make_model(base_final, args, local_rank)
    setattr(args, "_d4c_eval_dataset_build_wall_s", float(time.perf_counter() - _dataset_build_t0))
    return base_final, train_dataset, valid_dataset, model


def _metrics_final_dict_from_rows(merged: List[dict]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    all_pred = np.array([r["pred_rating"] for r in merged], dtype=np.float64)
    all_gt = np.array([r["gt_rating"] for r in merged], dtype=np.float64)
    diffs = all_pred - all_gt
    mae = round(float(np.mean(np.abs(diffs))), 4)
    rmse = round(float(np.sqrt(np.mean(np.square(diffs)))), 4)
    pred_tx = [r["pred_text"] for r in merged]
    ref_tx = [r["ref_text"] for r in merged]
    text_results = evaluate_text(pred_tx, ref_tx)
    paper_block = compute_paper_comparable_text_metrics(pred_tx, ref_tx)
    ext = extended_text_metrics_bundle(pred_tx, ref_tx)
    collapse = compute_collapse_stats(pred_tx, ref_tx, top_k_file=20)
    ref_mean = float((ext.get("corpus_level") or {}).get("mean_ref_len_words") or 0.0)
    dirty = compute_dirty_text_stats(pred_tx, ref_mean_len_words=ref_mean or None)
    final = {
        "metrics_schema_version": "d4c_metrics_mainline/2.0",
        "recommendation": {"mae": mae, "rmse": rmse},
        "explanation": text_results,
        "paper_metrics": paper_block,
        "text_metrics_corpus_and_sentence": ext,
        "collapse_stats": collapse,
        "dirty_text": dirty,
    }
    return final, pred_tx, ref_tx


def _run_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("step5 runner 仅支持 CUDA + NCCL DDP。")
    torch.cuda.set_device(local_rank)
    ddp_fast_backends = apply_ddp_fast_torch_backends()
    dist.init_process_group(backend="nccl")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    if rank == 0:
        log_path, run_id = create_run_paths(task_idx, args.log_file)
    else:
        log_path, run_id = None, None
    log_path, run_id = broadcast_run_paths_ddp(log_path, run_id, rank)
    args.log_file = log_path

    _setup = setup_train_logging(
        log_file=log_path,
        task_idx=task_idx,
        rank=rank,
        world_size=world_size,
        run_id=run_id,
    )
    train_logger = _setup["logger"]

    valid_dataloader = None
    eval_only = args.command != "train"
    try:
        base_final, train_dataset, valid_dataset, model = build_d4c_ddp_artefacts(
            args,
            world_size,
            local_rank,
            rank,
            command=args.command,
            show_datasets_progress=(rank == 0),
        )
        if args.command == "generate_samples":
            mxs = max(1, int(args.generate_max_samples))
            n_full = len(valid_dataset)
            mxs = min(mxs, n_full)
            valid_dataset = Subset(valid_dataset, list(range(mxs)))
        final_cfg = replace(
            base_final,
            run_id=run_id,
            logger=train_logger,
            log_file=log_path,
            valid_dataset=valid_dataset,
            ddp_fast_backends=ddp_fast_backends,
            rank0_only_logging=True,
        )
        step5_collate_fn = partial(
            _step5_collate_dynamic,
            dynamic_padding=bool(getattr(final_cfg, "train_dynamic_padding", True)),
            fixed_max_length=int(getattr(final_cfg, "train_label_max_length", 64)),
        )
        pin_memory = torch.cuda.is_available()
        _G = int(final_cfg.train_batch_size)
        if rank == 0:
            _decode_meta = {
                "command": args.command,
                "label_smoothing": final_cfg.label_smoothing,
                "repetition_penalty": final_cfg.repetition_penalty,
                "generate_temperature": final_cfg.generate_temperature,
                "generate_top_p": final_cfg.generate_top_p,
                "max_explanation_length": final_cfg.max_explanation_length,
                "train_label_max_length": int(getattr(final_cfg, "train_label_max_length", 128)),
                "decode_strategy": final_cfg.decode_strategy,
                "decode_seed": final_cfg.decode_seed,
                "no_repeat_ngram_size": final_cfg.no_repeat_ngram_size,
                "min_len": final_cfg.min_len,
                "ntoken_resolved": final_cfg.ntoken,
                "nhead": final_cfg.nhead,
                "nhid": final_cfg.nhid,
                "nlayers": final_cfg.nlayers,
                "dropout": final_cfg.dropout,
                "eval_single_process_safe": bool(getattr(args, "eval_single_process_safe", False)),
                "sanity_compare_ddp_single": bool(getattr(args, "sanity_compare_ddp_single", False)),
                "train_dynamic_padding": bool(getattr(final_cfg, "train_dynamic_padding", True)),
                "train_padding_strategy": str(getattr(final_cfg, "train_padding_strategy", "dynamic_batch")),
            }
            _trd_snap = training_diagnostics_snapshot(
                diagnostics_scope="child",
                effective_training_payload_json=os.environ.get("D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""),
                ddp_find_unused_parameters_effective=bool(final_cfg.ddp_find_unused_parameters),
            )
            _tfp = (os.environ.get("D4C_TRAINING_SEMANTIC_FINGERPRINT") or "").strip()
            _gfp = (os.environ.get("D4C_GENERATION_SEMANTIC_FINGERPRINT") or "").strip()
            _rdfp = (os.environ.get("D4C_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip()
            log_run_snapshot(
                train_logger,
                {
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                    "command": args.command,
                    "train_only": getattr(args, "train_only", False),
                    "eval_only": eval_only,
                    "rank": rank,
                    "world_size": world_size,
                    "local_rank": local_rank,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "train_global_batch_size": _G,
                    "train_batch_size_global": final_cfg.batch_size_global,
                    "train_per_device_batch_size": final_cfg.per_device_train_batch_size,
                    "eval_global_batch_size": int(final_cfg.eval_batch_size),
                    "eval_per_gpu_batch_size": (
                        int(final_cfg.eval_batch_size) // world_size
                        if int(final_cfg.eval_batch_size) % world_size == 0
                        else None
                    ),
                    "gradient_accumulation_steps": final_cfg.gradient_accumulation_steps,
                    "effective_global_batch_size": final_cfg.effective_global_batch_size,
                    "distributed_env": collect_distributed_env_for_meta(),
                    "decode_and_model_runtime": _decode_meta,
                    "training_diagnostics": _trd_snap,
                    "training_semantic_fingerprint": _tfp or None,
                    "generation_semantic_fingerprint": _gfp or None,
                    "runtime_diagnostics_fingerprint": _rdfp or None,
                },
                final_cfg.to_log_dict(),
            )
            _run_root = os.path.abspath(os.path.join(os.path.dirname(log_path), ".."))
            _cfg_resolved_path = os.path.join(_run_root, "config_resolved.json")
            _cfg_merged = dict(final_cfg.to_log_dict())
            _ignored = [k for k in _STEP5_IGNORED_FIELDS if k in _cfg_merged]
            for _k in _ignored:
                _cfg_merged.pop(_k, None)
            _cfg_merged["step5_ignored_fields"] = list(_ignored)
            _cfg_merged["training_diagnostics"] = _trd_snap
            _cfg_merged["training_semantic_fingerprint"] = _tfp or None
            _cfg_merged["generation_semantic_fingerprint"] = _gfp or None
            _cfg_merged["runtime_diagnostics_fingerprint"] = _rdfp or None

            _cfg_merged["runtime_env"] = runtime_env_dict_for_config_resolved()

            with open(_cfg_resolved_path, "w", encoding="utf-8") as _cf:
                json.dump(_cfg_merged, _cf, ensure_ascii=False, indent=2, default=str)
                _cf.write("\n")
            train_logger.info(
                "[Config resolved] wrote %s",
                _cfg_resolved_path,
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Config resolved] %s",
                json.dumps(_decode_meta, ensure_ascii=False, default=str),
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Step5 ignored fields] %s",
                ",".join(_ignored) if _ignored else "(none)",
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Fingerprints] training_semantic=%s generation_semantic=%s runtime_diag=%s",
                _tfp or "n/a",
                _gfp or "n/a",
                _rdfp or "n/a",
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Step5 train knobs] train_label_max_length=%s train_dynamic_padding=%s "
                "train_padding_strategy=%s loss_weight_repeat_ul=%s loss_weight_terminal_clean=%s",
                int(getattr(final_cfg, "train_label_max_length", 128)),
                bool(getattr(final_cfg, "train_dynamic_padding", True)),
                str(getattr(final_cfg, "train_padding_strategy", "dynamic_batch")),
                float(getattr(final_cfg, "loss_weight_repeat_ul", 0.0)),
                float(getattr(final_cfg, "loss_weight_terminal_clean", 0.0)),
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            flush_preset_load_events(train_logger)
        valid_sampler = None
        if not eval_only:
            if int(final_cfg.eval_batch_size) % world_size != 0:
                raise ValueError(
                    f"eval_batch_size={int(final_cfg.eval_batch_size)} 与 world_size={world_size} 不整除，无法按卡切分。"
                    "请修改 presets/eval_profiles/*.yaml 的 eval_batch_size，或调整 presets/hardware/*.yaml 的 ddp_world_size。"
                )
            valid_per_rank = int(final_cfg.eval_batch_size) // world_size
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=valid_per_rank,
                sampler=valid_sampler,
                shuffle=False,
                num_workers=final_cfg.dataloader_num_workers_valid,
                pin_memory=pin_memory,
                persistent_workers=final_cfg.dataloader_num_workers_valid > 0,
                prefetch_factor=final_cfg.dataloader_prefetch_factor_valid,
                collate_fn=step5_collate_fn,
            )
        if not eval_only:
            _A = max(1, int(final_cfg.gradient_accumulation_steps))
            train_drop_last = _A > 1
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=train_drop_last,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=final_cfg.per_device_train_batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=final_cfg.dataloader_num_workers_train,
                pin_memory=pin_memory,
                persistent_workers=final_cfg.dataloader_num_workers_train > 0,
                prefetch_factor=final_cfg.dataloader_prefetch_factor_train,
                drop_last=train_drop_last,
                collate_fn=step5_collate_fn,
            )
            _n_train_micro = len(train_dataloader)
            if _A > 1 and _n_train_micro % _A != 0:
                raise ValueError(
                    f"train DataLoader 每 epoch 批次数为 {_n_train_micro}，无法被 gradient_accumulation_steps={_A} 整除。"
                    f"请调整全局 batch、--per-device-batch-size、world_size 或数据划分；或令 accum=1。"
                )
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=final_cfg.ddp_find_unused_parameters,
            )
            try:
                trainModel_ddp(
                    model,
                    train_dataloader,
                    valid_dataloader,
                    sampler,
                    valid_sampler,
                    final_cfg,
                    rank,
                    world_size,
                    step5_collate_fn=step5_collate_fn,
                )
            except Exception as exc:
                if rank == 0:
                    log_training_crash(train_logger, exc)
                raise
        if eval_only and not os.path.isfile(final_cfg.save_file):
            raise FileNotFoundError(
                f"eval/test/generate_samples 需要已有权重文件，未找到: {final_cfg.save_file}\n"
                "请确认 D4C_STAGE_RUN_DIR 指向含 model/best_mainline.pth（或显式 --save_file）的训练 run。"
            )
        dist.barrier()
        run_final_eval = eval_only or (args.command == "train" and not getattr(args, "train_only", False))
        if run_final_eval:
            import time as _time

            _eval_t0 = _time.perf_counter()
            _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _split_lab = getattr(args, "_d4c_eval_split_label", "valid")
            _real_n = len(valid_dataset)
            _eval_global_bs = int(final_cfg.eval_batch_size)
            _eval_nw = int(final_cfg.dataloader_num_workers_valid)
            _eval_pf = final_cfg.dataloader_prefetch_factor_valid
            _review_rows: List[str] = []
            _review_meta: Dict[str, Any] = {"fast_path": "not_used", "review_rows_count": 0}
            _review_t0 = time.perf_counter()
            if str(args.command) == "eval-rerank":
                if rank == 0:
                    _review_rows, _review_meta = _load_review_by_sample_id(
                        str(getattr(args, "_d4c_eval_data_path", "") or "")
                    )
                gathered_review: List[Any] = [_review_rows, _review_meta]
                dist.broadcast_object_list(gathered_review, src=0)
                _review_rows = list(gathered_review[0] or [])
                _review_meta = dict(gathered_review[1] or {})
            _review_load_time = float(time.perf_counter() - _review_t0)
            single_safe = bool(getattr(args, "eval_single_process_safe", False)) and world_size > 1
            sanity_cmp = bool(getattr(args, "sanity_compare_ddp_single", False)) and world_size > 1
            if (not single_safe) and world_size > 1 and (_eval_global_bs % world_size != 0):
                raise ValueError(
                    f"eval_batch_size={_eval_global_bs} 与 world_size={world_size} 不整除，DDP 评测非法。"
                    "请修改 presets/eval_profiles/*.yaml 的 eval_batch_size，或调整 hardware preset 的 ddp_world_size。"
                )
            _embedded_eval_log_fh: Optional[logging.Handler] = None
            if rank == 0:
                _emb_lp = (os.environ.get("D4C_STEP5_EMBEDDED_EVAL_LOG") or "").strip()
                if _emb_lp:
                    _eld = os.path.dirname(os.path.abspath(_emb_lp))
                    if _eld:
                        os.makedirs(_eld, exist_ok=True)
                    _embedded_eval_log_fh = logging.FileHandler(_emb_lp, mode="w", encoding="utf-8")
                    _embedded_eval_log_fh.setLevel(logging.DEBUG)
                    _embedded_eval_log_fh.setFormatter(
                        logging.Formatter(
                            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                        )
                    )
                    train_logger.addHandler(_embedded_eval_log_fh)
                    train_logger.info(
                        "[eval.log] 训练后 valid 评估日志写入 %s",
                        _emb_lp,
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )

            def _eval_export_dir() -> str:
                _ed = (os.environ.get("D4C_EVAL_RUN_DIR") or "").strip()
                if not _ed:
                    raise RuntimeError(
                        "D4C_EVAL_RUN_DIR 未设置。请使用: python code/d4c.py eval|eval-rerank …"
                    )
                _ed = os.path.abspath(_ed)
                os.makedirs(_ed, exist_ok=True)
                return _ed

            def _rank0_write_eval_artifacts(
                merged: List[dict],
                final: Dict[str, Any],
                *,
                pipeline_tag: str,
                eval_model: Optional[torch.nn.Module] = None,
                eval_perf: Optional[Dict[str, Any]] = None,
            ) -> None:
                ed = _eval_export_dir()
                _eval_tag = eval_decode_tag(
                    decode_strategy=str(final_cfg.decode_strategy),
                    generate_temperature=float(final_cfg.generate_temperature),
                    generate_top_p=float(final_cfg.generate_top_p),
                )
                _is_rerank = str(args.command) == "eval-rerank"
                eval_sub = ed
                os.makedirs(eval_sub, exist_ok=True)
                _ebn = getattr(final_cfg, "eval_profile_name", None)
                if _ebn:
                    train_logger.info(
                        "[eval_profile_orchestrator] name=%s hardware_preset=%s decode_preset_stem=%s "
                        "rerank_preset_stem=%s global_eval_batch_size=%s ddp_world_size=%s",
                        _ebn,
                        (os.environ.get("D4C_HARDWARE_PRESET") or "").strip(),
                        (os.environ.get("D4C_DECODE_PRESET_STEM") or "").strip(),
                        (os.environ.get("D4C_RERANK_PRESET_STEM") or "").strip() or "-",
                        int(final_cfg.eval_batch_size),
                        int(getattr(final_cfg, "ddp_world_size", 1) or 1),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                _decode_cfg = {
                    "decode_strategy": final_cfg.decode_strategy,
                    "decode_seed": final_cfg.decode_seed,
                    "repetition_penalty": final_cfg.repetition_penalty,
                    "generate_temperature": final_cfg.generate_temperature,
                    "generate_top_p": final_cfg.generate_top_p,
                    "max_explanation_length": final_cfg.max_explanation_length,
                    "label_smoothing": final_cfg.label_smoothing,
                    "no_repeat_ngram_size": final_cfg.no_repeat_ngram_size,
                    "min_len": final_cfg.min_len,
                    "soft_max_len": getattr(final_cfg, "soft_max_len", None),
                    "hard_max_len": getattr(final_cfg, "hard_max_len", None),
                    "eos_boost_start": getattr(final_cfg, "eos_boost_start", 9999),
                    "eos_boost_value": getattr(final_cfg, "eos_boost_value", 0.0),
                    "tail_temperature": getattr(final_cfg, "tail_temperature", -1.0),
                    "tail_top_p": getattr(final_cfg, "tail_top_p", -1.0),
                    "forbid_eos_after_open_quote": getattr(final_cfg, "forbid_eos_after_open_quote", True),
                    "forbid_eos_after_open_bracket": getattr(final_cfg, "forbid_eos_after_open_bracket", True),
                    "forbid_bad_terminal_tokens": getattr(final_cfg, "forbid_bad_terminal_tokens", True),
                    "decode_token_repeat_window": getattr(final_cfg, "decode_token_repeat_window", 4),
                    "decode_token_repeat_max": getattr(final_cfg, "decode_token_repeat_max", 2),
                    "candidate_family": getattr(final_cfg, "candidate_family", "balanced"),
                    "loss_weight_repeat_ul": getattr(final_cfg, "loss_weight_repeat_ul", 0.0),
                    "loss_weight_terminal_clean": getattr(final_cfg, "loss_weight_terminal_clean", 0.0),
                    "terminal_clean_span": getattr(final_cfg, "terminal_clean_span", 3),
                }
                generation_semantic_resolved, _ = build_generation_semantic_resolved_and_fingerprint(
                    _decode_cfg
                )
                if _is_rerank:
                    _cfp_raw = json.dumps(_decode_cfg, ensure_ascii=False, sort_keys=True, default=str)
                    train_logger.info(
                        "[CheckpointSemantics] eval_rerank candidate_decode_fingerprint_sha1=%s "
                        "decode_strategy=%s num_return_sequences=%s",
                        hashlib.sha1(_cfp_raw.encode("utf-8")).hexdigest()[:16],
                        str(final_cfg.decode_strategy),
                        int(getattr(args, "num_return_sequences", 4) or 4),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                collapse_stats = final.get("collapse_stats") or {}
                metrics_payload = {
                    "metrics_schema_version": "4.0",
                    "training_semantic_fingerprint": (
                        (os.environ.get("D4C_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or None
                    ),
                    "generation_semantic_fingerprint": (
                        (os.environ.get("D4C_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or None
                    ),
                    "generation_semantic_resolved": generation_semantic_resolved,
                    "checkpoint": os.path.abspath(str(final_cfg.save_file)),
                    "eval_run_dir": os.path.abspath(eval_sub),
                    "metrics_json_path": os.path.join(os.path.abspath(eval_sub), "metrics.json"),
                    "task_idx": int(final_cfg.task_idx),
                    "split": _split_lab,
                    "split_csv": getattr(args, "_d4c_eval_data_path", ""),
                    "review_path": getattr(args, "_d4c_eval_data_path", ""),
                    "review_rows_count": int(_review_meta.get("review_rows_count", 0) or 0),
                    "review_load_fast_path": str(_review_meta.get("fast_path", "not_used")),
                    "seed": int(args.seed),
                    "world_size": world_size,
                    "eval_mode": "single_process_safe" if single_safe else "ddp_sharded",
                    "command": args.command,
                    "eval_profile_name": _ebn,
                    "eval_export_tag": _eval_tag,
                    "decode": _decode_cfg,
                    "collapse_stats": collapse_stats,
                    "paper_metrics": final.get("paper_metrics"),
                    "metrics": final,
                }
                try:
                    _ckp = os.path.abspath(str(final_cfg.save_file))
                    _parent = os.path.dirname(_ckp)
                    if os.path.basename(_parent) == "model":
                        _run_root = os.path.dirname(_parent)
                        metrics_payload["step5_checkpoint_run_dir"] = _run_root
                        metrics_payload["step5_run_id"] = os.path.basename(_run_root)
                except Exception:
                    pass
                merged_for_pred: List[dict] = merged
                if _is_rerank:
                    _rrm = _rerank_eval_cli_resolved(args)
                    _ex_mode = str(_rrm["export_examples_mode"]).strip().lower()
                    _rm = str(_rrm["rerank_method"])
                    rs = _aggregate_rerank_summary(
                        merged,
                        export_examples_mode=_ex_mode,
                        rerank_method=_rm,
                    )
                    rw = build_rerank_weights_dict(
                        weight_logprob=float(_rrm["rerank_weight_logprob"]),
                        weight_length=float(_rrm["rerank_weight_length"]),
                        weight_repeat=float(_rrm["rerank_weight_repeat"]),
                        weight_dirty=float(_rrm["rerank_weight_dirty"]),
                    )
                    _mtail_c = float(_rrm["rerank_malformed_tail_penalty"])
                    _mtok_c = float(_rrm["rerank_malformed_token_penalty"])
                    rs["rerank_weights"] = rw
                    rs["rerank_malformed_tail_coef"] = _mtail_c
                    rs["rerank_malformed_token_coef"] = _mtok_c
                    metrics_payload["rerank_enabled"] = True
                    metrics_payload["rerank_method"] = _rm
                    metrics_payload["num_return_sequences"] = int(_rrm["num_return_sequences"])
                    metrics_payload["rerank_top_k"] = int(_rrm["rerank_top_k"])
                    metrics_payload["rerank_weights"] = rw
                    metrics_payload["rerank_target_len_ratio"] = float(_rrm["rerank_target_len_ratio"])
                    metrics_payload["rerank_malformed_tail_penalty"] = _mtail_c
                    metrics_payload["rerank_malformed_token_penalty"] = _mtok_c
                    metrics_payload["export_examples_mode"] = _ex_mode
                    metrics_payload["export_full_rerank_examples"] = bool(
                        getattr(args, "export_full_rerank_examples", False)
                    )
                    metrics_payload["rerank_logprob_source"] = (
                        "per_token_log_softmax_at_chosen_id; same logits/preprocessing as generate "
                        "(nucleus: tempered softmax; greedy: argmax on same logits)"
                    )
                    metrics_payload["rerank_summary"] = rs
                    if "v3" in _rm.replace("_", ""):
                        metrics_payload["rerank_v3_summary"] = dict(rs)
                        _rrpj = (os.environ.get("D4C_RERANK_PROFILE_JSON") or "").strip()
                        if _rrpj:
                            try:
                                metrics_payload["rerank_v3_profile_effective"] = json.loads(_rrpj)
                            except Exception:
                                metrics_payload["rerank_v3_profile_effective"] = {}
                    merged_for_pred = [{k: v for k, v in r.items() if k != "_rerank"} for r in merged]
                    _export_full = bool(
                        getattr(args, "export_full_rerank_examples", False)
                    ) or (_ex_mode == "full")
                    _rr0 = time.perf_counter()
                    _write_rerank_artifacts(
                        eval_sub,
                        merged,
                        rerank_cfg={
                            "rerank_method": metrics_payload["rerank_method"],
                            "num_return_sequences": metrics_payload["num_return_sequences"],
                            "rerank_top_k": metrics_payload["rerank_top_k"],
                            "rerank_weights": rw,
                            "rerank_target_len_ratio": metrics_payload["rerank_target_len_ratio"],
                            "export_examples_mode": _ex_mode,
                            "export_full_rerank_examples": _export_full,
                            "rerank_malformed_tail_penalty": _mtail_c,
                            "rerank_malformed_token_penalty": _mtok_c,
                        },
                        rerank_summary=rs,
                        export_examples_mode=_ex_mode,
                        export_full_rerank_examples=_export_full,
                    )
                    if eval_perf is not None:
                        eval_perf["rerank_artifacts_write_time"] = float(time.perf_counter() - _rr0)
                if eval_model is not None:
                    _um = get_underlying_model(eval_model)
                    metrics_payload["generate_kwargs_effective"] = _um.get_generate_kwargs_effective()
                    metrics_payload["generate_kwargs_effective_v2"] = _um.get_generate_kwargs_effective_v2()
                _ckpt_abs = os.path.abspath(str(final_cfg.save_file))
                _eval_meta = {
                    "checkpoint": _ckpt_abs,
                    "eval_export_tag": _eval_tag,
                    "eval_run_dir": os.path.abspath(eval_sub),
                    "decode": _decode_cfg,
                    "recommendation": final.get("recommendation"),
                    "bleu4": final.get("explanation", {}).get("bleu", {}).get("4"),
                    "meteor": final.get("explanation", {}).get("meteor"),
                    "rouge_l": final.get("explanation", {}).get("rouge", {}).get("l"),
                    "collapse_top1_ratio": collapse_stats.get("top1_pred_ratio"),
                    "collapse_unique_ratio": collapse_stats.get("pred_unique_ratio"),
                    "collapse_stats": collapse_stats,
                }
                try:
                    with open(
                        os.path.join(eval_sub, "eval_checkpoint_sidecar.json"),
                        "w",
                        encoding="utf-8",
                    ) as _emf:
                        json.dump(_eval_meta, _emf, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pass
                csv_fields = ["sample_id", "pred_rating", "gt_rating", "pred_text", "ref_text"]
                if _is_rerank:
                    csv_fields.extend(["candidate_family", "lp_norm", "completion_ok"])
                _pw0 = time.perf_counter()
                write_predictions_csv(
                    os.path.join(eval_sub, "predictions.csv"), merged_for_pred, csv_fields
                )
                write_predictions_jsonl(os.path.join(eval_sub, "predictions.jsonl"), merged_for_pred)
                if eval_perf is not None:
                    eval_perf["predictions_write_time"] = float(time.perf_counter() - _pw0)
                _perf = dict(eval_perf) if eval_perf is not None else {}
                _perf["total_eval_time"] = float(time.perf_counter() - _eval_t0)
                _summary_keys = (
                    "review_load_time",
                    "tokenize_cache_time",
                    "eval_dataset_build_time",
                    "eval_dataloader_build_time",
                    "decode_time",
                    "rerank_feature_time",
                    "gather_time",
                    "metrics_time",
                    "predictions_write_time",
                    "rerank_scoring_time",
                    "rerank_artifacts_write_time",
                    "total_eval_time",
                )
                _rt_eval_snap = runtime_env_dict_for_config_resolved()
                metrics_payload["eval_performance"] = {
                    "global_eval_batch_size": int(_eval_global_bs),
                    "eval_per_gpu_batch_size": int(
                        min(_eval_global_bs, max(1, _real_n))
                        if single_safe
                        else (_eval_global_bs // world_size)
                    ),
                    "dataloader_num_workers_valid": int(_eval_nw),
                    "dataloader_prefetch_factor_valid": _eval_pf,
                    "hardware_preset": (os.environ.get("D4C_HARDWARE_PRESET") or "").strip() or None,
                    "runtime_env": _rt_eval_snap,
                    "summary": {k: float(_perf[k]) for k in _summary_keys if k in _perf},
                    "detail": _perf,
                }
                with open(os.path.join(eval_sub, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, ensure_ascii=False, indent=2, default=str)
                _cfg_eval = dict(final_cfg.to_log_dict())
                _ignored_eval = [k for k in _STEP5_IGNORED_FIELDS if k in _cfg_eval]
                for _k in _ignored_eval:
                    _cfg_eval.pop(_k, None)
                _cfg_eval["step5_ignored_fields"] = list(_ignored_eval)
                _cfg_eval["training_diagnostics"] = training_diagnostics_snapshot(
                    diagnostics_scope="child",
                    effective_training_payload_json=os.environ.get(
                        "D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""
                    ),
                    ddp_find_unused_parameters_effective=bool(final_cfg.ddp_find_unused_parameters),
                )
                _cfg_eval["training_semantic_fingerprint"] = (
                    (os.environ.get("D4C_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or None
                )
                _cfg_eval["generation_semantic_fingerprint"] = (
                    (os.environ.get("D4C_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or None
                )
                _cfg_eval["runtime_diagnostics_fingerprint"] = (
                    (os.environ.get("D4C_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip() or None
                )

                _cfg_eval["runtime_env"] = _rt_eval_snap

                with open(os.path.join(eval_sub, "config_resolved.json"), "w", encoding="utf-8") as f:
                    json.dump(_cfg_eval, f, ensure_ascii=False, indent=2, default=str)
                train_logger.info(
                    "[eval_performance] %s",
                    json.dumps(
                        metrics_payload.get("eval_performance") or {},
                        ensure_ascii=False,
                        default=str,
                    ),
                    extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                )
                log_sample_id_alignment_snippet(merged_for_pred, k=20, logger=train_logger)
                if _is_rerank:
                    _rs = metrics_payload.get("rerank_summary") or {}
                    train_logger.info(
                        "[rerank_effective] K=%s method=%s selected_not_best_logprob_rate=%.6g mean_selected_rerank_score=%s",
                        metrics_payload.get("num_return_sequences"),
                        metrics_payload.get("rerank_method"),
                        float(_rs.get("selected_not_best_logprob_rate", float("nan"))),
                        _rs.get("mean_selected_rerank_score"),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                    train_logger.info(
                        "[rerank_summary] %s",
                        json.dumps(_rs, ensure_ascii=False, default=str),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                _cw = collapse_stats.get("collapse_warnings") or []
                if _cw:
                    train_logger.warning(
                        "[Collapse warning] %s",
                        "; ".join(str(x) for x in _cw),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                _task_desc = (
                    f"Step 5 Task {task_idx} {args.command} (nproc={world_size}, split={_split_lab}): "
                    f"{args.auxiliary} -> {args.target} | eval_tag={_eval_tag}"
                )
                _eval_elapsed = time.perf_counter() - _eval_t0
                _eval_min, _eval_sec = divmod(int(_eval_elapsed), 60)
                _lines = format_final_results_lines(
                    final,
                    task_description=_task_desc,
                    start_time=_eval_start_str,
                    decode_cfg=_decode_cfg,
                    collapse_stats=collapse_stats,
                    eval_run_tag=_eval_tag,
                )
                _lines.append(f"Eval elapsed: {_eval_min}m {_eval_sec}s ({_eval_elapsed:.1f}s)")
                _lines.append(f"Eval artefacts: {eval_sub}")
                log_final_results_block(train_logger, _lines)
                finalize_run_log(train_logger)
                try:
                    flush_d4c_file_handlers(train_logger)
                    _digest_p = write_eval_digest_log(
                        eval_subdir=eval_sub,
                        metrics_final=final,
                        merged_rows=merged_for_pred,
                        final_cfg=final_cfg,
                        decode_cfg=dict(_decode_cfg),
                        active_log_file=(log_path or "").strip() or None,
                        task_idx=int(final_cfg.task_idx),
                        auxiliary=str(args.auxiliary),
                        target=str(args.target),
                        eval_export_tag=_eval_tag,
                        command=str(args.command),
                        eval_timing_summary=dict(
                            (metrics_payload.get("eval_performance") or {}).get("summary") or {}
                        ),
                    )
                    train_logger.info(
                        "[eval_digest] wrote %s",
                        _digest_p,
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                except Exception as _digest_exc:
                    train_logger.warning(
                        "[eval_digest] 生成 eval_digest.log 失败: %s",
                        _digest_exc,
                        exc_info=True,
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                train_logger.info("DONE.")

            if single_safe:
                dist.barrier()
                if rank == 0:
                    eval_model = _make_model(final_cfg, args, local_rank)
                    eval_model.load_state_dict(
                        torch.load(
                            final_cfg.save_file,
                            map_location=f"cuda:{local_rank}",
                            weights_only=True,
                        ),
                    )
                    train_logger.info(
                        "[generate_kwargs_effective] %s",
                        json.dumps(
                            get_underlying_model(eval_model).get_generate_kwargs_effective(),
                            ensure_ascii=False,
                            default=str,
                        ),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                    _bs = min(_eval_global_bs, max(1, _real_n))
                    _dl_c0 = time.perf_counter()
                    eval_dataloader = DataLoader(
                        valid_dataset,
                        batch_size=_bs,
                        shuffle=False,
                        num_workers=min(_eval_nw, 2),
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=min(_eval_nw, 2) > 0,
                        prefetch_factor=_eval_pf if min(_eval_nw, 2) > 0 else None,
                        collate_fn=step5_collate_fn,
                    )
                    _eval_perf: Dict[str, Any] = {
                        "review_load_time": _review_load_time,
                        "tokenize_cache_time": float(
                            getattr(args, "_d4c_eval_tokenize_cache_wall_s", 0.0) or 0.0
                        ),
                        "eval_dataset_build_time": float(
                            getattr(args, "_d4c_eval_dataset_build_wall_s", 0.0) or 0.0
                        ),
                        "eval_dataloader_build_time": float(time.perf_counter() - _dl_c0),
                    }
                    rows_local, _t_loc = _eval_rows_local(
                        eval_model, eval_dataloader, local_rank, args, review_rows=_review_rows
                    )
                    _eval_perf["decode_time"] = float(_t_loc.get("decode_time", 0.0))
                    _eval_perf["rerank_feature_time"] = float(_t_loc.get("rerank_feature_time", 0.0))
                    _eval_perf["rerank_scoring_time"] = float(_t_loc.get("rerank_scoring_time", 0.0))
                    _eval_perf["gather_time"] = 0.0
                    _mt0 = time.perf_counter()
                    merged = merge_eval_rows_by_sample_id([rows_local], _real_n)
                    final, _, _ = _metrics_final_dict_from_rows(merged)
                    _eval_perf["metrics_time"] = float(time.perf_counter() - _mt0)
                    _rank0_write_eval_artifacts(
                        merged,
                        final,
                        pipeline_tag=f"run_d4c_{args.command}_single_safe",
                        eval_model=eval_model,
                        eval_perf=_eval_perf,
                    )
                dist.barrier()
                if rank == 0 and sanity_cmp:
                    train_logger.info(
                        "[Eval sanity] 已使用 --eval-single-process-safe，跳过 DDP/单路二次对比（请分别跑 DDP 与 safe 两次对比指标）。",
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
            else:
                eval_model = _make_model(final_cfg, args, local_rank)
                eval_model.load_state_dict(
                    torch.load(
                        final_cfg.save_file,
                        map_location=f"cuda:{local_rank}",
                        weights_only=True,
                    ),
                )
                if rank == 0:
                    train_logger.info(
                        "[generate_kwargs_effective] %s",
                        json.dumps(
                            get_underlying_model(eval_model).get_generate_kwargs_effective(),
                            ensure_ascii=False,
                            default=str,
                        ),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                if _eval_global_bs % world_size != 0:
                    raise ValueError(
                        f"eval_batch_size={_eval_global_bs} 与 world_size={world_size} 不整除，DDP 评测非法。"
                        "请修改 presets/eval_profiles/*.yaml 的 eval_batch_size，或调整 hardware preset 的 ddp_world_size。"
                    )
                _eval_per_gpu = _eval_global_bs // world_size
                eval_sampler = DistributedSampler(
                    valid_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                _dl_c1 = time.perf_counter()
                eval_dataloader = DataLoader(
                    valid_dataset,
                    batch_size=_eval_per_gpu,
                    sampler=eval_sampler,
                    shuffle=False,
                    num_workers=_eval_nw,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=_eval_nw > 0,
                    prefetch_factor=_eval_pf,
                    collate_fn=step5_collate_fn,
                )
                _eval_perf_ddp: Dict[str, Any] = {}
                if rank == 0:
                    _eval_perf_ddp["review_load_time"] = _review_load_time
                    _eval_perf_ddp["tokenize_cache_time"] = float(
                        getattr(args, "_d4c_eval_tokenize_cache_wall_s", 0.0) or 0.0
                    )
                    _eval_perf_ddp["eval_dataset_build_time"] = float(
                        getattr(args, "_d4c_eval_dataset_build_wall_s", 0.0) or 0.0
                    )
                    _eval_perf_ddp["eval_dataloader_build_time"] = float(time.perf_counter() - _dl_c1)
                rows_local, _t_loc_ddp = _eval_rows_local(
                    eval_model, eval_dataloader, local_rank, args, review_rows=_review_rows
                )
                gathered_rows: List[Any] = [None] * world_size
                if rank == 0:
                    _ga0 = time.perf_counter()
                dist.all_gather_object(gathered_rows, rows_local)
                if rank == 0:
                    _eval_perf_ddp["gather_time"] = float(time.perf_counter() - _ga0)
                    _eval_perf_ddp["decode_time"] = float(_t_loc_ddp.get("decode_time", 0.0))
                    _eval_perf_ddp["rerank_feature_time"] = float(
                        _t_loc_ddp.get("rerank_feature_time", 0.0)
                    )
                    _eval_perf_ddp["rerank_scoring_time"] = float(_t_loc_ddp.get("rerank_scoring_time", 0.0))
                    _mm0 = time.perf_counter()
                    merged = merge_eval_rows_by_sample_id(gathered_rows, _real_n)
                    final, _, _ = _metrics_final_dict_from_rows(merged)
                    _eval_perf_ddp["metrics_time"] = float(time.perf_counter() - _mm0)
                    pl = (
                        f"run_d4c_{args.command}_eval"
                        if eval_only
                        else "run_d4c_train_eval"
                    )
                    _rank0_write_eval_artifacts(
                        merged,
                        final,
                        pipeline_tag=pl,
                        eval_model=eval_model,
                        eval_perf=_eval_perf_ddp,
                    )
                    if sanity_cmp:
                        eval_model_s = _make_model(final_cfg, args, local_rank)
                        eval_model_s.load_state_dict(
                            torch.load(
                                final_cfg.save_file,
                                map_location=f"cuda:{local_rank}",
                                weights_only=True,
                            ),
                        )
                        _bs2 = min(_eval_global_bs, max(1, _real_n))
                        dl_s = DataLoader(
                            valid_dataset,
                            batch_size=_bs2,
                            shuffle=False,
                            num_workers=min(_eval_nw, 2),
                            pin_memory=torch.cuda.is_available(),
                            collate_fn=step5_collate_fn,
                        )
                        rows_s = evalModel(eval_model_s, dl_s, local_rank)["rows"]
                        merged_s = merge_eval_rows_by_sample_id([rows_s], _real_n)
                        final_s, _, _ = _metrics_final_dict_from_rows(merged_s)
                        d_mae = abs(float(final["recommendation"]["mae"]) - float(final_s["recommendation"]["mae"]))
                        d_rmse = abs(float(final["recommendation"]["rmse"]) - float(final_s["recommendation"]["rmse"]))
                        d_bleu = abs(float(final["explanation"]["bleu"]["4"]) - float(final_s["explanation"]["bleu"]["4"]))
                        train_logger.info(
                            "[Eval sanity] DDP vs rank0-sequential | d_mae=%.6g d_rmse=%.6g d_bleu4=%.6g",
                            d_mae,
                            d_rmse,
                            d_bleu,
                            extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                        )
            if _embedded_eval_log_fh is not None:
                train_logger.removeHandler(_embedded_eval_log_fh)
                _embedded_eval_log_fh.close()
        elif rank == 0:
            _sf = final_cfg.save_file
            train_logger.info("DONE（train --train-only：已跳过训练后评估；权重: %s）。", _sf)
            finalize_run_log(train_logger)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


