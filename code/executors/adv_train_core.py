# -*- coding: utf-8 -*-
"""
INTERNAL EXECUTOR — Step3 域对抗模型与训练循环（非用户首选入口）。

- MAINLINE ENTRY：``python code/d4c.py step3|eval|…``（仓库根目录）。
- 本文件由 ``d4c.py`` 经 torchrun 分发给 **step3 runner** 调用；勿作为日常手工入口。
- Step4 引擎 ``executors.step4_engine`` 自本模块 ``import *`` 复用符号。
"""
import os
import sys
import copy
import json
import time
import hashlib
import warnings
import argparse
import contextlib
import logging
import math
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")
_ROOT = os.path.dirname(os.path.abspath(__file__))
# 在 import base_utils（会 import nltk）之前指定离线 NLTK 语料，避免 METEOR 触发 nltk.download
_nltk_data = os.path.join(os.path.dirname(_ROOT), "pretrained_models", "nltk_data")
if os.path.isdir(_nltk_data):
    os.environ.setdefault("NLTK_DATA", os.path.abspath(_nltk_data))
sys.path.insert(0, _ROOT)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*no current CUDA context.*", category=UserWarning)

from base_utils import *
from paths_config import get_data_dir, get_hf_cache_root, get_merged_data_dir, get_stage_run_dir, get_t5_small_dir
from d4c_core.runtime_env_pack import runtime_env_dict_for_config_resolved
import torch

# transformers 在 modeling_utils.load_state_dict 里用 torch.load(..., map_location=...) 未传 weights_only，
# PyTorch 2.4+ 会 FutureWarning。在 from_pretrained 前默认 weights_only=True，与官方推荐一致。
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

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch import optim
from torch.optim import lr_scheduler as lr_sched
import torch.distributed as dist
from transformers import T5Tokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from perf_monitor import PerfMonitor, gather_ddp_gpu_stats_for_epoch_log
from datasets import Dataset, DatasetDict, load_from_disk
from config import (
    FinalTrainingConfig,
    apply_ddp_fast_torch_backends,
    build_full_bleu_monitor_cfg_override,
    build_resolved_training_config,
    format_full_bleu_eval_epoch_decision_log_line,
    format_full_bleu_eval_resolved_log_line,
    format_full_bleu_monitor_log_line,
    get_eval_batch_size,
    get_dataloader_num_workers,
    get_dataloader_prefetch_factor,
    get_num_proc,
    hf_datasets_progress_bar,
    resolve_task_idx_from_aux_target,
    should_run_full_bleu_eval_epoch,
)
from training_hardware_inputs import collect_training_hardware_overrides_from_args
from d4c_core.training_diagnostics import training_diagnostics_snapshot
from lr_schedule_utils import resolve_warmup_steps, warmup_cosine_multiplier_lambda
from bleu_valid_ddp import bleu4_explanation_full_valid_ddp
from d4c_core.bleu_runtime import explanation_bleu4_quick_score
from d4c_core.gather_schema import GatheredBatch, require_gathered_batch
from train_logging import (
    create_run_paths,
    setup_train_logging,
    log_run_header,
    log_config_snapshot,
    flush_preset_load_events,
    format_epoch_training_block,
    log_epoch_training_block,
    broadcast_run_paths_ddp,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    append_eval_run_summaries,
    LOGGER_NAME,
    logger_has_file_handler,
    log_route_extra,
    ROUTE_DETAIL,
    ROUTE_SUMMARY,
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
    log_bf16_amp_note,
    log_grad_monitor,
    log_step_sample,
    log_training_crash,
    maybe_log_grad_norm_diff_ddp,
    parse_d4c_finite_check_mode,
    run_training_finite_checks,
    warn_empty_batch,
)

_EVAL_REQUIRES_TORCHRUN_MSG = (
    "step3 runner 的 eval 仅支持 torchrun / python -m torch.distributed.run 下的 DDP。\n"
    "用户日常：在项目根执行  python code/d4c.py step3 --eval-only …\n"
    "请勿使用 `python <薄壳>.py eval` 在非 torchrun 环境下直接启动。\n"
    "高级排障（须 torchrun，在 code/ 目录）见 docs/D4C_Scripts_and_Runtime_Guide.md 附录。\n"
    "多卡请设置 CUDA_VISIBLE_DEVICES 并使 nproc_per_node 与可见 GPU 数一致。"
)

_t5_base = get_t5_small_dir()
_t5_path = _t5_base if os.path.exists(_t5_base) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)

# HuggingFace tokenize 磁盘缓存：修改 Processor/tokenize 语义或需强制失效时与 step5 引擎同步递增
D4C_TOKENIZE_CACHE_VERSION = "v4"

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


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        hidden_states = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            hidden_states.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return output, hidden_states


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels
        loss = self.bce_with_logits(input, target)
        return loss


class Discriminator_u(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator_u, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.disc_u = Discriminator_u(hidden_size)
        self.disc_i = Discriminator_u(hidden_size)

    def forward(self, hidden_u, hidden_i):
        u_output = self.disc_u(hidden_u)
        i_output = self.disc_i(hidden_i)
        return u_output, i_output


class Processor():
    def __init__(self, auxiliary, target):
        self.max_length = 25
        self.auxiliary = auxiliary
        self.target = target

    def __call__(self, sample):
        user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
        item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
        raitng = torch.tensor(sample["rating"], dtype=torch.float)
        explanation = sample["explanation"]
        explanation_idx = tokenizer(explanation, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
        explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)

        if sample["domain"] == "auxiliary":
            domain_val = 0
        elif sample["domain"] == "target":
            domain_val = 1
        else:
            raise ValueError("Unknown domain!")

        domain_idx = torch.tensor(domain_val, dtype=torch.long)
        sample_id = torch.tensor(int(sample["sample_id"]), dtype=torch.long)
        return {
            "user_idx": user_idx,
            "item_idx": item_idx,
            "rating": raitng,
            "explanation_idx": explanation_idx,
            "domain_idx": domain_idx,
            "sample_id": sample_id,
        }


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

    def forward(self, hidden):
        mlp_vector = self.sigmoid(self.linear1(hidden))
        rating = self.linear2(mlp_vector).view(-1)
        return rating


class Model(nn.Module):
    def __init__(self, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, user_profiles, item_profiles, domain_profiles):
        super().__init__()
        self.domain_profiles = nn.Parameter(domain_profiles)
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.user_profiles = nn.Parameter(user_profiles)
        self.item_profiles = nn.Parameter(item_profiles)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.recommender = PETER_MLP(emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layers, nlayers)
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.emsize = emsize
        self.ntoken = int(ntoken)
        self.rating_loss_fn = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, user, item, tgt_input, domain_idx):
        device = user.device
        domain_embedding = self.domain_profiles[domain_idx].unsqueeze(dim=1)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        word_feature = self.word_embeddings(tgt_input)
        src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = generate_domain_mask(tgt_input.shape[1], device)
        hidden, hiddens = self.transformer_encoder(src=src, mask=attn_mask)
        user_hidden = hiddens[0][:,1]
        item_hidden = hiddens[0][:,2]
        rating = self.recommender(hidden[:,3])
        context_dist = self.hidden2token(hidden[:,4]).unsqueeze(1).repeat(1, tgt_input.shape[1], 1)
        word_dist = self.hidden2token(hidden[:,5:])
        return rating, context_dist, word_dist, user_hidden, item_hidden

    def gather(self, batch, device):
        user_idx, item_idx, rating, tgt_output, domain_idx, sample_id = batch
        # 配合 DataLoader(pin_memory=True) 使用 non_blocking=True，减少同步拷贝等待
        user_idx = user_idx.to(device, non_blocking=True)
        item_idx = item_idx.to(device, non_blocking=True)
        domain_idx = domain_idx.to(device, non_blocking=True)
        rating = rating.to(device, non_blocking=True).float()
        tgt_output = tgt_output.to(device, non_blocking=True)
        sample_id = sample_id.to(device, non_blocking=True)
        tgt_input = T5_shift_right(tgt_output)
        return GatheredBatch(
            user_idx=user_idx,
            item_idx=item_idx,
            rating=rating,
            tgt_input=tgt_input,
            tgt_output=tgt_output,
            domain_idx=domain_idx,
            sample_id=sample_id,
            exp_sample_weight=None,
        )

    def recommend(self, user, item, domain):
        domain_embedding = self.domain_profiles[domain].unsqueeze(dim=1)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, _ = self.transformer_encoder(src)
        rating = self.recommender(hidden[:,3])
        return rating

    def generate(self, user, item, domain):
        total_entropies = []
        max_len = 25
        bos_idx = 0
        device = user.device
        batch_size = user.shape[0]
        domain_embedding = self.domain_profiles[domain].unsqueeze(dim=1)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)
        for i in range(max_len):
            word_feature = self.word_embeddings(decoder_input_ids)
            src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)
            attn_mask = generate_domain_mask(decoder_input_ids.shape[1], device)
            hidden, _ = self.transformer_encoder(src=src, mask=attn_mask)
            dist = self.hidden2token(hidden).softmax(dim=-1)
            output_id = dist[:,-1,:].topk(1).indices
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
            entropies = compute_entropy(dist)
            total_entropies.append(entropies)
        total_entropies = torch.stack(total_entropies).mean(dim=0)
        return decoder_input_ids[:,1:], total_entropies


def validModel(model, valid_dataloader, device):
    _model = get_underlying_model(model)
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            g = require_gathered_batch(_model.gather(batch, device))
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = (
                g.user_idx,
                g.item_idx,
                g.rating,
                g.tgt_input,
                g.tgt_output,
                g.domain_idx,
            )
            with d4c_cuda_bf16_autocast():
                pred_rating, _, word_dist, _, _, = model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
            loss = loss_r + loss_e
            avg_loss += loss.item()
        avg_loss /= len(valid_dataloader)
        return avg_loss


def validModel_sum_batches(model, valid_dataloader, device):
    """
    DDP 训练时用于验证聚合（与 train loss、Step5 valid 一致：样本加权）：
    - 返回 (loss_sum, n_samples)，其中 loss_sum = Σ (batch 标量 loss × batch 内样本数)。
    - 外层对两维做 all_reduce(SUM) 后，current_valid_loss = 全局 loss_sum / 全局 n_samples。
    - 验证集在各 rank 上为无重叠划分（见 build_config_and_data_ddp 中 Subset + 索引），
      避免 DistributedSampler 补重复样本导致全局均值偏差。
    """
    _model = get_underlying_model(model)
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        n_samples = 0
        for batch in valid_dataloader:
            g = require_gathered_batch(_model.gather(batch, device))
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = (
                g.user_idx,
                g.item_idx,
                g.rating,
                g.tgt_input,
                g.tgt_output,
                g.domain_idx,
            )
            bsz = int(user_idx.size(0))
            with d4c_cuda_bf16_autocast():
                pred_rating, _, word_dist, _, _, = model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
            loss = loss_r + loss_e
            loss_sum += loss.detach().double().item() * bsz
            n_samples += bsz
        return loss_sum, n_samples


def _log_tokenize_done(
    phase: str,
    nproc: int,
    elapsed_s: float,
    log_file: Optional[str],
    *,
    also_print: bool = True,
) -> None:
    """datasets.map（desc=Tokenize）结束后的显式耗时与 num_proc，便于在日志中检索。"""
    msg = f"[Tokenize] {phase} 完成 | num_proc={nproc} | wall_time={elapsed_s:.2f}s"
    lg = logging.getLogger(LOGGER_NAME)
    # 已由 FileHandler 写文件时不再 print，避免与 shell 重定向双份
    if also_print and not logger_has_file_handler(lg):
        print(msg, flush=True)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


def _dist_barrier_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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
    valid_path: str,
    tok,
    max_length: int,
    cache_version: str,
) -> str:
    """稳定、简洁的 cache key 段（含可读版本前缀 + 12 位 sha1 截断）。"""
    parts = [
        f"train={os.path.abspath(train_path)}",
        f"train_mtime={_safe_file_mtime(train_path)}",
        f"valid={os.path.abspath(valid_path)}",
        f"valid_mtime={_safe_file_mtime(valid_path)}",
        f"tok={_tokenizer_cache_identity(tok)}",
        f"maxlen={int(max_length)}",
        f"ver={cache_version}",
    ]
    raw = "|".join(parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{cache_version}_{h}"


def _build_step3_cache_dir(
    task_idx: int,
    train_path: str,
    valid_path: str,
    processor,
    tok,
    cache_version: str = D4C_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str]:
    fp = _build_tokenize_cache_fingerprint(
        train_path=train_path,
        valid_path=valid_path,
        tok=tok,
        max_length=int(getattr(processor, "max_length", 25)),
        cache_version=cache_version,
    )
    cache_dir = os.path.join(get_hf_cache_root(task_idx), f"hf_cache_step3_{fp}")
    return cache_dir, fp


def _build_eval_tokenize_cache_fingerprint(
    *,
    eval_data_path: str,
    tok,
    max_length: int,
    cache_version: str,
) -> str:
    """
    Step3 eval 子命令专用 tokenize 缓存 key（与 train 的 train+valid 缓存独立）。
    字段需覆盖：eval 数据路径、mtime、tokenizer、max_length、版本、mode=eval。
    """
    parts = [
        "mode=eval",
        f"data={os.path.abspath(eval_data_path)}",
        f"data_mtime={_safe_file_mtime(eval_data_path)}",
        f"tok={_tokenizer_cache_identity(tok)}",
        f"maxlen={int(max_length)}",
        f"ver={cache_version}",
    ]
    raw = "|".join(parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{cache_version}_{h}"


def _build_step3_eval_cache_dir(
    task_idx: int,
    eval_data_path: str,
    processor,
    tok,
    cache_version: str = D4C_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str]:
    fp = _build_eval_tokenize_cache_fingerprint(
        eval_data_path=eval_data_path,
        tok=tok,
        max_length=int(getattr(processor, "max_length", 25)),
        cache_version=cache_version,
    )
    cache_dir = os.path.join(get_hf_cache_root(task_idx), f"hf_cache_step3_eval_{fp}")
    return cache_dir, fp


def _hf_dataset_cache_ready(cache_dir: str) -> bool:
    return os.path.isdir(cache_dir) and os.path.isfile(os.path.join(cache_dir, "dataset_dict.json"))


def _log_tokenize_cache_line(msg: str, log_file: Optional[str]) -> None:
    lg = logging.getLogger(LOGGER_NAME)
    if not logger_has_file_handler(lg):
        print(msg, flush=True)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


def _map_tokenize_train_valid_to_hf_cache(
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
    log_file: Optional[str],
) -> DatasetDict:
    """
    rank0 负责 map + save_to_disk；barrier 后所有 rank load_from_disk。
    cache 已存在时各 rank 直接 load（跳过 map）。
    """
    if _hf_dataset_cache_ready(cache_dir):
        t_hit0 = time.perf_counter()
        encoded_data = load_from_disk(cache_dir)
        elapsed_hit = time.perf_counter() - t_hit0
        if rank == 0 and log_tokenize:
            msg = (
                f"[Tokenize] {phase} cache hit | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"load_wall_time={elapsed_hit:.2f}s"
            )
            _log_tokenize_cache_line(msg, log_file)
        return encoded_data

    if rank == 0:
        t0 = time.perf_counter()
        with hf_datasets_progress_bar(show_datasets_progress):
            encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
        encoded_data.save_to_disk(cache_dir)
        elapsed = time.perf_counter() - t0
        if log_tokenize:
            _log_tokenize_done(phase, nproc, elapsed, log_file)
            msg = (
                f"[Tokenize] {phase} cache miss | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"build_wall_time={elapsed:.2f}s"
            )
            _log_tokenize_cache_line(msg, log_file)

    _dist_barrier_if_initialized()
    encoded_data = load_from_disk(cache_dir)
    return encoded_data


def _load_advtrain_artefacts(
    args,
    device: int,
    resolved: FinalTrainingConfig,
    *,
    rank: int = 0,
    log_tokenize: bool = True,
    show_datasets_progress: bool = True,
):
    task_idx = int(resolved.task_idx)
    path = os.path.join(get_merged_data_dir(), str(task_idx))
    train_path = os.path.join(path, "aug_train.csv")
    valid_path = os.path.join(path, "aug_valid.csv")
    train_df = pd.read_csv(train_path)
    nuser = int(train_df["user_idx"].max()) + 1
    nitem = int(train_df["item_idx"].max()) + 1

    os.makedirs(get_stage_run_dir(task_idx), exist_ok=True)
    nproc = int(resolved.num_proc)
    save_file = args.save_file or os.path.join(get_stage_run_dir(task_idx), "model", "model.pth")

    valid_df = pd.read_csv(valid_path)
    train_df["item"] = train_df["item"].astype(str)
    valid_df["item"] = valid_df["item"].astype(str)
    # 与 step5 一致：仅保留有 explanation 的训练行（nuser/nitem 已在上方用过滤前 train_df 计算）
    train_df = train_df[train_df["explanation"].notna()].reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    train_df["sample_id"] = np.arange(len(train_df), dtype=np.int64)
    valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "valid": Dataset.from_pandas(valid_df),
    })
    processor = Processor(args.auxiliary, args.target)
    cache_dir, cache_fp = _build_step3_cache_dir(
        task_idx, train_path, valid_path, processor, tokenizer,
    )
    if rank == 0 and log_tokenize:
        _log_tokenize_cache_line(
            f"[Tokenize] step3 cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
            getattr(args, "log_file", None),
        )
    _tok_lg = logging.getLogger(LOGGER_NAME)
    with d4c_timing_phase(
        _tok_lg,
        "tokenize_pipeline_step3_train_valid",
        route=ROUTE_SUMMARY,
        rank=rank,
    ):
        encoded_data = _map_tokenize_train_valid_to_hf_cache(
            datasets=datasets,
            processor=processor,
            nproc=nproc,
            cache_dir=cache_dir,
            cache_fingerprint=cache_fp,
            rank=rank,
            show_datasets_progress=show_datasets_progress,
            log_tokenize=log_tokenize,
            phase="train+valid",
            log_file=getattr(args, "log_file", None),
        )
    encoded_data.set_format("torch")
    train_dataset = TensorDataset(
        encoded_data["train"]["user_idx"], encoded_data["train"]["item_idx"],
        encoded_data["train"]["rating"], encoded_data["train"]["explanation_idx"],
        encoded_data["train"]["domain_idx"], encoded_data["train"]["sample_id"],
    )
    valid_dataset = TensorDataset(
        encoded_data["valid"]["user_idx"], encoded_data["valid"]["item_idx"],
        encoded_data["valid"]["rating"], encoded_data["valid"]["explanation_idx"],
        encoded_data["valid"]["domain_idx"], encoded_data["valid"]["sample_id"],
    )

    suser_profiles = torch.tensor(
        np.load(os.path.join(get_data_dir(), args.auxiliary, "user_profiles.npy")),
        dtype=torch.float, device=device,
    )
    sitem_profiles = torch.tensor(
        np.load(os.path.join(get_data_dir(), args.auxiliary, "item_profiles.npy")),
        dtype=torch.float, device=device,
    )
    sdomain_profiles = torch.tensor(
        np.load(os.path.join(get_data_dir(), args.auxiliary, "domain.npy")),
        dtype=torch.float, device=device,
    )
    tuser_profiles = torch.tensor(
        np.load(os.path.join(get_data_dir(), args.target, "user_profiles.npy")),
        dtype=torch.float, device=device,
    )
    titem_profiles = torch.tensor(
        np.load(os.path.join(get_data_dir(), args.target, "item_profiles.npy")),
        dtype=torch.float, device=device,
    )
    tdomain_profiles = torch.tensor(
        np.load(os.path.join(get_data_dir(), args.target, "domain.npy")),
        dtype=torch.float, device=device,
    )

    domain_profiles = torch.cat([sdomain_profiles.unsqueeze(0), tdomain_profiles.unsqueeze(0)], dim=0)
    user_profiles = torch.cat([tuser_profiles, suser_profiles], dim=0)
    item_profiles = torch.cat([titem_profiles, sitem_profiles], dim=0)

    model = Model(
        nuser,
        nitem,
        int(len(tokenizer)),
        resolved.emsize,
        resolved.nhead,
        resolved.nhid,
        args.nlayers,
        resolved.dropout,
        user_profiles,
        item_profiles,
        domain_profiles,
    ).to(device)
    discriminator = Discriminator(resolved.emsize).to(device)
    return train_dataset, valid_dataset, model, discriminator, nuser, nitem, save_file


def _distributed_valid_sample_indices(n_valid: int, world_size: int, rank: int) -> list[int]:
    """
    验证集按样本无重叠划分到各 rank（不补重复样本）。
    与 DistributedSampler(drop_last=False) 不同：后者为对齐 total_size 会复制索引，导致
    valid loss 的 all_reduce 加权平均偏离「全验证集各样本恰好一次」的真实均值。
    分片为连续区间，与 bleu4_explanation_full_valid_ddp 的按 rank 切块一致（仅按余数平衡各卡条数）。
    """
    if n_valid <= 0:
        return []
    if world_size <= 1:
        return list(range(n_valid))
    base = n_valid // world_size
    rem = n_valid % world_size
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    return list(range(start, start + size))


def build_config_and_data_ddp(args, rank: int, world_size: int, local_rank: int) -> tuple:
    _tid = resolve_task_idx_from_aux_target(args.auxiliary, args.target)
    if _tid is None:
        raise ValueError("未知的 auxiliary/target 组合")
    _ro = collect_training_hardware_overrides_from_args(args)
    resolved = build_resolved_training_config(
        args,
        task_idx=_tid,
        world_size=world_size,
        hardware_overrides=_ro,
    )
    G = int(resolved.train_batch_size)
    P = int(resolved.per_device_train_batch_size)
    A = int(resolved.gradient_accumulation_steps)

    train_dataset, valid_dataset, model, discriminator, nuser, nitem, save_file = _load_advtrain_artefacts(
        args,
        local_rank,
        resolved,
        rank=rank,
        log_tokenize=(rank == 0),
        show_datasets_progress=(rank == 0),
    )

    train_drop_last = A > 1
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=train_drop_last,
    )
    pin_memory = torch.cuda.is_available()
    nw_train = int(resolved.dataloader_num_workers_train)
    nw_valid = int(resolved.dataloader_num_workers_valid)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=P,
        sampler=sampler,
        shuffle=False,
        num_workers=nw_train,
        pin_memory=pin_memory,
        persistent_workers=nw_train > 0,
        prefetch_factor=resolved.dataloader_prefetch_factor_train,
        drop_last=train_drop_last,
    )
    n_train_micro = len(train_dataloader)
    if A > 1 and n_train_micro % A != 0:
        raise ValueError(
            f"train DataLoader 每 epoch 批次数为 {n_train_micro}，无法被 gradient_accumulation_steps={A} 整除。"
            f"请调整全局 batch、--per-device-batch-size、world_size 或数据划分；或令 accum=1。"
        )
    if G % world_size != 0:
        raise ValueError(
            f"train_batch_size={G} 与 world_size={world_size} 不整除，无法得到每卡 batch。"
            "请修改 training preset（train_batch_size / per_device_train_batch_size / gradient_accumulation_steps）或 hardware preset（ddp_world_size）。"
        )
    valid_per_rank = G // world_size
    n_valid = len(valid_dataset)
    if world_size > 1:
        _v_idx = _distributed_valid_sample_indices(n_valid, world_size, rank)
        # 即使某 rank 分片为空也必须用 Subset(..., [])，不能用「if _v_idx」回退全量（[] 在 Python 中为假值）
        valid_shard: TensorDataset | Subset = Subset(valid_dataset, _v_idx)
    else:
        valid_shard = valid_dataset
    valid_dataloader = DataLoader(
        valid_shard,
        batch_size=valid_per_rank,
        shuffle=False,
        num_workers=nw_valid,
        pin_memory=pin_memory,
        persistent_workers=nw_valid > 0,
        prefetch_factor=resolved.dataloader_prefetch_factor_valid,
    )

    _ddp_find_unused = bool(resolved.ddp_find_unused_parameters)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=_ddp_find_unused,
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=_ddp_find_unused,
    )
    final_cfg = replace(
        resolved,
        nuser=nuser,
        nitem=nitem,
        save_file=save_file,
        device=local_rank,
        device_ids=tuple(range(world_size)),
        ddp_world_size=world_size,
        nlayers=args.nlayers,
        valid_dataset=valid_dataset,
        ddp_find_unused_parameters=_ddp_find_unused,
        rank0_only_logging=True,
    )

    return final_cfg, train_dataloader, valid_dataloader, model, discriminator, sampler


@contextlib.contextmanager
def _ddp_no_sync_both(model, discriminator, world_size: int, sync_gradients: bool):
    """在梯度累积的非边界微批上使用 DDP no_sync，边界步上规约梯度。"""
    if world_size <= 1 or sync_gradients:
        yield
    else:
        with model.no_sync(), discriminator.no_sync():
            yield


def _effective_adversarial_coef_epoch(
    epoch_1_based: int,
    *,
    schedule_enabled: bool,
    skip_epochs: int,
    warmup_epochs: int,
    target: float,
    legacy_coef: float,
) -> float:
    """未启用 schedule 时恒为 legacy_coef；启用时前 skip_epochs 个 epoch 系数为 0，之后线性升至 target。"""
    if not schedule_enabled:
        return float(legacy_coef)
    if epoch_1_based <= skip_epochs:
        return 0.0
    if warmup_epochs <= 0:
        return float(target)
    t = epoch_1_based - skip_epochs
    return float(target) * min(1.0, t / float(warmup_epochs))


def trainModel_ddp(
    model,
    discriminator,
    train_dataloader,
    valid_dataloader,
    sampler,
    final_cfg: FinalTrainingConfig,
    rank,
    world_size,
    max_steps=None,
    save_final_checkpoint: bool = False,
):
    epochs = final_cfg.epochs
    G = int(final_cfg.train_batch_size)
    P = int(final_cfg.per_device_train_batch_size)
    A = max(1, int(final_cfg.gradient_accumulation_steps))
    eff = int(final_cfg.effective_global_batch_size)
    initial_lr = float(final_cfg.scheduler_initial_lr)
    learning_rate = initial_lr
    coef = float(final_cfg.coef)
    adversarial_coef_legacy = float(final_cfg.adversarial_coef)
    adv_schedule_on = bool(final_cfg.adversarial_schedule_enabled)
    adv_skip_epochs = int(final_cfg.adversarial_start_epoch)
    adv_warmup_epochs = int(final_cfg.adversarial_warmup_epochs)
    adv_coef_target = float(final_cfg.adversarial_coef_target)
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
    best_bleu4 = -1.0
    best_full_bleu4 = -1.0
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

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=initial_lr * 0.5, weight_decay=1e-5)
    sched = d_sched = None
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
        d_sched = lr_sched.LambdaLR(d_optimizer, lr_lambda)
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
    adversarial_loss_fn = BCEWithLogitsLoss(label_smoothing=0.3)
    if rank == 0:
        if adv_schedule_on:
            _adv_msg = (
                "Adversarial schedule: skip_first_epochs=%d (no D / no G adv), then warmup_epochs=%d to coef_target=%g "
                "(--adv legacy=%g); first adversarial epoch index = %d (1-based)"
                % (
                    adv_skip_epochs,
                    adv_warmup_epochs,
                    adv_coef_target,
                    adversarial_coef_legacy,
                    adv_skip_epochs + 1,
                )
            )
            if _lg:
                _lg.info(_adv_msg, extra=log_route_extra(_lg, ROUTE_SUMMARY))
            else:
                print(_adv_msg, flush=True)
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
    step_count = 0
    global_step = 0
    step_iv = max(1, d4c_log_step_interval())
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
            "[DDP] ddp_find_unused_parameters=%s（来自 FinalTrainingConfig）",
            bool(final_cfg.ddp_find_unused_parameters),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[Diag] D4C_GRAD_TOPK=%d（仅 >0 时 log_grad_monitor 打印 top 参数 grad norm）",
            d4c_grad_topk(),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            epoch_1 = epoch + 1
            adv_coef_epoch = _effective_adversarial_coef_epoch(
                epoch_1,
                schedule_enabled=adv_schedule_on,
                skip_epochs=adv_skip_epochs,
                warmup_epochs=adv_warmup_epochs,
                target=adv_coef_target,
                legacy_coef=adversarial_coef_legacy,
            )
            use_adv = adv_coef_epoch > 0.0
            if rank == 0:
                perf.epoch_start()
            model.train()
            discriminator.train()
            loss_sum = torch.zeros((), dtype=torch.double, device=device)
            adv_sum = torch.zeros((), dtype=torch.double, device=device)
            n_samples = torch.zeros((), dtype=torch.double, device=device)
            micro_step_epoch = 0
            d_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            inv_accum = 1.0 / float(A)
            iterator = train_dataloader
            if rank == 0:
                iterator = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in iterator:
                step_count += 1
                micro_step_epoch += 1
                sync = micro_step_epoch % A == 0
                sync_ctx = _ddp_no_sync_both(model, discriminator, world_size, sync)
                g = require_gathered_batch(_model.gather(batch, device))
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = (
                    g.user_idx,
                    g.item_idx,
                    g.rating,
                    g.tgt_input,
                    g.tgt_output,
                    g.domain_idx,
                )
                bsz = int(user_idx.size(0))
                warn_empty_batch(_lg, global_step=global_step, epoch=epoch_1, bsz=bsz)
                if use_adv:
                    # 同一微批上 D 与 G 的两次 backward 须在同一段 no_sync 内，避免中间触发规约
                    with sync_ctx:
                        with d4c_cuda_bf16_autocast():
                            pred_rating, context_dist, word_dist, user_hidden, item_hidden = model(
                                user_idx, item_idx, tgt_input, domain_idx,
                            )
                            target_labels = torch.ones(user_hidden.size(0), 1, device=device)
                            aux_labels = torch.zeros(user_hidden.size(0), 1, device=device)

                            aux_u_output, aux_i_output = discriminator(
                                user_hidden[domain_idx == 0], item_hidden[domain_idx == 0],
                            )
                            d_loss_aux_u = adversarial_loss_fn(aux_u_output, aux_labels[domain_idx == 0])
                            d_loss_aux_i = adversarial_loss_fn(aux_i_output, aux_labels[domain_idx == 0])
                            target_u_output, target_i_output = discriminator(
                                user_hidden[domain_idx == 1], item_hidden[domain_idx == 1],
                            )
                            d_loss_target_u = adversarial_loss_fn(target_u_output, target_labels[domain_idx == 1])
                            d_loss_target_i = adversarial_loss_fn(target_i_output, target_labels[domain_idx == 1])
                            d_loss = (d_loss_aux_u + d_loss_aux_i + d_loss_target_u + d_loss_target_i) / 4.0
                        (d_loss * inv_accum).backward()

                        with d4c_cuda_bf16_autocast():
                            pred_rating, context_dist, word_dist, user_hidden, item_hidden = model(
                                user_idx, item_idx, tgt_input, domain_idx,
                            )
                            fake_aux_u_output, fake_aux_i_output = discriminator(
                                user_hidden[domain_idx == 1], item_hidden[domain_idx == 1],
                            )
                            g_loss_u = adversarial_loss_fn(fake_aux_u_output, aux_labels[domain_idx == 1])
                            g_loss_i = adversarial_loss_fn(fake_aux_i_output, aux_labels[domain_idx == 1])
                            g_loss = (g_loss_u + g_loss_i) / 2.0
                            loss_r = _model.rating_loss_fn(pred_rating, rating)
                            loss_e = _model.exp_loss_fn(word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
                            loss_c = _model.exp_loss_fn(context_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
                            loss = 0.1 * loss_r + coef * loss_c + loss_e + adv_coef_epoch * g_loss
                        (loss * inv_accum).backward()

                    if sync:
                        if (global_step + 1) % grad_iv == 0:
                            log_grad_monitor(
                                _lg,
                                model,
                                global_step=global_step + 1,
                                epoch=epoch_1,
                                route_detail=ROUTE_DETAIL,
                            )
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        d_optimizer.step()
                        optimizer.step()
                        d_optimizer.zero_grad(set_to_none=True)
                        optimizer.zero_grad(set_to_none=True)
                        # LambdaLR：必须在 optimizer.step() 之后调用，使内部 step 与全局优化步一致
                        if sched is not None:
                            sched.step()
                        if d_sched is not None:
                            d_sched.step()
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
                            if d4c_log_step_loss_parts():
                                _extra = {
                                    "loss_r": float(loss_r.detach().item()),
                                    "loss_e": float(loss_e.detach().item()),
                                    "loss_c": float(loss_c.detach().item()),
                                    "d_loss": float(d_loss.detach().item()),
                                    "g_loss": float(g_loss.detach().item()),
                                }
                            log_step_sample(
                                _lg,
                                global_step=global_step,
                                epoch=epoch_1,
                                lr=float(_lr_now),
                                train_loss_batch=float(loss.detach().item()),
                                extra=_extra,
                            )
                    adv_sum = adv_sum + g_loss.detach().double() * bsz
                else:
                    with sync_ctx:
                        with d4c_cuda_bf16_autocast():
                            pred_rating, context_dist, word_dist, user_hidden, item_hidden = model(
                                user_idx, item_idx, tgt_input, domain_idx,
                            )
                            loss_r = _model.rating_loss_fn(pred_rating, rating)
                            loss_e = _model.exp_loss_fn(word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
                            loss_c = _model.exp_loss_fn(context_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
                            loss = 0.1 * loss_r + coef * loss_c + loss_e
                        (loss * inv_accum).backward()
                    if sync:
                        if (global_step + 1) % grad_iv == 0:
                            log_grad_monitor(
                                _lg,
                                model,
                                global_step=global_step + 1,
                                epoch=epoch_1,
                                route_detail=ROUTE_DETAIL,
                            )
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # LambdaLR：必须在 optimizer.step() 之后调用
                        if sched is not None:
                            sched.step()
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
                            if d4c_log_step_loss_parts():
                                _extra = {
                                    "loss_r": float(loss_r.detach().item()),
                                    "loss_e": float(loss_e.detach().item()),
                                    "loss_c": float(loss_c.detach().item()),
                                }
                            log_step_sample(
                                _lg,
                                global_step=global_step,
                                epoch=epoch_1,
                                lr=float(_lr_now),
                                train_loss_batch=float(loss.detach().item()),
                                extra=_extra,
                            )

                loss_sum = loss_sum + loss.detach().double() * bsz
                n_samples += bsz
                if step_count % step_iv == 0 and rank == 0:
                    run_training_finite_checks(
                        _finite_mode,
                        loss,
                        word_dist,
                        _lg,
                        global_step=global_step,
                        epoch=epoch_1,
                        route_detail=ROUTE_DETAIL,
                    )

                # 用于快速验证：跑到指定 steps 后直接退出，观察是否触发 DDP reduction 错误
                if max_steps is not None and step_count >= max_steps:
                    return

            ddp_heartbeat(_lg, "before_train_loss_allreduce", rank=rank, epoch=epoch_1)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(adv_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
            ddp_heartbeat(_lg, "after_train_loss_allreduce", rank=rank, epoch=epoch_1)
            avg_loss = (loss_sum / n_samples).item()
            avg_adv_loss = (adv_sum / n_samples).item()

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

            # 各 rank valid 分片上样本加权求和，一次 all_reduce 得全局 avg valid loss（与 train / Step5 一致）。
            _t_valid0 = time.perf_counter()
            valid_loss_sum, valid_n_samples = validModel_sum_batches(model, valid_dataloader, device)
            if rank == 0 and _lg:
                _lg.info(
                    "[Timing] valid_loss_forward end epoch=%d elapsed_s=%.3f",
                    epoch_1,
                    time.perf_counter() - _t_valid0,
                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                )
            v_stat = torch.tensor(
                [valid_loss_sum, float(valid_n_samples)],
                dtype=torch.double,
                device=device,
            )
            dist.all_reduce(v_stat, op=dist.ReduceOp.SUM)
            current_valid_loss = float(v_stat[0] / v_stat[1]) if v_stat[1] > 0 else 0.0

            bleu4_score_this_epoch = None
            quick_bleu4 = None
            full_bleu4_val = None
            is_full_eval_epoch = (
                checkpoint_metric == "bleu4"
                and valid_dataset_for_bleu is not None
                and should_run_full_bleu_eval_epoch(epoch + 1, fe_sched)
            )
            if rank == 0 and _lg and checkpoint_metric == "bleu4" and valid_dataset_for_bleu is not None:
                _lg.info(
                    format_full_bleu_eval_epoch_decision_log_line(epoch_1, is_full_eval_epoch),
                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                )
            if checkpoint_metric == "bleu4" and valid_dataset_for_bleu is not None:
                if rank == 0:
                    with d4c_timing_phase(
                        _lg,
                        f"bleu_quick_epoch_{epoch_1}",
                        route=ROUTE_SUMMARY,
                        rank=0,
                    ):
                        quick_bleu4 = explanation_bleu4_quick_score(
                            model,
                            tokenizer,
                            valid_dataset_for_bleu,
                            device,
                            quick_eval_max_samples,
                            rank=0,
                            logger=_lg,
                            dataloader_num_workers=min(2, final_cfg.dataloader_num_workers_valid),
                            dataloader_prefetch_factor=final_cfg.dataloader_prefetch_factor_valid,
                        )
                    bleu4_score_this_epoch = quick_bleu4
                if world_size > 1:
                    dist.barrier()
                if is_full_eval_epoch:
                    _t_full_bleu = time.perf_counter()
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
                        cfg_override=full_bleu_monitor_cfg_override,
                    )
                    if rank == 0 and _lg:
                        _lg.info(
                            "[Timing] bleu_full_valid_ddp end epoch=%d elapsed_s=%.3f",
                            epoch_1,
                            time.perf_counter() - _t_full_bleu,
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bleu_line = None
                if checkpoint_metric == "bleu4" and valid_dataset_for_bleu is not None and quick_bleu4 is not None:
                    if dual_bleu:
                        fstr = f"{full_bleu4_val:.4f}" if full_bleu4_val is not None else "na"
                        ckpt_src = "full_bleu_monitor" if is_full_eval_epoch else "trend_quick_bleu4"
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
                            + " | checkpoint_metric_source=quick_bleu4"
                        )
                lr_sched_line = None
                if sched is not None and ws_resolved is not None:
                    lr_sched_line = (
                        f"scheduler_type=warmup_cosine "
                        f"initial_lr={initial_lr:.6g} current_lr={lr_epoch:.6g} min_lr={min_lr_effective:.6g} "
                        f"min_lr_ratio={min_lr_ratio:.6g} warmup_steps={ws_resolved} total_steps={total_steps_plan} "
                        f"scheduler_steps_end_of_epoch={global_step} warmup_ratio={warmup_ratio_logged:.6g}"
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
                    adv_loss=avg_adv_loss,
                    adversarial_coef=adv_coef_epoch,
                    bleu_line=bleu_line,
                    lr_schedule_detail=lr_sched_line,
                )
                log_epoch_training_block(_lg, block)

            if dual_bleu and checkpoint_metric == "bleu4":
                if current_valid_loss > prev_valid_loss:
                    valid_loss_stall += 1
                    if lr_scheduler_type != "warmup_cosine":
                        learning_rate /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = learning_rate
                        for param_group in d_optimizer.param_groups:
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
                        )
            else:
                if current_valid_loss > prev_valid_loss:
                    enduration += 1
                    if lr_scheduler_type != "warmup_cosine":
                        learning_rate /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = learning_rate
                        for param_group in d_optimizer.param_groups:
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
                        )
            prev_valid_loss = current_valid_loss

            if (
                dual_bleu
                and checkpoint_metric == "bleu4"
                and is_full_eval_epoch
                and full_bleu4_val is not None
            ):
                if full_bleu4_val > best_full_bleu4:
                    best_full_bleu4 = full_bleu4_val
                    full_eval_stall = 0
                    if rank == 0:
                        d4c_save_checkpoint(
                            get_underlying_model(model).state_dict(),
                            str(final_cfg.save_file),
                            epoch=epoch_1,
                            reason="dual_bleu_full_bleu_monitor_improved",
                            logger=_lg,
                        )
                else:
                    # quick BLEU 不参与早停；仅在达到 min_epochs 后，对「未刷新 best 的 full eval」计数
                    if epoch + 1 >= min_epochs:
                        full_eval_stall += 1
            elif (
                not dual_bleu
                and checkpoint_metric == "bleu4"
                and rank == 0
                and valid_dataset_for_bleu is not None
                and bleu4_score_this_epoch is not None
                and bleu4_score_this_epoch > best_bleu4
            ):
                best_bleu4 = bleu4_score_this_epoch
                d4c_save_checkpoint(
                    get_underlying_model(model).state_dict(),
                    str(final_cfg.save_file),
                    epoch=epoch_1,
                    reason="bleu_quick_improved",
                    logger=_lg,
                )

            if rank == 0 and dual_bleu and checkpoint_metric == "bleu4":
                _cur_full = (
                    f"{full_bleu4_val:.4f}"
                    if (is_full_eval_epoch and full_bleu4_val is not None)
                    else "na"
                )
                _should_stop = (epoch + 1) >= min_epochs and (
                    full_eval_stall >= early_stop_patience_full
                    or valid_loss_stall >= early_stop_patience_loss
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

            if dual_bleu and checkpoint_metric == "bleu4":
                if (epoch + 1) >= min_epochs and (
                    full_eval_stall >= early_stop_patience_full
                    or valid_loss_stall >= early_stop_patience_loss
                ):
                    break
            elif epoch + 1 >= min_epochs and enduration >= early_stop_patience:
                break
    finally:
        if rank == 0 and perf is not None:
            perf.finish()
        if save_final_checkpoint and rank == 0:
            _sf = os.path.abspath(os.path.expanduser(str(final_cfg.save_file)))
            d4c_save_checkpoint(
                get_underlying_model(model).state_dict(),
                _sf,
                epoch=int(epochs),
                reason="save_final_checkpoint",
                logger=_lg,
                is_last=True,
            )


def _eval_collect_shard_predictions(model, test_dataloader, device):
    _model = get_underlying_model(model)
    model = model.to(device)
    model.eval()
    prediction_ratings = []
    ground_truth_ratings = []
    prediction_exps = []
    reference_exps = []
    # inference_mode 与 no_grad 数值一致，略少开销，适合纯推理
    with torch.inference_mode():
        for batch in test_dataloader:
            g = require_gathered_batch(_model.gather(batch, device))
            user_idx, item_idx, rating, tgt_output, domain_idx = (
                g.user_idx,
                g.item_idx,
                g.rating,
                g.tgt_output,
                g.domain_idx,
            )
            with d4c_cuda_bf16_autocast():
                pred_ratings = _model.recommend(user_idx, item_idx, domain_idx)
                pred_exps, _ = _model.generate(user_idx, item_idx, domain_idx)
            prediction_ratings.extend(pred_ratings.tolist())
            ground_truth_ratings.extend(rating.tolist())
            prediction_exps.extend(tokenizer.batch_decode(pred_exps, skip_special_tokens=True))
            reference_exps.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
    return prediction_ratings, ground_truth_ratings, prediction_exps, reference_exps


def metrics_from_eval_lists(prediction_ratings, ground_truth_ratings, prediction_exps, reference_exps):
    prediction_ratings = np.array(prediction_ratings)
    ground_truth_ratings = np.array(ground_truth_ratings)
    rating_diffs = prediction_ratings - ground_truth_ratings
    mae = round(np.mean(np.abs(rating_diffs)), 4)
    rmse = round(np.sqrt(np.mean(np.square(rating_diffs))), 4)
    text_results = evaluate_text(prediction_exps, reference_exps)
    return {"recommendation": {"mae": mae, "rmse": rmse}, "explanation": text_results}


def evalModel(model, test_dataloader, device):
    pr, gt, pe, re = _eval_collect_shard_predictions(model, test_dataloader, device)
    return metrics_from_eval_lists(pr, gt, pe, re)


def build_config_and_dataloader(args, ddp_rank: int, ddp_world_size: int, local_rank: int):
    primary_device = local_rank

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    path = os.path.join(get_merged_data_dir(), str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    nuser = train_df['user_idx'].max() + 1
    nitem = train_df['item_idx'].max() + 1

    batch_size = args.batch_size if args.batch_size is not None else get_eval_batch_size()
    nproc = args.num_proc if args.num_proc is not None else get_num_proc()

    if batch_size % ddp_world_size != 0:
        raise ValueError(
            f"eval_batch_size={batch_size} 与 world_size={ddp_world_size} 不整除，DDP 评测非法。"
            "请到 presets/eval_profiles/*.yaml 修改 eval_batch_size，或调整 hardware preset 的 ddp_world_size。"
        )
    loader_batch_size = batch_size // ddp_world_size

    config = {
        "task_idx": task_idx,
        "device": primary_device if torch.cuda.is_available() else args.device,
        "log_file": args.log_file,
        "save_file": args.save_file
        or os.path.join(get_stage_run_dir(task_idx), "model", "model.pth"),
        "batch_size": loader_batch_size,
        "emsize": 768,
        "nlayers": args.nlayers,
        "nhid": 2048,
        "ntoken": len(tokenizer),
        "dropout": 0.2,
        "nuser": nuser,
        "nitem": nitem,
        "nhead": 2
    }
    config["batch_size_global"] = batch_size

    valid_path = os.path.join(path, "aug_valid.csv")
    valid_df = pd.read_csv(valid_path)
    valid_df['item'] = valid_df['item'].astype(str)
    valid_df = valid_df.reset_index(drop=True)
    valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
    datasets = DatasetDict({
        "valid": Dataset.from_pandas(valid_df)
    })
    processor = Processor(args.auxiliary, args.target)
    cache_dir, cache_fp = _build_step3_eval_cache_dir(
        task_idx=int(task_idx),
        eval_data_path=os.path.abspath(valid_path),
        processor=processor,
        tok=tokenizer,
    )
    if ddp_rank == 0:
        _log_tokenize_cache_line(
            f"[Tokenize] eval valid cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
            getattr(args, "log_file", None),
        )
    encoded_data = _map_tokenize_train_valid_to_hf_cache(
        datasets=datasets,
        processor=processor,
        nproc=nproc,
        cache_dir=cache_dir,
        cache_fingerprint=cache_fp,
        rank=ddp_rank,
        show_datasets_progress=(ddp_rank == 0),
        log_tokenize=(ddp_rank == 0),
        phase="eval valid",
        log_file=getattr(args, "log_file", None),
    )
    encoded_data.set_format("torch")
    valid_dataset = TensorDataset(
        encoded_data['valid']['user_idx'], encoded_data['valid']['item_idx'],
        encoded_data['valid']['rating'], encoded_data['valid']['explanation_idx'],
        encoded_data['valid']['domain_idx'], encoded_data['valid']['sample_id'],
    )
    n_samples = len(valid_dataset)
    shard_idx = list(range(ddp_rank, n_samples, ddp_world_size))
    valid_dataset = Subset(valid_dataset, shard_idx)
    eval_world_size = max(ddp_world_size, 1)
    dl_valid = get_dataloader_num_workers("valid")
    num_workers = min(max(1, dl_valid // eval_world_size), 8)
    pin_memory = torch.cuda.is_available()
    _pf_ev = get_dataloader_prefetch_factor(num_workers, split="valid")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=_pf_ev,
    )

    suser_profiles = torch.tensor(np.load(os.path.join(get_data_dir(), args.auxiliary, "user_profiles.npy")), dtype=torch.float)
    sitem_profiles = torch.tensor(np.load(os.path.join(get_data_dir(), args.auxiliary, "item_profiles.npy")), dtype=torch.float)
    sdomain_profiles = torch.tensor(np.load(os.path.join(get_data_dir(), args.auxiliary, "domain.npy")), dtype=torch.float)
    tuser_profiles = torch.tensor(np.load(os.path.join(get_data_dir(), args.target, "user_profiles.npy")), dtype=torch.float)
    titem_profiles = torch.tensor(np.load(os.path.join(get_data_dir(), args.target, "item_profiles.npy")), dtype=torch.float)
    tdomain_profiles = torch.tensor(np.load(os.path.join(get_data_dir(), args.target, "domain.npy")), dtype=torch.float)

    domain_profiles = torch.cat([sdomain_profiles.unsqueeze(0), tdomain_profiles.unsqueeze(0)], dim=0)
    user_profiles = torch.cat([tuser_profiles, suser_profiles], dim=0)
    item_profiles = torch.cat([titem_profiles, sitem_profiles], dim=0)

    model = Model(
        config.get("nuser"), config.get("nitem"), config.get("ntoken"),
        config.get("emsize"), config.get("nhead"), config.get("nhid"),
        config.get("nlayers"), config.get("dropout"),
        user_profiles, item_profiles, domain_profiles
    )
    _map = f"cuda:{config.get('device')}" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(config.get("save_file"), map_location=_map, weights_only=True))
    model = model.to(config.get("device"))

    return config, valid_dataloader, model


def _run_train_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    ddp_fast_backends = apply_ddp_fast_torch_backends()
    dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
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

    try:
        base_final, train_dataloader, valid_dataloader, model, discriminator, sampler = build_config_and_data_ddp(
            args, rank, world_size, local_rank,
        )
        final_cfg = replace(
            base_final,
            run_id=run_id,
            logger=train_logger,
            log_file=log_path,
            ddp_fast_backends=ddp_fast_backends,
        )
        if rank == 0:
            _meta = {
                "run_id": run_id,
                "task_idx": task_idx,
                "rank": rank,
                "world_size": world_size,
                "cuda_available": bool(torch.cuda.is_available()),
                "local_rank": local_rank,
                "learning_rate": final_cfg.learning_rate,
                "batch_size": final_cfg.train_batch_size,
                "batch_size_global": final_cfg.batch_size_global,
                "per_device_batch_size": final_cfg.per_device_train_batch_size,
                "gradient_accumulation_steps": final_cfg.gradient_accumulation_steps,
                "effective_global_batch_size": final_cfg.effective_global_batch_size,
                "epochs": final_cfg.epochs,
                "save_file": os.path.abspath(str(final_cfg.save_file)),
                "log_file": os.path.abspath(log_path),
                "auxiliary": args.auxiliary,
                "target": args.target,
                "distributed_env": collect_distributed_env_for_meta(),
            }
            log_run_header(train_logger, _meta)
            _cfg_snap = dict(final_cfg.to_log_dict())
            _cfg_snap["training_diagnostics"] = training_diagnostics_snapshot(
                diagnostics_scope="child",
                effective_training_payload_json=os.environ.get("D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""),
                ddp_find_unused_parameters_effective=bool(final_cfg.ddp_find_unused_parameters),
            )
            _cfg_snap["training_semantic_fingerprint"] = (
                (os.environ.get("D4C_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or None
            )
            _cfg_snap["generation_semantic_fingerprint"] = (
                (os.environ.get("D4C_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or None
            )
            _cfg_snap["runtime_diagnostics_fingerprint"] = (
                (os.environ.get("D4C_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip() or None
            )
            _cfg_snap["runtime_env"] = runtime_env_dict_for_config_resolved()
            log_config_snapshot(train_logger, _cfg_snap)
            train_logger.info(
                "[Fingerprints] training_semantic=%s generation_semantic=%s runtime_diag=%s",
                (os.environ.get("D4C_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or "n/a",
                (os.environ.get("D4C_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or "n/a",
                (os.environ.get("D4C_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip() or "n/a",
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            flush_preset_load_events(train_logger)
            _run_root = os.path.abspath(os.path.join(os.path.dirname(log_path), ".."))
            _cfg_resolved_path = os.path.join(_run_root, "config_resolved.json")
            with open(_cfg_resolved_path, "w", encoding="utf-8") as _cf:
                json.dump(_cfg_snap, _cf, ensure_ascii=False, indent=2, default=str)
                _cf.write("\n")
            train_logger.info(
                "[Config resolved] wrote %s",
                _cfg_resolved_path,
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )

        try:
            trainModel_ddp(
                model,
                discriminator,
                train_dataloader,
                valid_dataloader,
                sampler,
                final_cfg,
                rank,
                world_size,
                max_steps=getattr(args, "max_steps", None),
                save_final_checkpoint=bool(getattr(args, "save_final_checkpoint", False)),
            )
        except Exception as exc:
            if rank == 0:
                log_training_crash(train_logger, exc)
            raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _eval_torchrun_env() -> bool:
    return (
        "LOCAL_RANK" in os.environ
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )


def _write_eval_results_log(
    config,
    final,
    *,
    task_description: Optional[str] = None,
    pipeline: str = "AdvTrain_eval",
    domain_from: str = "",
    domain_to: str = "",
    start_time: Optional[str] = None,
    eval_elapsed: Optional[float] = None,
):
    lines = format_final_results_lines(final, task_description=task_description, start_time=start_time)
    if eval_elapsed is not None:
        _m, _s = divmod(int(eval_elapsed), 60)
        lines.append(f"Eval elapsed: {_m}m {_s}s ({eval_elapsed:.1f}s)")
    log_path = config.get("log_file")
    lg = config.get("logger")
    log_final_results_block(lg, lines)
    finalize_run_log(lg)
    append_eval_run_summaries(
        final,
        task_idx=int(config.get("task_idx") or 0),
        run_id=str(config.get("run_id") or ""),
        pipeline=pipeline,
        domain_from=domain_from,
        domain_to=domain_to,
        log_file=log_path if isinstance(log_path, str) else None,
        save_file=config.get("save_file"),
        task_description=task_description,
        start_time=start_time,
        eval_elapsed=eval_elapsed,
    )
    if lg is not None:
        lg.info("(eval 指标已写入 %s)", os.path.abspath(log_path) if log_path else log_path)
    else:
        logging.info("(eval 指标已写入 %s)", os.path.abspath(log_path) if log_path else log_path)


def _run_eval_ddp(args):
    if not torch.cuda.is_available():
        raise RuntimeError("step3 runner eval 的 torchrun DDP 需要 CUDA + NCCL。")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
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
    ev_logger = _setup["logger"]

    try:
        config, valid_dataloader, model = build_config_and_dataloader(
            args, ddp_rank=rank, ddp_world_size=world_size, local_rank=local_rank,
        )
        config["logger"] = ev_logger
        config["run_id"] = run_id
        if rank == 0:
            log_run_header(
                ev_logger,
                {
                    "run_id": run_id,
                    "task_idx": task_idx,
                    "rank": rank,
                    "world_size": world_size,
                    "mode": "eval_ddp",
                    "cuda_available": True,
                    "local_rank": local_rank,
                    "batch_size": config.get("batch_size_global", config.get("batch_size")),
                    "save_file": os.path.abspath(str(config.get("save_file", ""))),
                    "log_file": os.path.abspath(log_path),
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                },
            )
        _eval_t0 = time.time()
        _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pr, gt, pe, re = _eval_collect_shard_predictions(model, valid_dataloader, local_rank)
        payload = {
            "prediction_ratings": pr,
            "ground_truth_ratings": gt,
            "prediction_exps": pe,
            "reference_exps": re,
        }
        if rank == 0:
            gathered = [None] * world_size
        else:
            gathered = None
        dist.gather_object(payload, gathered, dst=0)

        if rank == 0:
            all_pr, all_gt, all_pe, all_re = [], [], [], []
            for shard in gathered:
                all_pr.extend(shard["prediction_ratings"])
                all_gt.extend(shard["ground_truth_ratings"])
                all_pe.extend(shard["prediction_exps"])
                all_re.extend(shard["reference_exps"])
            final = metrics_from_eval_lists(all_pr, all_gt, all_pe, all_re)
            _eval_elapsed = time.time() - _eval_t0
            _td = (
                f"Step 3 AdvTrain DDP eval Task {task_idx} (nproc={world_size}): "
                f"{args.auxiliary} -> {args.target}"
            )
            _write_eval_results_log(
                config,
                final,
                task_description=_td,
                pipeline="AdvTrain_eval_ddp",
                domain_from=args.auxiliary,
                domain_to=args.target,
                start_time=_eval_start_str,
                eval_elapsed=_eval_elapsed,
            )
            ev_logger.info("DONE.")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _dispatch_eval(args):
    if not _eval_torchrun_env():
        raise RuntimeError(_EVAL_REQUIRES_TORCHRUN_MSG)
    if not torch.cuda.is_available():
        raise RuntimeError("检测到分布式启动环境但无 CUDA，无法使用 eval DDP（需要 CUDA + NCCL）。")
    _run_eval_ddp(args)


def _add_train_args(p):
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志路径；默认 logs/task{idx}_时间戳.log；目录由 D4C_LOG_DIR 指定",
    )
    p.add_argument("--device", type=int, default=0, help="DDP 下以 LOCAL_RANK 为准，可忽略")
    p.add_argument("--auxiliary", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率；不传则由 build_resolved_training_config 按 BASE→TASK→预设→ENV→CLI 解析",
    )
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None, help="不传则用 config.epochs")
    p.add_argument(
        "--coef",
        type=float,
        default=None,
        help="不传则由 resolve：TASK_DEFAULTS / 命名预设 / ENV / CLI 覆盖链决定",
    )
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--adv",
        type=float,
        default=None,
        help="对抗系数；不传则由 resolve（task / preset / CLI）决定",
    )
    p.add_argument(
        "--adversarial-start-epoch",
        type=int,
        default=None,
        help="前 N 个 epoch（1..N）不启用对抗；自第 N+1 个 epoch 起进入 warmup。不设则全程 --adv（旧行为）。D4C_ADVERSARIAL_START_EPOCH",
    )
    p.add_argument(
        "--adversarial-warmup-epochs",
        type=int,
        default=None,
        help="从第 N+1 个 epoch 起，对抗系数由 0 线性升至目标的持续 epoch 数；0=第 N+1 轮即满幅。D4C_ADVERSARIAL_WARMUP_EPOCHS",
    )
    p.add_argument(
        "--adversarial-coef-target",
        type=float,
        default=None,
        help="对抗系数目标；默认等于 --adv。D4C_ADVERSARIAL_COEF_TARGET",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="训练全局有效 batch（每优化步跨所有 rank 的样本总数）",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="梯度累积步数；默认 config / D4C_GRADIENT_ACCUMULATION_STEPS；满足 G=P×world_size×A",
    )
    p.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="单卡 DataLoader 微批；显存不足时减小并配合全局 G 推出 accum；或 D4C_PER_DEVICE_BATCH_SIZE",
    )
    p.add_argument("--num-proc", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="快速验证用：最多训练到 N 个 batch 就退出")
    p.add_argument(
        "--save-final-checkpoint",
        action="store_true",
        help="训练进程退出时（含 --max-steps 提前结束）由 rank0 无条件写入 save_file；"
        "不改变 loss/BLEU 选模逻辑。适用于 smoke / debug；正式训练默认关闭。",
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="早停生效前最少训练的 epoch 数；默认 TRAIN_MIN_EPOCHS / config",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="验证损失相对上一轮变差的连续次数上限（改进时会清零）；默认 TRAIN_EARLY_STOP_PATIENCE",
    )
    p.add_argument(
        "--early-stop-patience-full",
        type=int,
        default=None,
        help="dual_bleu 时：连续多少次 full BLEU eval 未刷新 best 则早停（quick 不参与）；"
        "默认 TRAIN_EARLY_STOP_PATIENCE_FULL 或与 --early-stop-patience 相同",
    )
    p.add_argument(
        "--early-stop-patience-loss",
        type=int,
        default=None,
        help="dual_bleu：valid_loss 连续变差早停次数，与 patience_full 独立；"
        "默认 TRAIN_EARLY_STOP_PATIENCE_LOSS 或同 --early-stop-patience",
    )
    p.add_argument(
        "--checkpoint-metric",
        type=str,
        choices=["loss", "bleu4"],
        default="bleu4",
        help="保存 checkpoint 的依据：valid loss 下降 或 验证集 BLEU-4（子集，与论文选模一致）",
    )
    p.add_argument(
        "--bleu4-max-samples",
        type=int,
        default=None,
        help="按 BLEU-4 选模时验证集最多采样条数；默认 TRAIN_BLEU4_MAX_SAMPLES",
    )
    p.add_argument(
        "--scheduler-initial-lr",
        type=float,
        default=None,
        help="优化器初始 LR；若设置则覆盖 --learning_rate（均在 resolve 内处理）",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="warmup 步数；等价 D4C_WARMUP_STEPS",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="warmup 占计划总步数比例；等价 D4C_WARMUP_RATIO",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="cosine 末端 LR / initial_lr；等价 D4C_MIN_LR_RATIO",
    )
    p.add_argument(
        "--quick-eval-max-samples",
        type=int,
        default=None,
        help="每 epoch quick BLEU 子集；等价 D4C_QUICK_EVAL_MAX_SAMPLES",
    )
    p.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DDP 是否检测未参与 loss 的参数；默认真（避免多分支/对抗训练报错），可用 --no-ddp-find-unused-parameters 换略快训练",
    )


def _add_eval_args(p):
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志路径；默认 logs/task{idx}_时间戳.log",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--auxiliary", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-proc", type=int, default=None)


