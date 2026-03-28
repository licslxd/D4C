import os
import sys
import time
import hashlib
import logging
# 离线模式：禁止从网络加载
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_utils import *
from paths_config import DATA_DIR, MERGED_DATA_DIR, T5_SMALL_DIR, get_checkpoint_task_dir
from config import (
    FinalTrainingConfig,
    apply_ddp_fast_torch_backends,
    build_resolved_training_config,
    hf_datasets_progress_bar,
    resolve_task_idx_from_aux_target,
    should_run_full_bleu_eval_epoch,
)
from training_runtime_inputs import collect_training_runtime_overrides_from_args
from lr_schedule_utils import resolve_warmup_steps, warmup_cosine_multiplier_lambda
from bleu_valid_ddp import bleu4_explanation_full_valid_ddp
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
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from transformers import T5Tokenizer
from torch import nn, optim
from torch.nn.modules.transformer import _get_activation_fn
import argparse
import contextlib
import math
from dataclasses import replace
from datetime import datetime
from typing import Optional, Tuple
from tqdm import tqdm
from torch.optim import lr_scheduler as lr_sched
from datasets import Dataset, DatasetDict, load_from_disk
from perf_monitor import PerfMonitor, gather_ddp_gpu_stats_for_epoch_log
import numpy as np
import copy
import torch.nn.functional as F
from train_logging import (
    create_run_paths,
    setup_train_logging,
    log_run_snapshot,
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
    check_finite_loss,
    check_finite_tensor,
    collect_distributed_env_for_meta,
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
    warn_empty_batch,
)

# 离线加载：优先使用本地 T5 目录
_t5_path = T5_SMALL_DIR if os.path.exists(T5_SMALL_DIR) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)

# HuggingFace tokenize 磁盘缓存：与 AdvTrain.D4C_TOKENIZE_CACHE_VERSION 保持同步
D4C_TOKENIZE_CACHE_VERSION = "v2"

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
            domain_val = 0  # Auxiliary domain
        elif sample["domain"] == "target":
            domain_val = 1  # Target domain
        else:
            raise ValueError("Unknown domain!")

        domain_idx = torch.tensor(domain_val, dtype=torch.long)
        return {"user_idx": user_idx, "item_idx": item_idx, "rating": raitng, "explanation_idx": explanation_idx, "domain_idx": domain_idx}

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


class Model(nn.Module):
    def __init__(self, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, user_profiles, item_profiles, domain_profiles):
        super().__init__()
        self.domain_profiles = nn.Parameter(domain_profiles)
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.user_profiles = nn.Parameter(user_profiles)  # user_profiles
        self.item_profiles = nn.Parameter(item_profiles)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.recommender = PETER_MLP(emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one 
        self.transformer_encoder = CustomTransformerEncoder(encoder_layers, nlayers)   # loop over the one above
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.emsize = emsize
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
        word_feature = self.word_embeddings(tgt_input)   # in shape (N,seqlen, emsize)
        src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        
        # peter mask and pad mask
        attn_mask = generate_domain_mask(tgt_input.shape[1], device)
        hidden, _ = self.transformer_encoder(src=src, mask=attn_mask)
        rating = self.recommender(hidden[:,3])
        context_dist = self.hidden2token(hidden[:,4]).unsqueeze(1).repeat(1, tgt_input.shape[1], 1) 
        word_dist = self.hidden2token(hidden[:,5:])
        return rating, context_dist, word_dist  # (N), (N,seqlen,emsize), (N,seqlen,emsize) respectively
    
    def gather(self, batch, device):
        user_idx, item_idx, rating, tgt_output, domain_idx = batch
        # 配合 DataLoader(pin_memory=True) 使用 non_blocking=True，减少同步拷贝等待
        user_idx = user_idx.to(device, non_blocking=True)
        item_idx = item_idx.to(device, non_blocking=True)
        domain_idx = domain_idx.to(device, non_blocking=True)
        rating = rating.to(device, non_blocking=True).float()
        tgt_output = tgt_output.to(device, non_blocking=True)
        tgt_input = T5_shift_right(tgt_output)
        return user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx

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
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)   # in shape (N,1)
        for i in range(max_len):
            word_feature = self.word_embeddings(decoder_input_ids) 
            src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)  # in shape: (N, 2+1, emsize)
            attn_mask = generate_domain_mask(decoder_input_ids.shape[1], device)
            hidden, attention_scores = self.transformer_encoder(src=src, mask=attn_mask)     # in shape (N, 3, emsize)
            dist = self.hidden2token(hidden).softmax(dim=-1)
            output_id = dist[:,-1,:].topk(1).indices                       # in shape (N, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
            entropies = compute_entropy(dist)
            total_entropies.append(entropies)
        total_entropies = torch.stack(total_entropies).mean(dim=0)
        return decoder_input_ids[:,1:], total_entropies, attention_scores # removing <BOS>


def _valid_bleu4_quick_d4c(
    model,
    valid_dataset,
    device,
    max_samples: int,
    *,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: Optional[int],
) -> float:
    """验证集前 max_samples 条上算 BLEU-4，仅 rank0 调用。"""
    _m = get_underlying_model(model)
    n = min(len(valid_dataset), max_samples)
    if n <= 0:
        return 0.0
    subset = Subset(valid_dataset, list(range(n)))
    bs = min(32, n)
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
    pred_texts = []
    ref_texts = []
    _m.eval()
    with torch.inference_mode():
        for batch in dl:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _m.gather(batch, device)
            gen_ids, *_ = _m.generate(user_idx, item_idx, domain_idx)
            pred_texts.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            ref_texts.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
    scores = compute_bleu1234_only(pred_texts, ref_texts)
    return float(scores.get("4", 0.0))


@contextlib.contextmanager
def _ddp_no_sync_model(model, world_size: int, sync_gradients: bool):
    """梯度累积非边界微批上使用 DDP no_sync。"""
    if world_size <= 1 or sync_gradients:
        yield
    else:
        with model.no_sync():
            yield


def _d4c_step5_bf16_autocast_enabled() -> bool:
    """D4C_BF16=1 且当前 CUDA 设备支持 bf16 时启用训练态 autocast。"""
    if os.environ.get("D4C_BF16", "0") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_bf16_supported())


def trainModel_ddp(
    model,
    train_dataloader,
    valid_dataloader,
    sampler,
    valid_sampler,
    final_cfg: FinalTrainingConfig,
    rank,
    world_size,
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
    use_bf16 = _d4c_step5_bf16_autocast_enabled()
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
    full_eval_every = int(final_cfg.full_eval_every_epochs)
    full_eval_phased = final_cfg.full_eval_phased
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
            _bf16_msg = "[Train] bf16 autocast: ON (D4C_BF16=1, CUDA bf16 supported)"
        elif os.environ.get("D4C_BF16", "0") != "1":
            _bf16_msg = "[Train] bf16 autocast: OFF (D4C_BF16 is not 1)"
        elif not torch.cuda.is_available():
            _bf16_msg = "[Train] bf16 autocast: OFF (CUDA not available)"
        else:
            _bf16_msg = "[Train] bf16 autocast: OFF (torch.cuda.is_bf16_supported() is False)"
        if _lg:
            _lg.info(_bf16_msg, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_bf16_msg, flush=True)
        log_bf16_amp_note(_lg, use_bf16, has_grad_scaler=False)
        _sched = (
            f"phased{full_eval_phased}"
            if full_eval_phased is not None
            else f"uniform_interval={full_eval_every}"
        )
        _es = (
            f"Early stop: min_epochs={min_epochs}, patience={early_stop_patience} (非 dual_bleu 时 valid 变差), "
            f"early_stop_patience_full={early_stop_patience_full} (dual_bleu: full BLEU 未刷新 best), "
            f"early_stop_patience_loss={early_stop_patience_loss} (dual_bleu: valid_loss 连续变差), "
            f"checkpoint_metric={checkpoint_metric}, quick_eval_max_samples={quick_eval_max_samples}, "
            f"full_bleu_schedule={_sched}, dual_bleu_eval={dual_bleu}"
        )
        if _lg:
            _lg.info(_es, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_es, flush=True)
        if _lg:
            _lg.info(
                "Train profile: lr_scheduler=%s warmup_epochs=%g full_bleu_schedule=%s",
                lr_scheduler_type,
                warmup_epochs,
                _sched,
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
    grad_iv = max(1, d4c_log_grad_interval())
    micro_step_count = 0
    try:
        for epoch in range(epochs):
            epoch_1 = epoch + 1
            sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            if rank == 0:
                perf.epoch_start()
            model.train()
            loss_sum = torch.zeros((), dtype=torch.double, device=device)
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
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
                bsz = int(user_idx.size(0))
                warn_empty_batch(_lg, global_step=global_step, epoch=epoch_1, bsz=bsz)
                with sync_ctx:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input, domain_idx)
                        factual_idx = (domain_idx == 1).squeeze()
                        counterfactual_idx = (domain_idx == 0).squeeze()
                        if factual_idx.sum() > 0:
                            loss_r_factual = _model.rating_loss_fn(pred_rating[factual_idx], rating[factual_idx])
                            loss_e_factual = _model.exp_loss_fn(
                                word_dist[factual_idx].view(-1, 32128), tgt_output[factual_idx].reshape(-1)
                            )
                            loss_c_factual = _model.exp_loss_fn(
                                context_dist[factual_idx].view(-1, 32128), tgt_output[factual_idx].reshape(-1)
                            )
                            loss_factual = coef * loss_r_factual + coef * loss_c_factual + loss_e_factual
                        else:
                            loss_factual = torch.zeros((), device=device)
                        if counterfactual_idx.sum() > 0:
                            loss_r_counterfactual = _model.rating_loss_fn(
                                pred_rating[counterfactual_idx], rating[counterfactual_idx]
                            )
                            loss_e_counterfactual = _model.exp_loss_fn(
                                word_dist[counterfactual_idx].view(-1, 32128),
                                tgt_output[counterfactual_idx].reshape(-1),
                            )
                            loss_c_counterfactual = _model.exp_loss_fn(
                                context_dist[counterfactual_idx].view(-1, 32128),
                                tgt_output[counterfactual_idx].reshape(-1),
                            )
                            loss_counterfactual = eta * (
                                coef * loss_r_counterfactual + coef * loss_c_counterfactual + loss_e_counterfactual
                            )
                        else:
                            loss_counterfactual = torch.zeros((), device=device)
                        loss = loss_factual + loss_counterfactual
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
                        if d4c_log_step_loss_parts():
                            _extra = {
                                "loss_factual": float(loss_factual.detach().item()),
                                "loss_counterfactual": float(loss_counterfactual.detach().item()),
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
                if micro_step_count % step_iv == 0 and rank == 0:
                    check_finite_loss(loss, _lg, global_step=global_step, epoch=epoch_1)
                    check_finite_tensor(word_dist, _lg, name="word_dist", global_step=global_step, epoch=epoch_1)
            ddp_heartbeat(_lg, "before_train_loss_allreduce", rank=rank, epoch=epoch_1)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
            ddp_heartbeat(_lg, "after_train_loss_allreduce", rank=rank, epoch=epoch_1)
            avg_loss = (loss_sum / n_samples).item()
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
            v_loss_sum, v_n = validModel(model, valid_dataloader, device)
            if rank == 0 and _lg:
                _lg.info(
                    "[Timing] valid_loss_forward end epoch=%d elapsed_s=%.3f",
                    epoch_1,
                    time.perf_counter() - _t_valid0,
                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                )
            v_stat = torch.tensor([v_loss_sum, float(v_n)], dtype=torch.double, device=device)
            dist.all_reduce(v_stat, op=dist.ReduceOp.SUM)
            current_valid_loss = float(v_stat[0] / v_stat[1]) if v_stat[1] > 0 else 0.0
            bleu4_score_this_epoch = None
            quick_bleu4 = None
            full_bleu4_val = None
            is_full_eval_epoch = (
                checkpoint_metric == "bleu4"
                and valid_dataset_for_bleu is not None
                and should_run_full_bleu_eval_epoch(epoch + 1, full_eval_every, full_eval_phased)
            )
            if checkpoint_metric == "bleu4" and valid_dataset_for_bleu is not None:
                if rank == 0:
                    with d4c_timing_phase(
                        _lg,
                        f"bleu_quick_epoch_{epoch_1}",
                        route=ROUTE_SUMMARY,
                        rank=0,
                    ):
                        quick_bleu4 = _valid_bleu4_quick_d4c(
                            model,
                            valid_dataset_for_bleu,
                            device,
                            quick_eval_max_samples,
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
                        ckpt_src = (
                            "full_bleu4"
                            if is_full_eval_epoch
                            else "trend_quick_bleu4"
                        )
                        bleu_line = (
                            f"quick_bleu4={quick_bleu4:.4f} | full_bleu4={fstr} | "
                            f"checkpoint_metric_source={ckpt_src}"
                        )
                    else:
                        bleu_line = (
                            f"quick_bleu4={quick_bleu4:.4f} | full_bleu4="
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
            if dual_bleu and checkpoint_metric == "bleu4":
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
                            reason="dual_bleu_full_bleu_improved",
                            logger=_lg,
                        )
                else:
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
                    f"early_stop_dual: current_full_bleu4={_cur_full} "
                    f"best_full_bleu4={best_full_bleu4:.4f} "
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


def validModel(model, valid_dataloader, device):
    _model = get_underlying_model(model)
    # forward 必须用底层 _model，避免 DDP 包装后的 model(...) 在局部执行时触发额外 NCCL collective。
    _model.eval()
    loss_sum = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
            bsz = int(user_idx.size(0))
            pred_rating, context_dist, word_dist = _model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss = loss_r + loss_e
            loss_sum += float(loss.detach().item()) * bsz
            n_samples += bsz
    return loss_sum, n_samples


def evalModel(model, test_dataloader, device):
    _model = get_underlying_model(model)
    model = model.to(device)
    model.eval()
    prediction_ratings = []
    ground_truth_ratings = []
    prediction_exps = []
    reference_exps = []
    with torch.no_grad():
        for batch in test_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
            pred_ratings = model.recommend(user_idx, item_idx, domain_idx)
            # generate() 当前返回 (pred_exps, entropies, attention_scores)；
            # 评估阶段只需生成文本，兼容后续返回项扩展。
            pred_exps, *_ = model.generate(user_idx, item_idx,  domain_idx)
            prediction_ratings.extend(pred_ratings.tolist())
            ground_truth_ratings.extend(rating.tolist())
            prediction_exps.extend(tokenizer.batch_decode(pred_exps, skip_special_tokens=True))
            reference_exps.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))

    prediction_ratings  = np.array(prediction_ratings)
    ground_truth_ratings = np.array(ground_truth_ratings)
    return {
        "recommendation": {"_pred": prediction_ratings, "_gt": ground_truth_ratings},
        "explanation": {"_pred_exps": prediction_exps, "_ref_exps": reference_exps},
    }


def _load_profile_tensors(auxiliary, target, device_idx):
    sdomain = torch.tensor(
        np.load(os.path.join(DATA_DIR, auxiliary, "domain.npy")), dtype=torch.float, device=device_idx,
    )
    tdomain = torch.tensor(
        np.load(os.path.join(DATA_DIR, target, "domain.npy")), dtype=torch.float, device=device_idx,
    )
    domain_profiles = torch.cat([sdomain.unsqueeze(0), tdomain.unsqueeze(0)], dim=0)
    user_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, target, "user_profiles.npy")), dtype=torch.float, device=device_idx,
    )
    item_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, target, "item_profiles.npy")), dtype=torch.float, device=device_idx,
    )
    return domain_profiles, user_profiles, item_profiles


def _make_model(final_cfg: FinalTrainingConfig, args, device_idx):
    domain_profiles, user_profiles, item_profiles = _load_profile_tensors(args.auxiliary, args.target, device_idx)
    return Model(
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
    ).to(device_idx)


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


def _build_step5_cache_dir(
    ckpt_task_dir: str,
    train_path: str,
    valid_path: str,
    processor,
    tok,
    *,
    eval_only: bool,
    cache_version: str = D4C_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str]:
    fp = _build_tokenize_cache_fingerprint(
        train_path=train_path,
        valid_path=valid_path,
        tok=tok,
        max_length=int(getattr(processor, "max_length", 25)),
        cache_version=cache_version,
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


def build_d4c_ddp_artefacts(
    args, world_size, local_rank, rank, eval_only=False, *, show_datasets_progress: bool = True,
):
    task_idx = resolve_task_idx_from_aux_target(args.auxiliary, args.target)
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")
    _ro = collect_training_runtime_overrides_from_args(args)
    resolved = build_resolved_training_config(
        args,
        task_idx=task_idx,
        world_size=world_size,
        runtime_overrides=_ro,
    )
    path = os.path.join(DATA_DIR, args.target)
    _ckpt_task = get_checkpoint_task_dir(task_idx)
    os.makedirs(_ckpt_task, exist_ok=True)
    train_path = os.path.join(_ckpt_task, "factuals_counterfactuals.csv")
    valid_path = os.path.join(path, "valid.csv")
    train_df = pd.read_csv(train_path)
    nuser = int(train_df["user_idx"].max()) + 1
    nitem = int(train_df["item_idx"].max()) + 1
    save_file = args.save_file or os.path.join(_ckpt_task, "model.pth")
    nproc = int(resolved.num_proc)
    base_final = replace(
        resolved,
        nuser=nuser,
        nitem=nitem,
        save_file=save_file,
        device=local_rank,
        device_ids=tuple(range(world_size)),
        ddp_world_size=world_size,
        nlayers=args.nlayers,
    )
    valid_df = pd.read_csv(valid_path)
    valid_df["domain"] = "target"
    processor = Processor(args.auxiliary, args.target)
    cache_dir, cache_fp = _build_step5_cache_dir(
        _ckpt_task,
        train_path,
        valid_path,
        processor,
        tokenizer,
        eval_only=eval_only,
    )
    if rank == 0:
        _log_step5_tokenize_line(
            f"[Tokenize] step5 cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
        )
    if eval_only:
        datasets = DatasetDict({"valid": Dataset.from_pandas(valid_df)})
        _st5 = logging.getLogger(LOGGER_NAME)
        with d4c_timing_phase(
            _st5,
            "tokenize_pipeline_step5_valid",
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
                phase="valid",
            )
        encoded_data.set_format("torch")
        train_dataset = None
    else:
        train_df = train_df[train_df["explanation"].notna()]
        datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "valid": Dataset.from_pandas(valid_df),
        })
        _st5 = logging.getLogger(LOGGER_NAME)
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
        encoded_data.set_format("torch")
        train_dataset = TensorDataset(
            encoded_data["train"]["user_idx"],
            encoded_data["train"]["item_idx"],
            encoded_data["train"]["rating"],
            encoded_data["train"]["explanation_idx"],
            encoded_data["train"]["domain_idx"],
        )
    valid_dataset = TensorDataset(
        encoded_data["valid"]["user_idx"],
        encoded_data["valid"]["item_idx"],
        encoded_data["valid"]["rating"],
        encoded_data["valid"]["explanation_idx"],
        encoded_data["valid"]["domain_idx"],
    )
    model = _make_model(base_final, args, local_rank)
    return base_final, train_dataset, valid_dataset, model


def _run_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("run-d4c.py 仅支持 CUDA + NCCL DDP。")
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

    gb = getattr(args, "global_batch_size", None)
    if gb is not None:
        if args.batch_size is not None and args.batch_size != gb:
            raise ValueError(
                f"--batch-size ({args.batch_size}) 与 --global-batch-size ({gb}) 冲突，请只指定其一或保持一致。"
            )
        args.batch_size = gb

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
    try:
        base_final, train_dataset, valid_dataset, model = build_d4c_ddp_artefacts(
            args,
            world_size,
            local_rank,
            rank,
            eval_only=args.eval_only,
            show_datasets_progress=(rank == 0),
        )
        final_cfg = replace(
            base_final,
            run_id=run_id,
            logger=train_logger,
            log_file=log_path,
            valid_dataset=valid_dataset,
            ddp_fast_backends=ddp_fast_backends,
            ddp_find_unused_parameters=True,
            rank0_only_logging=True,
        )
        pin_memory = torch.cuda.is_available()
        _G = int(final_cfg.train_batch_size)
        valid_per_rank = max(1, _G // world_size)
        if rank == 0:
            log_run_snapshot(
                train_logger,
                {
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                    "eval_only": args.eval_only,
                    "train_only": args.train_only,
                    "rank": rank,
                    "world_size": world_size,
                    "local_rank": local_rank,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "batch_size": _G,
                    "batch_size_global": final_cfg.batch_size_global,
                    "per_device_batch_size": final_cfg.per_device_train_batch_size,
                    "gradient_accumulation_steps": final_cfg.gradient_accumulation_steps,
                    "effective_global_batch_size": final_cfg.effective_global_batch_size,
                    "distributed_env": collect_distributed_env_for_meta(),
                },
                final_cfg.to_log_dict(),
            )
            flush_preset_load_events(train_logger)
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
        )
        if not args.eval_only:
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
                )
            except Exception as exc:
                if rank == 0:
                    log_training_crash(train_logger, exc)
                raise
        if args.eval_only and not os.path.isfile(final_cfg.save_file):
            raise FileNotFoundError(
                f"--eval-only 需要已有权重文件，未找到: {final_cfg.save_file}\n"
                "请设置 D4C_CHECKPOINT_GROUP / D4C_CHECKPOINT_SUBDIR 指向训练产物目录，或使用 --save-file。"
            )
        dist.barrier()
        run_final_eval = args.eval_only or not args.train_only
        if run_final_eval:
            import time as _time

            _eval_t0 = _time.time()
            _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 所有 rank 都加载模型并分片推理，rank 0 聚合后计算指标
            eval_model = _make_model(final_cfg, args, local_rank)
            eval_model.load_state_dict(
                torch.load(
                    final_cfg.save_file,
                    map_location=f"cuda:{local_rank}",
                    weights_only=True,
                ),
            )
            eval_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            _real_valid_len = len(valid_dataset)
            _eval_global_bs = int(final_cfg.eval_batch_size)
            _eval_per_gpu = max(1, _eval_global_bs // world_size)
            _eval_nw = int(final_cfg.dataloader_num_workers_valid)
            _eval_pf = final_cfg.dataloader_prefetch_factor_valid
            eval_dataloader = DataLoader(
                valid_dataset,
                batch_size=_eval_per_gpu,
                sampler=eval_sampler,
                shuffle=False,
                num_workers=_eval_nw,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=_eval_nw > 0,
                prefetch_factor=_eval_pf,
            )
            local_final = evalModel(eval_model, eval_dataloader, local_rank)
            # gather 各 rank 的原始结果到 rank 0
            _local_obj = {
                "pred_rat": local_final["recommendation"]["_pred"],
                "gt_rat": local_final["recommendation"]["_gt"],
                "pred_exps": local_final["explanation"]["_pred_exps"],
                "ref_exps": local_final["explanation"]["_ref_exps"],
            }
            _gathered = [None] * world_size
            dist.all_gather_object(_gathered, _local_obj)
            if rank == 0:
                all_pred = np.concatenate([g["pred_rat"] for g in _gathered])[:_real_valid_len]
                all_gt = np.concatenate([g["gt_rat"] for g in _gathered])[:_real_valid_len]
                all_pred_exps = sum((g["pred_exps"] for g in _gathered), [])[:_real_valid_len]
                all_ref_exps = sum((g["ref_exps"] for g in _gathered), [])[:_real_valid_len]
                diffs = all_pred - all_gt
                mae = round(np.mean(np.abs(diffs)), 4)
                rmse = round(np.sqrt(np.mean(np.square(diffs))), 4)
                text_results = evaluate_text(all_pred_exps, all_ref_exps)
                final = {"recommendation": {"mae": mae, "rmse": rmse}, "explanation": text_results}
                _task_desc = (
                    f"Step 5 DDP Task {task_idx} 仅 eval (nproc={world_size}): {args.auxiliary} -> {args.target}"
                    if args.eval_only
                    else f"Step 5 DDP Task {task_idx} 训练 (nproc={world_size}): {args.auxiliary} -> {args.target}"
                )
                _eval_elapsed = _time.time() - _eval_t0
                _eval_min, _eval_sec = divmod(int(_eval_elapsed), 60)
                _lines = format_final_results_lines(final, task_description=_task_desc, start_time=_eval_start_str)
                _lines.append(f"Eval elapsed: {_eval_min}m {_eval_sec}s ({_eval_elapsed:.1f}s)")
                log_final_results_block(train_logger, _lines)
                finalize_run_log(train_logger)
                append_eval_run_summaries(
                    final,
                    task_idx=int(final_cfg.task_idx),
                    run_id=str(run_id or ""),
                    pipeline="run_d4c_eval_only" if args.eval_only else "run_d4c_train_eval",
                    domain_from=args.auxiliary,
                    domain_to=args.target,
                    log_file=log_path,
                    save_file=final_cfg.save_file,
                    task_description=_task_desc,
                    start_time=_eval_start_str,
                    eval_elapsed=_eval_elapsed,
                )
                train_logger.info("DONE.")
        elif rank == 0:
            _sf = final_cfg.save_file
            train_logger.info("DONE（--train-only：已跳过训练后 valid 评估；权重: %s）。", _sf)
            finalize_run_log(train_logger)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _reject_legacy_gpus_argv(argv):
    for arg in argv[1:]:
        if arg == "--gpus" or arg.startswith("--gpus="):
            sys.stderr.write(
                "run-d4c.py: error: --gpus has been removed.\n"
                "Use CUDA_VISIBLE_DEVICES and torchrun --standalone --nproc_per_node=<K> run-d4c.py ...\n"
                "Single-GPU DDP smoke: torchrun --standalone --nproc_per_node=1 run-d4c.py ...\n"
            )
            raise SystemExit(2)


def main():
    _reject_legacy_gpus_argv(sys.argv)
    parser = argparse.ArgumentParser(description="Step 5 主训练：须用 torchrun 启动（DistributedDataParallel）")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件；默认 logs/task{idx}_时间戳.log；D4C_LOG_DIR 指定目录",
    )
    parser.add_argument("--auxiliary", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率；不传则由 build_resolved_training_config 解析",
    )
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数，不传则用 config.epochs")
    parser.add_argument(
        "--coef",
        type=float,
        default=None,
        help="不传则由 build_resolved_training_config 从 TASK_DEFAULTS / 预设 / CLI 解析",
    )
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="不传则 resolve 默认 1e-3；显式传入视为 CLI 覆盖",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="全局有效 batch（每优化步跨所有 rank）；与 --global-batch-size 同义择一",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="同 --batch-size，强调全局语义",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="梯度累积步数；config / D4C_GRADIENT_ACCUMULATION_STEPS；满足 G=P×world_size×A",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="单卡训练微批；显存不足时减小；或 D4C_PER_DEVICE_BATCH_SIZE",
    )
    parser.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数，不传则用 config.num_proc")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="跳过训练，仅加载 checkpoint 并在 valid 上输出 FINAL RESULTS（须已有 model.pth）",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="训练结束后跳过 valid 上的收尾评估（与 --eval-only 互斥；训练中仍会按 epoch 做 valid）",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="早停生效前最少 epoch；默认 TRAIN_MIN_EPOCHS",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="valid 连续变差次数上限（改进时清零）；默认 TRAIN_EARLY_STOP_PATIENCE",
    )
    parser.add_argument(
        "--early-stop-patience-full",
        type=int,
        default=None,
        help="dual_bleu：连续多少次 full BLEU eval 未刷新 best 则早停（quick 不参与）；"
        "默认 TRAIN_EARLY_STOP_PATIENCE_FULL 或与 --early-stop-patience 相同",
    )
    parser.add_argument(
        "--early-stop-patience-loss",
        type=int,
        default=None,
        help="dual_bleu：valid_loss 连续变差早停次数，与 full BLEU 的 patience_full 独立；"
        "默认 TRAIN_EARLY_STOP_PATIENCE_LOSS 或同 --early-stop-patience",
    )
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        choices=["loss", "bleu4"],
        default="bleu4",
        help="checkpoint 依据：valid loss 或验证集 BLEU-4（子集）",
    )
    parser.add_argument(
        "--bleu4-max-samples",
        type=int,
        default=None,
        help="TRAIN_BLEU4_MAX_SAMPLES 默认链；亦作 quick 子集回退上限",
    )
    parser.add_argument(
        "--quick-eval-max-samples",
        type=int,
        default=None,
        help="每 epoch quick BLEU-4 子集大小；等价 D4C_QUICK_EVAL_MAX_SAMPLES",
    )
    parser.add_argument(
        "--full-eval-every",
        type=int,
        default=None,
        help="每 N epoch 做 full valid BLEU；不设且未 export D4C_FULL_EVAL_EVERY 时用分阶段默认（build_resolved_training_config）",
    )
    parser.add_argument(
        "--scheduler-initial-lr",
        type=float,
        default=None,
        help="优化器初始学习率；若设置则覆盖 --learning_rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="warmup 步数（优先于 ratio）；等价 D4C_WARMUP_STEPS",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="warmup 占计划总步数比例；等价 D4C_WARMUP_RATIO",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="cosine 末端 LR 相对 initial_lr 的比例；等价 D4C_MIN_LR_RATIO",
    )
    args = parser.parse_args()
    if args.eval_only and args.train_only:
        parser.error("--eval-only 与 --train-only 不能同时使用")
    if "RANK" not in os.environ:
        print(
            "错误: 请使用 torchrun 启动 Step 5，例如:\n"
            "  torchrun --standalone --nproc_per_node=2 run-d4c.py --auxiliary A --target B ...\n"
            "单卡: DDP_NPROC=1 bash sh/run_step5_optimized.sh --task N --step3-subdir step3_opt_<时间戳>",
            file=sys.stderr,
        )
        sys.exit(1)
    _run_ddp(args)


if __name__ == "__main__":
    main()