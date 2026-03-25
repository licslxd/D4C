# -*- coding: utf-8 -*-
"""
域对抗预训练：模型与数据管线、DDP 训练、评估 CLI 合一。
- 库用法：from AdvTrain import *（与 generate_counterfactual 等兼容）
- 训练：torchrun ... AdvTrain.py train --auxiliary A --target B ...
- 评估：python AdvTrain.py eval ...（单进程；可选 --gpus 做 DataParallel）
        或 torchrun ... AdvTrain.py eval ...（多进程 DDP，与训练同 DDP_NPROC）
"""
import os
import sys
import copy
import time
import warnings
import argparse
import contextlib
import logging
import math
from datetime import datetime
from typing import Optional

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
from paths_config import T5_SMALL_DIR, DATA_DIR, MERGED_DATA_DIR, get_checkpoint_task_dir
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
from datasets import Dataset, DatasetDict
from config import (
    get_train_batch_size,
    get_eval_batch_size,
    get_epochs,
    resolve_task_idx_from_aux_target,
    resolve_ddp_train_microbatch_layout,
    get_per_device_train_batch_size_optional,
    get_num_proc,
    get_dataloader_num_workers,
    get_dataloader_prefetch_factor,
    get_ddp_train_num_workers_per_rank,
    hf_datasets_progress_bar,
    get_train_min_epochs,
    get_train_early_stop_patience,
    get_train_early_stop_patience_full,
    resolve_early_stop_patience_loss,
    get_train_bleu4_max_samples,
    get_train_mode,
    get_exact_reproduction,
    get_allow_large_batch,
    get_lr_scheduler_type,
    get_warmup_epochs,
    resolve_full_bleu_eval_training,
    should_run_full_bleu_eval_epoch,
    get_quick_eval_max_samples,
    apply_optimized_torch_backends,
    get_min_lr_ratio,
    get_scheduler_initial_lr,
    get_d4c_warmup_steps_optional,
    get_d4c_warmup_ratio_optional,
)
from lr_schedule_utils import resolve_warmup_steps, warmup_cosine_multiplier_lambda
from bleu_valid_ddp import bleu4_explanation_full_valid_ddp
from train_logging import (
    create_run_paths,
    setup_train_logging,
    log_run_header,
    log_config_snapshot,
    format_epoch_training_block,
    log_epoch_training_block,
    broadcast_run_paths_ddp,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    append_eval_run_summaries,
    LOGGER_NAME,
    logger_has_file_handler,
)

_t5_path = T5_SMALL_DIR if os.path.exists(T5_SMALL_DIR) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)

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
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
            pred_rating, _, word_dist, _, _, = model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss = loss_r + loss_e
            avg_loss += loss.item()
        avg_loss /= len(valid_dataloader)
        return avg_loss


def validModel_sum_batches(model, valid_dataloader, device):
    """
    DDP 训练时用于验证聚合：
    - 返回 (loss_sum, num_batches)，在外面用 all_reduce 聚合再除以总 batches。
    - 与原 validModel 的“按 batch 求平均”保持一致，避免语义漂移。
    """
    _model = get_underlying_model(model)
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        n_batches = 0
        for batch in valid_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
            pred_rating, _, word_dist, _, _, = model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss = loss_r + loss_e
            loss_sum += loss.item()
            n_batches += 1
        return loss_sum, n_batches


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


def _load_advtrain_artefacts(
    args,
    device: int,
    batch_size: int,
    *,
    log_tokenize: bool = True,
    show_datasets_progress: bool = True,
):
    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    path = os.path.join(MERGED_DATA_DIR, str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    nuser = train_df["user_idx"].max() + 1
    nitem = train_df["item_idx"].max() + 1

    os.makedirs(get_checkpoint_task_dir(task_idx), exist_ok=True)
    epochs = args.epochs if args.epochs is not None else get_epochs(task_idx)
    nproc = args.num_proc if args.num_proc is not None else get_num_proc()

    config = {
        "task_idx": task_idx,
        "device": device,
        "device_ids": None,
        "log_file": args.log_file,
        "save_file": args.save_file or os.path.join(get_checkpoint_task_dir(task_idx), "model.pth"),
        "learning_rate": args.learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_proc": nproc,
        "emsize": 768,
        "nlayers": args.nlayers,
        "nhid": 2048,
        "ntoken": 32128,
        "dropout": 0.2,
        "nuser": nuser,
        "nitem": nitem,
        "coef": args.coef,
        "adversarial_coef": args.adv,
        "nhead": 2,
    }

    valid_df = pd.read_csv(os.path.join(path, "aug_valid.csv"))
    train_df["item"] = train_df["item"].astype(str)
    valid_df["item"] = valid_df["item"].astype(str)
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "valid": Dataset.from_pandas(valid_df),
    })
    processor = Processor(args.auxiliary, args.target)
    t0 = time.perf_counter()
    with hf_datasets_progress_bar(show_datasets_progress):
        encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
    if log_tokenize:
        _log_tokenize_done(
            "train+valid",
            nproc,
            time.perf_counter() - t0,
            getattr(args, "log_file", None),
        )
    encoded_data.set_format("torch")
    train_dataset = TensorDataset(
        encoded_data["train"]["user_idx"], encoded_data["train"]["item_idx"],
        encoded_data["train"]["rating"], encoded_data["train"]["explanation_idx"],
        encoded_data["train"]["domain_idx"],
    )
    valid_dataset = TensorDataset(
        encoded_data["valid"]["user_idx"], encoded_data["valid"]["item_idx"],
        encoded_data["valid"]["rating"], encoded_data["valid"]["explanation_idx"],
        encoded_data["valid"]["domain_idx"],
    )

    suser_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, args.auxiliary, "user_profiles.npy")),
        dtype=torch.float, device=device,
    )
    sitem_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, args.auxiliary, "item_profiles.npy")),
        dtype=torch.float, device=device,
    )
    sdomain_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, args.auxiliary, "domain.npy")),
        dtype=torch.float, device=device,
    )
    tuser_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, args.target, "user_profiles.npy")),
        dtype=torch.float, device=device,
    )
    titem_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, args.target, "item_profiles.npy")),
        dtype=torch.float, device=device,
    )
    tdomain_profiles = torch.tensor(
        np.load(os.path.join(DATA_DIR, args.target, "domain.npy")),
        dtype=torch.float, device=device,
    )

    domain_profiles = torch.cat([sdomain_profiles.unsqueeze(0), tdomain_profiles.unsqueeze(0)], dim=0)
    user_profiles = torch.cat([tuser_profiles, suser_profiles], dim=0)
    item_profiles = torch.cat([titem_profiles, sitem_profiles], dim=0)

    model = Model(
        config.get("nuser"), config.get("nitem"), config.get("ntoken"),
        config.get("emsize"), config.get("nhead"), config.get("nhid"),
        config.get("nlayers"), config.get("dropout"),
        user_profiles, item_profiles, domain_profiles,
    ).to(device)
    discriminator = Discriminator(config.get("emsize")).to(device)
    return config, train_dataset, valid_dataset, model, discriminator


def build_config_and_data_ddp(args, rank: int, world_size: int, local_rank: int):
    _tid = resolve_task_idx_from_aux_target(args.auxiliary, args.target)
    if _tid is None:
        raise ValueError("未知的 auxiliary/target 组合")
    G_raw = args.batch_size if args.batch_size is not None else get_train_batch_size(_tid)
    if getattr(args, "per_device_batch_size", None) is not None:
        p_res = int(args.per_device_batch_size)
    else:
        p_res = get_per_device_train_batch_size_optional(_tid)
    a_cli = getattr(args, "gradient_accumulation_steps", None)

    G, P, A = resolve_ddp_train_microbatch_layout(
        int(G_raw),
        world_size,
        per_device_batch_size=p_res,
        gradient_accumulation_steps=a_cli,
        task_idx=_tid,
    )
    eff_global = P * world_size * A
    if eff_global != G:
        raise RuntimeError(f"内部错误: 期望 G={G} 与 P×W×A={eff_global} 一致")

    config, train_dataset, valid_dataset, model, discriminator = _load_advtrain_artefacts(
        args,
        local_rank,
        P,
        log_tokenize=(rank == 0),
        show_datasets_progress=(rank == 0),
    )
    config["batch_size_global"] = G
    config["effective_global_batch_size"] = eff_global
    config["per_device_batch_size"] = P
    config["gradient_accumulation_steps"] = A
    config["ddp_world_size"] = world_size
    config["batch_size"] = P
    config["device"] = local_rank
    config["device_ids"] = list(range(world_size))

    train_drop_last = A > 1
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=train_drop_last,
    )
    # DataLoader 并行加载：与 datasets.map 的 num_proc 独立，见 get_ddp_train_num_workers_per_rank
    num_workers = get_ddp_train_num_workers_per_rank(world_size)
    dl_valid_cfg = get_dataloader_num_workers("valid")
    valid_num_workers = max(1, min(dl_valid_cfg, num_workers))
    pin_memory = torch.cuda.is_available()
    _pf_train = get_dataloader_prefetch_factor(num_workers)
    _pf_valid = get_dataloader_prefetch_factor(valid_num_workers)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=P,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=_pf_train,
        drop_last=train_drop_last,
    )
    n_train_micro = len(train_dataloader)
    if A > 1 and n_train_micro % A != 0:
        raise ValueError(
            f"train DataLoader 每 epoch 批次数为 {n_train_micro}，无法被 gradient_accumulation_steps={A} 整除。"
            f"请调整全局 batch、--per-device-batch-size、world_size 或数据划分；或令 accum=1。"
        )
    # valid：不按梯度累积拆微批，每 rank 仍用「全局 batch / world_size」以维持与旧版一致的 valid 吞吐语义
    valid_per_rank = max(1, G // world_size)
    # valid 也分片到所有 rank，避免 rank0 单点跑 valid 导致其它 rank 在 broadcast/all_reduce 上卡死。
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
        num_workers=valid_num_workers,
        pin_memory=pin_memory,
        persistent_workers=valid_num_workers > 0,
        prefetch_factor=_pf_valid,
    )

    # True：对抗/多分支 forward 下常有子图未参与某步 loss；False 会触发 DDP reduction 报错（见 PyTorch 文档）
    _ddp_find_unused = bool(getattr(args, "ddp_find_unused_parameters", True))
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
    config["ddp_find_unused_parameters"] = _ddp_find_unused
    config["rank0_only_logging"] = True
    config["dataloader_num_workers"] = {"train": int(num_workers), "valid": int(valid_num_workers)}
    config["min_epochs"] = (
        args.min_epochs if getattr(args, "min_epochs", None) is not None else get_train_min_epochs()
    )
    config["early_stop_patience"] = (
        args.early_stop_patience
        if getattr(args, "early_stop_patience", None) is not None
        else get_train_early_stop_patience()
    )
    if getattr(args, "early_stop_patience_full", None) is not None:
        config["early_stop_patience_full"] = max(1, int(args.early_stop_patience_full))
    elif getattr(args, "early_stop_patience", None) is not None:
        config["early_stop_patience_full"] = max(1, int(args.early_stop_patience))
    else:
        config["early_stop_patience_full"] = get_train_early_stop_patience_full()
    config["early_stop_patience_loss"] = resolve_early_stop_patience_loss(
        getattr(args, "early_stop_patience_loss", None),
        getattr(args, "early_stop_patience", None),
    )
    config["checkpoint_metric"] = getattr(args, "checkpoint_metric", "bleu4")
    config["bleu4_max_samples"] = (
        args.bleu4_max_samples
        if getattr(args, "bleu4_max_samples", None) is not None
        else get_train_bleu4_max_samples()
    )
    config["quick_eval_max_samples"] = (
        args.quick_eval_max_samples
        if getattr(args, "quick_eval_max_samples", None) is not None
        else get_quick_eval_max_samples(int(config["bleu4_max_samples"]))
    )
    _fe, _phased = resolve_full_bleu_eval_training(
        getattr(args, "full_eval_every", None),
        task_idx=_tid,
    )
    config["full_eval_every_epochs"] = _fe
    config["full_bleu_eval_every_epochs"] = _fe
    config["full_eval_phased"] = _phased
    config["dual_bleu_eval_optimized"] = (
        get_train_mode() == "optimized"
        and config["checkpoint_metric"] == "bleu4"
        and (_fe > 0 or _phased is not None)
    )
    config["valid_dataset"] = valid_dataset
    _apply_adversarial_schedule_to_config(args, config)

    return config, train_dataloader, valid_dataloader, model, discriminator, sampler


@contextlib.contextmanager
def _ddp_no_sync_both(model, discriminator, world_size: int, sync_gradients: bool):
    """在梯度累积的非边界微批上使用 DDP no_sync，边界步上规约梯度。"""
    if world_size <= 1 or sync_gradients:
        yield
    else:
        with model.no_sync(), discriminator.no_sync():
            yield


def _valid_bleu4_quick(model, valid_dataset, device, max_samples: int) -> float:
    """在验证集前 max_samples 条上算 BLEU-4（与 evaluate_text 词级 BLEU 一致），仅 rank0 调用。"""
    _m = get_underlying_model(model)
    n = min(len(valid_dataset), max_samples)
    if n <= 0:
        return 0.0
    subset = Subset(valid_dataset, list(range(n)))
    bs = min(32, n)
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
    pred_texts = []
    ref_texts = []
    _m.eval()
    with torch.inference_mode():
        for batch in dl:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _m.gather(batch, device)
            gen_ids, _ = _m.generate(user_idx, item_idx, domain_idx)
            pred_texts.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            ref_texts.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
    scores = compute_bleu1234_only(pred_texts, ref_texts)
    return float(scores.get("4", 0.0))


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


def _apply_adversarial_schedule_to_config(args, config: dict) -> None:
    """写入 adversarial_schedule_enabled / start_epoch / warmup_epochs / coef_target；未配置时与旧版一致。"""
    legacy = float(config["adversarial_coef"])
    start = getattr(args, "adversarial_start_epoch", None)
    if start is None and "D4C_ADVERSARIAL_START_EPOCH" in os.environ:
        start = int(os.environ["D4C_ADVERSARIAL_START_EPOCH"])
    if start is None:
        config["adversarial_schedule_enabled"] = False
        config["adversarial_start_epoch"] = 0
        config["adversarial_warmup_epochs"] = 0
        config["adversarial_coef_target"] = legacy
        return
    w = getattr(args, "adversarial_warmup_epochs", None)
    if w is None:
        if "D4C_ADVERSARIAL_WARMUP_EPOCHS" in os.environ:
            w = int(os.environ["D4C_ADVERSARIAL_WARMUP_EPOCHS"])
        else:
            w = 0
    t = getattr(args, "adversarial_coef_target", None)
    if t is None:
        if "D4C_ADVERSARIAL_COEF_TARGET" in os.environ:
            t = float(os.environ["D4C_ADVERSARIAL_COEF_TARGET"])
        else:
            t = legacy
    config["adversarial_schedule_enabled"] = True
    # adversarial_start_epoch = N：前 N 个 epoch（1..N）不启用对抗，自第 N+1 个 epoch 起 warmup
    config["adversarial_start_epoch"] = max(0, int(start))
    config["adversarial_warmup_epochs"] = max(0, int(w))
    config["adversarial_coef_target"] = float(t)


def trainModel_ddp(
    model,
    discriminator,
    train_dataloader,
    valid_dataloader,
    sampler,
    config,
    rank,
    world_size,
    max_steps=None,
):
    epochs = config["epochs"]
    G = int(config.get("batch_size_global", config.get("batch_size", 0)))
    P = int(config.get("per_device_batch_size", config.get("batch_size", 0)))
    A = max(1, int(config.get("gradient_accumulation_steps", 1)))
    eff = int(config.get("effective_global_batch_size", P * world_size * A))
    initial_lr = float(config.get("scheduler_initial_lr", config["learning_rate"]))
    learning_rate = initial_lr
    coef = config["coef"]
    adversarial_coef_legacy = float(config["adversarial_coef"])
    adv_schedule_on = bool(config.get("adversarial_schedule_enabled", False))
    adv_skip_epochs = int(config.get("adversarial_start_epoch", 0))
    adv_warmup_epochs = int(config.get("adversarial_warmup_epochs", 0))
    adv_coef_target = float(config.get("adversarial_coef_target", adversarial_coef_legacy))
    _model = get_underlying_model(model)
    device = config["device"]
    n_micro = len(train_dataloader)
    n_steps = max(1, n_micro // A)
    train_info = (
        f"[Train] global_batch_size={G} effective_global_batch_size={eff} "
        f"per_device_batch_size={P} gradient_accumulation_steps={A} world_size={world_size} "
        f"micro_batches_per_epoch={n_micro} optimizer_steps_per_epoch={n_steps} epochs={epochs}"
    )
    _lg = config.get("logger")
    min_epochs = int(config.get("min_epochs", get_train_min_epochs()))
    early_stop_patience = int(config.get("early_stop_patience", get_train_early_stop_patience()))
    early_stop_patience_full = int(
        config.get("early_stop_patience_full", get_train_early_stop_patience_full())
    )
    early_stop_patience_loss = int(
        config.get(
            "early_stop_patience_loss",
            resolve_early_stop_patience_loss(None, None),
        )
    )
    checkpoint_metric = config.get("checkpoint_metric", "bleu4")
    bleu4_max_samples = int(config.get("bleu4_max_samples", get_train_bleu4_max_samples()))
    quick_eval_max_samples = int(
        config.get("quick_eval_max_samples", get_quick_eval_max_samples(bleu4_max_samples))
    )
    valid_dataset_for_bleu = config.get("valid_dataset")
    lr_scheduler_type = config.get("lr_scheduler", get_lr_scheduler_type())
    warmup_epochs = float(config.get("warmup_epochs", get_warmup_epochs()))
    full_eval_every = int(
        config.get("full_eval_every_epochs", config.get("full_bleu_eval_every_epochs", 0))
    )
    full_eval_phased = config.get("full_eval_phased")
    dual_bleu = bool(config.get("dual_bleu_eval_optimized", False))
    train_mode = str(config.get("train_mode", get_train_mode()))
    min_lr_ratio = float(
        config.get("min_lr_ratio", get_min_lr_ratio(config.get("task_idx"))),
    )
    warmup_steps_env = config.get("d4c_warmup_steps")
    warmup_ratio_env = config.get("d4c_warmup_ratio")
    total_steps_plan = max(1, int(epochs * n_steps))
    best_bleu4 = -1.0
    best_full_bleu4 = -1.0
    enduration = 0
    full_eval_stall = 0
    valid_loss_stall = 0
    prev_valid_loss = float("inf")
    if rank == 0:
        if _lg:
            _lg.info(train_info)
        else:
            print(train_info, flush=True)
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
            f"full_bleu_schedule={_sched}, dual_bleu_eval_optimized={dual_bleu}"
        )
        if _lg:
            _lg.info(_es)
        else:
            print(_es, flush=True)
        if _lg:
            _lg.info(
                "Train profile: mode=%s exact_reproduction=%s allow_large_batch=%s "
                "lr_scheduler=%s warmup_epochs=%g full_bleu_schedule=%s",
                get_train_mode(),
                get_exact_reproduction(),
                get_allow_large_batch(),
                lr_scheduler_type,
                warmup_epochs,
                _sched,
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
            train_mode=train_mode,
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
                _lg.info(_adv_msg)
            else:
                print(_adv_msg, flush=True)
    device_ids = config.get("device_ids") or [device]
    train_nw = getattr(train_dataloader, "num_workers", 0)
    valid_nw = getattr(valid_dataloader, "num_workers", 0) if valid_dataloader is not None else None
    perf = None
    if rank == 0:
        perf = PerfMonitor(
            device=config["device"],
            log_file=config.get("log_file"),
            num_proc=config.get("num_proc"),
            device_ids=device_ids,
            train_num_workers=train_nw,
            valid_num_workers=valid_nw,
            training_logger=_lg,
        )
        perf.start()
    step_count = 0
    global_step = 0
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
            loss_sum = 0.0
            adv_sum = 0.0
            n_samples = 0
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
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
                bsz = int(user_idx.size(0))
                if use_adv:
                    # 同一微批上 D 与 G 的两次 backward 须在同一段 no_sync 内，避免中间触发规约
                    with sync_ctx:
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
                        loss_e = _model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
                        loss_c = _model.exp_loss_fn(context_dist.view(-1, 32128), tgt_output.reshape(-1))
                        loss = 0.1 * loss_r + coef * loss_c + loss_e + adv_coef_epoch * g_loss
                        (loss * inv_accum).backward()

                    if sync:
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
                    adv_sum += g_loss.item() * bsz
                else:
                    with sync_ctx:
                        pred_rating, context_dist, word_dist, user_hidden, item_hidden = model(
                            user_idx, item_idx, tgt_input, domain_idx,
                        )
                        loss_r = _model.rating_loss_fn(pred_rating, rating)
                        loss_e = _model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
                        loss_c = _model.exp_loss_fn(context_dist.view(-1, 32128), tgt_output.reshape(-1))
                        loss = 0.1 * loss_r + coef * loss_c + loss_e
                        (loss * inv_accum).backward()
                    if sync:
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # LambdaLR：必须在 optimizer.step() 之后调用
                        if sched is not None:
                            sched.step()
                        global_step += 1

                loss_sum += loss.item() * bsz
                n_samples += bsz

                # 用于快速验证：跑到指定 steps 后直接退出，观察是否触发 DDP reduction 错误
                if max_steps is not None and step_count >= max_steps:
                    return

            t_ls = torch.tensor([loss_sum], dtype=torch.double, device=device)
            t_as = torch.tensor([adv_sum], dtype=torch.double, device=device)
            t_ns = torch.tensor([float(n_samples)], dtype=torch.double, device=device)
            dist.all_reduce(t_ls, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_as, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_ns, op=dist.ReduceOp.SUM)
            avg_loss = t_ls.item() / t_ns.item()
            avg_adv_loss = t_as.item() / t_ns.item()

            lr_epoch = optimizer.param_groups[0]["lr"]
            _gu, _gm, _gpeak = gather_ddp_gpu_stats_for_epoch_log(rank, world_size, int(device))
            if rank == 0:
                rec = perf.epoch_end(epoch + 1, len(train_dataloader), emit_log=False)
                rec["gpu_util"] = _gu
                rec["gpu_mem"] = _gm
                if _gpeak is not None:
                    rec["gpu_mem_bytes"] = _gpeak
            else:
                rec = None

            # 每个 rank 都跑自己的 valid shard，然后 all_reduce 聚合为全局“按 batch 平均”的 valid loss。
            valid_loss_sum, valid_n_batches = validModel_sum_batches(model, valid_dataloader, device)
            t_ls = torch.tensor([valid_loss_sum], dtype=torch.double, device=device)
            t_nb = torch.tensor([float(valid_n_batches)], dtype=torch.double, device=device)
            dist.all_reduce(t_ls, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_nb, op=dist.ReduceOp.SUM)
            current_valid_loss = float(t_ls.item() / t_nb.item())

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
                    quick_bleu4 = _valid_bleu4_quick(
                        model, valid_dataset_for_bleu, device, quick_eval_max_samples
                    )
                    bleu4_score_this_epoch = quick_bleu4
                if world_size > 1:
                    dist.barrier()
                if is_full_eval_epoch:
                    full_bleu4_val = bleu4_explanation_full_valid_ddp(
                        model,
                        valid_dataset_for_bleu,
                        tokenizer=tokenizer,
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        batch_size=32,
                    )
            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bleu_line = None
                if checkpoint_metric == "bleu4" and valid_dataset_for_bleu is not None and quick_bleu4 is not None:
                    if dual_bleu:
                        fstr = f"{full_bleu4_val:.4f}" if full_bleu4_val is not None else "na"
                        ckpt_src = "full_bleu4" if is_full_eval_epoch else "trend_quick_bleu4"
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
                        torch.save(get_underlying_model(model).state_dict(), config.get("save_file"))
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
                        torch.save(get_underlying_model(model).state_dict(), config.get("save_file"))
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
                        torch.save(get_underlying_model(model).state_dict(), config.get("save_file"))
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
                torch.save(get_underlying_model(model).state_dict(), config.get("save_file"))

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
                    _lg.info(_es_line)
                else:
                    print(_es_line, flush=True)

            dist.barrier()

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
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
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


def build_config_and_dataloader(args, ddp_rank=None, ddp_world_size=None, local_rank=None):
    use_eval_ddp = ddp_world_size is not None
    if use_eval_ddp:
        if ddp_rank is None or local_rank is None:
            raise ValueError("DDP eval 须同时提供 ddp_rank 与 local_rank")
        if args.gpus and ddp_rank == 0:
            logging.warning("torchrun eval DDP 下忽略 --gpus，请用 CUDA_VISIBLE_DEVICES 指定可见 GPU。")
        device_ids = [local_rank]
        primary_device = local_rank
    elif args.gpus:
        device_ids = [int(x.strip()) for x in args.gpus.split(",")]
        primary_device = device_ids[0]
    else:
        device_ids = [args.device]
        primary_device = args.device

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    path = os.path.join(MERGED_DATA_DIR, str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    nuser = train_df['user_idx'].max() + 1
    nitem = train_df['item_idx'].max() + 1

    batch_size = args.batch_size if args.batch_size is not None else get_eval_batch_size()
    nproc = args.num_proc if args.num_proc is not None else get_num_proc()

    if use_eval_ddp:
        if batch_size % ddp_world_size != 0:
            raise ValueError(
                f"DDP eval 要求全局 batch_size ({batch_size}) 能被进程数 ({ddp_world_size}) 整除，请调整 --batch-size。"
            )
        loader_batch_size = batch_size // ddp_world_size
    else:
        loader_batch_size = batch_size

    config = {
        "task_idx": task_idx,
        "device": primary_device if torch.cuda.is_available() else args.device,
        "log_file": args.log_file,
        "save_file": args.save_file or os.path.join(get_checkpoint_task_dir(task_idx), "model.pth"),
        "batch_size": loader_batch_size,
        "emsize": 768,
        "nlayers": args.nlayers,
        "nhid": 2048,
        "ntoken": 32128,
        "dropout": 0.2,
        "nuser": nuser,
        "nitem": nitem,
        "nhead": 2
    }
    if use_eval_ddp:
        config["batch_size_global"] = batch_size

    valid_df = pd.read_csv(path + "/aug_valid.csv")
    valid_df['item'] = valid_df['item'].astype(str)
    datasets = DatasetDict({
        "valid": Dataset.from_pandas(valid_df)
    })
    processor = Processor(args.auxiliary, args.target)
    t0 = time.perf_counter()
    with hf_datasets_progress_bar(ddp_rank is None or ddp_rank == 0):
        encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
    if ddp_rank is None or ddp_rank == 0:
        _log_tokenize_done(
            "eval valid",
            nproc,
            time.perf_counter() - t0,
            getattr(args, "log_file", None),
        )
    encoded_data.set_format("torch")
    valid_dataset = TensorDataset(
        encoded_data['valid']['user_idx'], encoded_data['valid']['item_idx'],
        encoded_data['valid']['rating'], encoded_data['valid']['explanation_idx'],
        encoded_data['valid']['domain_idx']
    )
    if use_eval_ddp:
        n_samples = len(valid_dataset)
        shard_idx = list(range(ddp_rank, n_samples, ddp_world_size))
        valid_dataset = Subset(valid_dataset, shard_idx)
        eval_world_size = max(ddp_world_size, 1)
        dl_valid = get_dataloader_num_workers("valid")
        num_workers = min(max(1, dl_valid // eval_world_size), 8)
        pin_memory = torch.cuda.is_available()
        _pf_ev = get_dataloader_prefetch_factor(num_workers)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=_pf_ev,
        )
    else:
        num_workers = min(get_dataloader_num_workers("valid"), 8)
        pin_memory = torch.cuda.is_available()
        _pf_ev = get_dataloader_prefetch_factor(num_workers)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=_pf_ev,
        )

    suser_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, args.auxiliary, "user_profiles.npy")), dtype=torch.float)
    sitem_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, args.auxiliary, "item_profiles.npy")), dtype=torch.float)
    sdomain_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, args.auxiliary, "domain.npy")), dtype=torch.float)
    tuser_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, args.target, "user_profiles.npy")), dtype=torch.float)
    titem_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, args.target, "item_profiles.npy")), dtype=torch.float)
    tdomain_profiles = torch.tensor(np.load(os.path.join(DATA_DIR, args.target, "domain.npy")), dtype=torch.float)

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
    if not use_eval_ddp and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    return config, valid_dataloader, model


def _run_train_ddp(args):
    if getattr(args, "train_mode", None):
        os.environ["D4C_TRAIN_MODE"] = str(args.train_mode).strip().lower()
    if getattr(args, "scheduler_initial_lr", None) is not None:
        os.environ["D4C_INITIAL_LR"] = str(args.scheduler_initial_lr)
    if getattr(args, "warmup_steps", None) is not None:
        os.environ["D4C_WARMUP_STEPS"] = str(args.warmup_steps)
    if getattr(args, "warmup_ratio", None) is not None:
        os.environ["D4C_WARMUP_RATIO"] = str(args.warmup_ratio)
    if getattr(args, "min_lr_ratio", None) is not None:
        os.environ["D4C_MIN_LR_RATIO"] = str(args.min_lr_ratio)
    if getattr(args, "quick_eval_max_samples", None) is not None:
        os.environ["D4C_QUICK_EVAL_MAX_SAMPLES"] = str(args.quick_eval_max_samples)
    if getattr(args, "full_eval_every", None) is not None:
        os.environ["D4C_FULL_EVAL_EVERY"] = str(args.full_eval_every)
    if getattr(args, "early_stop_patience_full", None) is not None:
        os.environ["TRAIN_EARLY_STOP_PATIENCE_FULL"] = str(args.early_stop_patience_full)
    if getattr(args, "early_stop_patience_loss", None) is not None:
        os.environ["TRAIN_EARLY_STOP_PATIENCE_LOSS"] = str(args.early_stop_patience_loss)
    if getattr(args, "adversarial_start_epoch", None) is not None:
        os.environ["D4C_ADVERSARIAL_START_EPOCH"] = str(args.adversarial_start_epoch)
    if getattr(args, "adversarial_warmup_epochs", None) is not None:
        os.environ["D4C_ADVERSARIAL_WARMUP_EPOCHS"] = str(args.adversarial_warmup_epochs)
    if getattr(args, "adversarial_coef_target", None) is not None:
        os.environ["D4C_ADVERSARIAL_COEF_TARGET"] = str(args.adversarial_coef_target)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    ddp_fast_backends = apply_optimized_torch_backends()
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

    try:
        config, train_dataloader, valid_dataloader, model, discriminator, sampler = build_config_and_data_ddp(
            args, rank, world_size, local_rank,
        )
        config["run_id"] = run_id
        config["logger"] = train_logger
        config["train_mode"] = get_train_mode()
        config["exact_reproduction"] = get_exact_reproduction()
        config["allow_large_batch"] = get_allow_large_batch()
        config["lr_scheduler"] = get_lr_scheduler_type()
        config["scheduler_type"] = config["lr_scheduler"]
        config["warmup_epochs"] = get_warmup_epochs()
        config["scheduler_initial_lr"] = get_scheduler_initial_lr(args.learning_rate)
        config["min_lr_ratio"] = get_min_lr_ratio(task_idx)
        config["d4c_warmup_steps"] = get_d4c_warmup_steps_optional()
        config["d4c_warmup_ratio"] = get_d4c_warmup_ratio_optional()
        config["ddp_fast_backends"] = ddp_fast_backends
        if rank == 0:
            log_run_header(
                train_logger,
                {
                    "mode": get_train_mode(),
                    "exact_reproduction": get_exact_reproduction(),
                    "allow_large_batch": get_allow_large_batch(),
                    "run_id": run_id,
                    "task_idx": task_idx,
                    "rank": rank,
                    "world_size": world_size,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "local_rank": local_rank,
                    "learning_rate": config["learning_rate"],
                    "batch_size": config.get("batch_size_global", config.get("batch_size")),
                    "batch_size_global": config.get("batch_size_global"),
                    "per_device_batch_size": config.get("per_device_batch_size"),
                    "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
                    "effective_global_batch_size": config.get("effective_global_batch_size"),
                    "epochs": config["epochs"],
                    "save_file": os.path.abspath(str(config.get("save_file", ""))),
                    "log_file": os.path.abspath(log_path),
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                },
            )
            log_config_snapshot(train_logger, config)

        trainModel_ddp(
            model,
            discriminator,
            train_dataloader,
            valid_dataloader,
            sampler,
            config,
            rank,
            world_size,
            max_steps=getattr(args, "max_steps", None),
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _eval_torchrun_env():
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


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


def _run_eval_single(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        primary = int(args.gpus.split(",")[0].strip()) if args.gpus else args.device
        _ = torch.zeros(1, device=f"cuda:{primary}")

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")
    log_path, run_id = create_run_paths(task_idx, args.log_file)
    args.log_file = log_path
    _setup = setup_train_logging(
        log_file=log_path,
        task_idx=task_idx,
        rank=0,
        world_size=1,
        run_id=run_id,
    )
    ev_logger = _setup["logger"]

    config, valid_dataloader, model = build_config_and_dataloader(args)
    config["logger"] = ev_logger
    config["run_id"] = run_id
    log_run_header(
        ev_logger,
        {
            "run_id": run_id,
            "task_idx": task_idx,
            "rank": 0,
            "world_size": 1,
            "mode": "eval",
            "cuda_available": bool(torch.cuda.is_available()),
            "learning_rate": None,
            "batch_size": config.get("batch_size_global", config.get("batch_size")),
            "save_file": os.path.abspath(str(config.get("save_file", ""))),
            "log_file": os.path.abspath(log_path),
            "auxiliary": args.auxiliary,
            "target": args.target,
        },
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(config["device"])

    _eval_t0 = time.time()
    _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final = evalModel(model, valid_dataloader, config.get("device"))
    _eval_elapsed = time.time() - _eval_t0
    _td = f"Step 3 AdvTrain eval Task {task_idx}: {args.auxiliary} -> {args.target}"
    _write_eval_results_log(
        config,
        final,
        task_description=_td,
        pipeline="AdvTrain_eval",
        domain_from=args.auxiliary,
        domain_to=args.target,
        start_time=_eval_start_str,
        eval_elapsed=_eval_elapsed,
    )
    ev_logger.info("DONE.")


def _run_eval_ddp(args):
    if not torch.cuda.is_available():
        raise RuntimeError("AdvTrain.py eval 的 torchrun DDP 需要 CUDA + NCCL。")
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
    if _eval_torchrun_env():
        if not torch.cuda.is_available():
            raise RuntimeError("检测到 torchrun 环境但无 CUDA，无法使用 eval DDP。")
        _run_eval_ddp(args)
    else:
        _run_eval_single(args)


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
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None, help="不传则用 config.epochs")
    p.add_argument("--coef", type=float, default=0.5)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--adv", type=float, default=1.0)
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
        help="训练全局有效 batch（每优化步跨所有 rank 的样本总数）；与 --global-batch-size 同义择一即可",
    )
    p.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="同 --batch-size：显式强调「全局 batch」语义",
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
    p.add_argument("--gpus", type=str, default=None, help="兼容保留；train 的 GPU 由 torchrun 分配")
    p.add_argument("--max-steps", type=int, default=None, help="快速验证用：最多训练到 N 个 batch 就退出")
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
        help="optimized+dual_bleu 时：连续多少次 full BLEU eval 未刷新 best 则早停（quick 不参与）；"
        "默认 TRAIN_EARLY_STOP_PATIENCE_FULL 或与 --early-stop-patience 相同",
    )
    p.add_argument(
        "--early-stop-patience-loss",
        type=int,
        default=None,
        help="optimized+dual_bleu：valid_loss 连续变差早停次数，与 patience_full 独立；"
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
        "--train-mode",
        type=str,
        choices=["reproduction", "optimized"],
        default=None,
        help="训练模式：reproduction 论文复现；optimized 工程优化（等价 export D4C_TRAIN_MODE）",
    )
    p.add_argument(
        "--scheduler-initial-lr",
        type=float,
        default=None,
        help="优化器初始 LR，覆盖 --learning_rate；等价 D4C_INITIAL_LR",
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
        "--full-eval-every",
        type=int,
        default=None,
        help="每 N epoch full valid BLEU；不设且未 export D4C_FULL_EVAL_EVERY 时 optimized 用分阶段默认",
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
    p.add_argument("--gpus", type=str, default=None, help="仅单进程 eval：多卡时用 DataParallel；torchrun eval 下忽略")


def main():
    parser = argparse.ArgumentParser(description="AdvTrain：train（需 torchrun）| eval")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="DDP 域对抗预训练，须 torchrun 启动")
    _add_train_args(p_train)
    p_eval = sub.add_parser("eval", help="valid 评估：python 单进程，或 torchrun 多卡 DDP")
    _add_eval_args(p_eval)
    args = parser.parse_args()
    if args.command == "train":
        _run_train_ddp(args)
    else:
        _dispatch_eval(args)


if __name__ == "__main__":
    main()
