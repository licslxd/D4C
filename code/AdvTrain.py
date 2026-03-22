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
import logging
from datetime import datetime
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
import torch.distributed as dist
from transformers import T5Tokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from perf_monitor import PerfMonitor
from datasets import Dataset, DatasetDict
from config import get_train_batch_size, get_epochs, get_num_proc, get_dataloader_num_workers, get_max_parallel_cpu
from train_logging import (
    create_run_paths,
    setup_train_logging,
    log_run_header,
    format_epoch_line,
    broadcast_run_paths_ddp,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    append_eval_run_summaries,
    LOGGER_NAME,
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
    if also_print:
        print(msg, flush=True)
    lg = logging.getLogger(LOGGER_NAME)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


def _load_advtrain_artefacts(args, device: int, batch_size: int, *, log_tokenize: bool = True):
    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break

    path = os.path.join(MERGED_DATA_DIR, str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    nuser = train_df["user_idx"].max() + 1
    nitem = train_df["item_idx"].max() + 1

    os.makedirs(get_checkpoint_task_dir(task_idx), exist_ok=True)
    epochs = args.epochs if args.epochs is not None else get_epochs()
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
    global_bs = args.batch_size if args.batch_size is not None else get_train_batch_size()
    if global_bs % world_size != 0:
        raise ValueError(
            f"DDP 要求全局 batch_size ({global_bs}) 能被进程数 ({world_size}) 整除，"
            f"请调整 --batch-size 或 WORLD_SIZE。"
        )
    per_gpu = global_bs // world_size
    config, train_dataset, valid_dataset, model, discriminator = _load_advtrain_artefacts(
        args, local_rank, per_gpu, log_tokenize=(rank == 0),
    )
    config["batch_size_global"] = global_bs
    config["batch_size"] = per_gpu
    config["device"] = local_rank
    config["device_ids"] = list(range(world_size))

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # DataLoader 并行加载：与 datasets.map 的 num_proc 独立，见 config.get_dataloader_num_workers
    nproc = args.num_proc if args.num_proc is not None else get_num_proc()
    dl_train = get_dataloader_num_workers("train")
    num_workers = min(max(1, dl_train // max(world_size, 1)), get_max_parallel_cpu())
    pin_memory = torch.cuda.is_available()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_gpu,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
    )
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
        batch_size=per_gpu,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        # 训练里存在“某些参数有时不参与 loss 计算”的情况。
        # 需要开启 unused 参数检测，避免 DDP reduction 报错。
        find_unused_parameters=True,
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[local_rank],
        output_device=local_rank,
        # 同上：避免出现 reduction 不匹配。
        find_unused_parameters=True,
    )
    return config, train_dataloader, valid_dataloader, model, discriminator, sampler


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
    batch_size_disp = config.get("batch_size_global", config.get("batch_size", "?"))
    learning_rate = config["learning_rate"]
    coef = config["coef"]
    adversarial_coef = config["adversarial_coef"]
    _model = get_underlying_model(model)
    device = config["device"]
    n_steps = len(train_dataloader)
    train_info = f"[Train] batch_size={batch_size_disp}, epochs={epochs}, steps_per_epoch={n_steps} (DDP x{world_size})"
    _lg = config.get("logger")
    if rank == 0:
        if _lg:
            _lg.info(train_info)
        else:
            print(train_info, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate * 0.5, weight_decay=1e-5)
    adversarial_loss_fn = BCEWithLogitsLoss(label_smoothing=0.3)
    enduration = 0
    prev_valid_loss = float("inf")
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
    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            if rank == 0:
                perf.epoch_start()
            model.train()
            discriminator.train()
            loss_sum = 0.0
            adv_sum = 0.0
            n_samples = 0
            iterator = train_dataloader
            if rank == 0:
                iterator = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in iterator:
                step_count += 1
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
                bsz = int(user_idx.size(0))
                pred_rating, context_dist, word_dist, user_hidden, item_hidden = model(
                    user_idx, item_idx, tgt_input, domain_idx,
                )
                target_labels = torch.ones(user_hidden.size(0), 1, device=device)
                aux_labels = torch.zeros(user_hidden.size(0), 1, device=device)

                d_optimizer.zero_grad()
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
                d_loss.backward()
                d_optimizer.step()

                optimizer.zero_grad()
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
                loss = 0.1 * loss_r + coef * loss_c + loss_e + adversarial_coef * g_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                loss_sum += loss.item() * bsz
                adv_sum += g_loss.item() * bsz
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
            if rank == 0:
                perf.epoch_end(epoch + 1, len(train_dataloader))

            # 每个 rank 都跑自己的 valid shard，然后 all_reduce 聚合为全局“按 batch 平均”的 valid loss。
            valid_loss_sum, valid_n_batches = validModel_sum_batches(model, valid_dataloader, device)
            t_ls = torch.tensor([valid_loss_sum], dtype=torch.double, device=device)
            t_nb = torch.tensor([float(valid_n_batches)], dtype=torch.double, device=device)
            dist.all_reduce(t_ls, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_nb, op=dist.ReduceOp.SUM)
            current_valid_loss = float(t_ls.item() / t_nb.item())

            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = format_epoch_line(
                    epoch + 1,
                    current_time,
                    lr_epoch,
                    avg_loss,
                    current_valid_loss,
                    adv_loss=avg_adv_loss,
                )
                if _lg:
                    _lg.info(line)
                else:
                    print(line, flush=True)

            if current_valid_loss > prev_valid_loss:
                learning_rate /= 2.0
                enduration += 1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate
                for param_group in d_optimizer.param_groups:
                    param_group["lr"] = learning_rate
            else:
                if rank == 0:
                    torch.save(get_underlying_model(model).state_dict(), config.get("save_file"))
            prev_valid_loss = current_valid_loss
            if enduration >= 5:
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

    path = os.path.join(MERGED_DATA_DIR, str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    nuser = train_df['user_idx'].max() + 1
    nitem = train_df['item_idx'].max() + 1

    batch_size = args.batch_size if args.batch_size is not None else get_train_batch_size()
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
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        num_workers = min(get_dataloader_num_workers("valid"), 8)
        pin_memory = torch.cuda.is_available()
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
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
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
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
        config, train_dataloader, valid_dataloader, model, discriminator, sampler = build_config_and_data_ddp(
            args, rank, world_size, local_rank,
        )
        config["run_id"] = run_id
        config["logger"] = train_logger
        if rank == 0:
            log_run_header(
                train_logger,
                {
                    "run_id": run_id,
                    "task_idx": task_idx,
                    "rank": rank,
                    "world_size": world_size,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "local_rank": local_rank,
                    "learning_rate": config["learning_rate"],
                    "batch_size": config.get("batch_size_global", config.get("batch_size")),
                    "epochs": config["epochs"],
                    "save_file": os.path.abspath(str(config.get("save_file", ""))),
                    "log_file": os.path.abspath(log_path),
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                },
            )
            train_logger.info("Config dict: %s", {k: v for k, v in config.items() if k != "logger"})

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
):
    lines = format_final_results_lines(final, task_description=task_description)
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

    final = evalModel(model, valid_dataloader, config.get("device"))
    _td = f"Step 3 AdvTrain eval Task {task_idx}: {args.auxiliary} -> {args.target}"
    _write_eval_results_log(
        config,
        final,
        task_description=_td,
        pipeline="AdvTrain_eval",
        domain_from=args.auxiliary,
        domain_to=args.target,
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
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-proc", type=int, default=None)
    p.add_argument("--gpus", type=str, default=None, help="兼容保留；train 的 GPU 由 torchrun 分配")
    p.add_argument("--max-steps", type=int, default=None, help="快速验证用：最多训练到 N 个 batch 就退出")


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
