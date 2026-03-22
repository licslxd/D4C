import os
import sys
# 离线模式：禁止从网络加载
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_utils import *
from paths_config import DATA_DIR, MERGED_DATA_DIR, T5_SMALL_DIR, get_checkpoint_task_dir
from config import get_train_batch_size, get_epochs, get_num_proc, get_dataloader_num_workers, get_max_parallel_cpu
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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from transformers import T5Tokenizer
from torch import nn, optim
from torch.nn.modules.transformer import _get_activation_fn
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from perf_monitor import PerfMonitor
import numpy as np
import copy
import torch.nn.functional as F
from train_logging import (
    create_run_paths,
    setup_train_logging,
    log_run_snapshot,
    format_epoch_line,
    broadcast_run_paths_ddp,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    append_eval_run_summaries,
)

# 离线加载：优先使用本地 T5 目录
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


def trainModel_ddp(model, train_dataloader, valid_dataloader, sampler, config, rank, world_size):
    epochs = config["epochs"]
    batch_size_disp = config.get("batch_size_global", config.get("batch_size", "?"))
    learning_rate = config["learning_rate"]
    coef = config["coef"]
    eta = config["eta"]
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
    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            if rank == 0:
                perf.epoch_start()
            model.train()
            loss_sum = 0.0
            n_samples = 0
            iterator = train_dataloader
            if rank == 0:
                iterator = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in iterator:
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
                bsz = int(user_idx.size(0))
                pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input, domain_idx)
                factual_idx = (domain_idx == 1).squeeze()
                counterfactual_idx = (domain_idx == 0).squeeze()
                if factual_idx.sum() > 0:
                    loss_r_factual = _model.rating_loss_fn(pred_rating[factual_idx], rating[factual_idx])
                    loss_e_factual = _model.exp_loss_fn(word_dist[factual_idx].view(-1, 32128), tgt_output[factual_idx].reshape(-1))
                    loss_c_factual = _model.exp_loss_fn(context_dist[factual_idx].view(-1, 32128), tgt_output[factual_idx].reshape(-1))
                    loss_factual = coef * loss_r_factual + coef * loss_c_factual + loss_e_factual
                else:
                    loss_factual = torch.zeros((), device=device)
                if counterfactual_idx.sum() > 0:
                    loss_r_counterfactual = _model.rating_loss_fn(pred_rating[counterfactual_idx], rating[counterfactual_idx])
                    loss_e_counterfactual = _model.exp_loss_fn(word_dist[counterfactual_idx].view(-1, 32128), tgt_output[counterfactual_idx].reshape(-1))
                    loss_c_counterfactual = _model.exp_loss_fn(context_dist[counterfactual_idx].view(-1, 32128), tgt_output[counterfactual_idx].reshape(-1))
                    loss_counterfactual = eta * (coef * loss_r_counterfactual + coef * loss_c_counterfactual + loss_e_counterfactual)
                else:
                    loss_counterfactual = torch.zeros((), device=device)
                loss = loss_factual + loss_counterfactual
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                loss_sum += loss.item() * bsz
                n_samples += bsz
            t_ls = torch.tensor([loss_sum], dtype=torch.double, device=device)
            t_ns = torch.tensor([float(n_samples)], dtype=torch.double, device=device)
            dist.all_reduce(t_ls, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_ns, op=dist.ReduceOp.SUM)
            avg_loss = t_ls.item() / t_ns.item()
            lr_epoch = optimizer.param_groups[0]["lr"]
            if rank == 0:
                perf.epoch_end(epoch + 1, len(train_dataloader))

            # 两个 rank 都跑 validModel（forward 在 validModel 内用底层 _model，不走 DDP 包装后的 model(...)）
            current_valid_loss = validModel(model, valid_dataloader, device)
            v_tensor = torch.tensor([current_valid_loss], dtype=torch.float32, device=device)
            dist.broadcast(v_tensor, src=0)
            current_valid_loss = float(v_tensor.item())
            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = format_epoch_line(
                    epoch + 1, current_time, lr_epoch, avg_loss, current_valid_loss,
                )
                if _lg:
                    _lg.info(line)
                else:
                    print(line, flush=True)
            if current_valid_loss > prev_valid_loss:
                learning_rate /= 2.0
                enduration += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            else:
                if rank == 0:
                    torch.save(get_underlying_model(model).state_dict(), config.get("save_file"))
            prev_valid_loss = current_valid_loss
            if enduration >= 5:
                break
    finally:
        if rank == 0 and perf is not None:
            perf.finish()


def validModel(model, valid_dataloader, device):
    _model = get_underlying_model(model)
    # 两个 rank 都可以跑 valid；forward 必须用底层 _model，避免 DDP 包装后的 model(...)
    # 在局部执行情况下触发额外 NCCL collective，导致 watchdog 超时。
    _model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = _model.gather(batch, device)
            pred_rating, context_dist, word_dist = _model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss = loss_r + loss_e
            avg_loss += loss.item()
        avg_loss /= len(valid_dataloader)
        return avg_loss


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
    rating_diffs = prediction_ratings - ground_truth_ratings
    mae = round(np.mean(np.abs(rating_diffs)), 4)
    rmse = round(np.sqrt(np.mean(np.square(rating_diffs))),4)
    text_results = evaluate_text(prediction_exps, reference_exps)
    return {"recommendation": {"mae":mae, "rmse":rmse}, "explanation":text_results}


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


def _make_model(config, args, device_idx):
    domain_profiles, user_profiles, item_profiles = _load_profile_tensors(args.auxiliary, args.target, device_idx)
    return Model(
        config["nuser"], config["nitem"], config["ntoken"], config["emsize"], config["nhead"],
        config["nhid"], config["nlayers"], config["dropout"],
        user_profiles, item_profiles, domain_profiles,
    ).to(device_idx)


def build_d4c_ddp_artefacts(args, world_size, local_rank, eval_only=False):
    global_bs = args.batch_size if args.batch_size is not None else get_train_batch_size()
    if global_bs % world_size != 0:
        raise ValueError(
            f"DDP 要求全局 batch_size ({global_bs}) 能被进程数 ({world_size}) 整除，请调整 --batch-size 或 nproc_per_node。"
        )
    per_gpu = global_bs // world_size
    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    path = os.path.join(DATA_DIR, args.target)
    _ckpt_task = get_checkpoint_task_dir(task_idx)
    os.makedirs(_ckpt_task, exist_ok=True)
    train_path = os.path.join(_ckpt_task, "factuals_counterfactuals.csv")
    train_df = pd.read_csv(train_path)
    nuser = int(train_df["user_idx"].max()) + 1
    nitem = int(train_df["item_idx"].max()) + 1
    save_file = args.save_file or os.path.join(_ckpt_task, "model.pth")
    epochs = args.epochs if args.epochs is not None else get_epochs()
    nproc = args.num_proc if args.num_proc is not None else get_num_proc()
    config = {
        "task_idx": task_idx,
        "device": local_rank,
        "log_file": args.log_file,
        "save_file": save_file,
        "learning_rate": args.learning_rate,
        "epochs": epochs,
        "batch_size": per_gpu,
        "batch_size_global": global_bs,
        "num_proc": nproc,
        "emsize": 768,
        "nlayers": args.nlayers,
        "nhid": 2048,
        "ntoken": 32128,
        "dropout": 0.2,
        "nuser": nuser,
        "nitem": nitem,
        "coef": args.coef,
        "nhead": 2,
        "eta": args.eta,
        "device_ids": list(range(world_size)),
    }
    valid_df = pd.read_csv(path + "/valid.csv")
    valid_df["domain"] = "target"
    processor = Processor(args.auxiliary, args.target)
    if eval_only:
        datasets = DatasetDict({"valid": Dataset.from_pandas(valid_df)})
        encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
        encoded_data.set_format("torch")
        train_dataset = None
    else:
        train_df = train_df[train_df["explanation"].notna()]
        datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "valid": Dataset.from_pandas(valid_df),
        })
        encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
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
    model = _make_model(config, args, local_rank)
    return config, train_dataset, valid_dataset, model


def _run_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("run-d4c.py 仅支持 CUDA + NCCL DDP。")
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
    train_logger = _setup["logger"]

    valid_dataloader = None
    try:
        config, train_dataset, valid_dataset, model = build_d4c_ddp_artefacts(
            args, world_size, local_rank, eval_only=args.eval_only,
        )
        config["run_id"] = run_id
        config["logger"] = train_logger
        if rank == 0:
            log_run_snapshot(
                train_logger,
                {
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                    "eval_only": args.eval_only,
                    "rank": rank,
                    "world_size": world_size,
                    "local_rank": local_rank,
                    "cuda_available": bool(torch.cuda.is_available()),
                },
                config,
            )
        nproc = config.get("num_proc", None) or get_num_proc()
        dl_train = get_dataloader_num_workers("train")
        num_workers = min(max(1, dl_train // max(world_size, 1)), get_max_parallel_cpu())
        pin_memory = torch.cuda.is_available()
        # 两个 rank 都创建 valid_dataloader，这样两张卡可以同时跑 validModel，
        # 避免部分 NCCL collective 的调度不一致导致 watchdog 超时。
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
        if not args.eval_only:
            sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=2,
            )
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
            )
            trainModel_ddp(model, train_dataloader, valid_dataloader, sampler, config, rank, world_size)
        if args.eval_only and not os.path.isfile(config["save_file"]):
            raise FileNotFoundError(
                f"--eval-only 需要已有权重文件，未找到: {config['save_file']}\n"
                "请设置 D4C_CHECKPOINT_GROUP / D4C_CHECKPOINT_SUBDIR 指向训练产物目录，或使用 --save-file。"
            )
        dist.barrier()
        if rank == 0:
            eval_model = _make_model(config, args, local_rank)
            eval_model.load_state_dict(
                torch.load(
                    config["save_file"],
                    map_location=f"cuda:{local_rank}",
                    weights_only=True,
                ),
            )
            final = evalModel(eval_model, valid_dataloader, local_rank)
            _task_desc = (
                f"Step 5 DDP Task {task_idx} 仅 eval (nproc={world_size}): {args.auxiliary} -> {args.target}"
                if args.eval_only
                else f"Step 5 DDP Task {task_idx} 训练 (nproc={world_size}): {args.auxiliary} -> {args.target}"
            )
            _lines = format_final_results_lines(final, task_description=_task_desc)
            log_final_results_block(train_logger, _lines)
            finalize_run_log(train_logger)
            append_eval_run_summaries(
                final,
                task_idx=int(config.get("task_idx") or 0),
                run_id=str(run_id or ""),
                pipeline="run_d4c_eval_only" if args.eval_only else "run_d4c_train_eval",
                domain_from=args.auxiliary,
                domain_to=args.target,
                log_file=log_path,
                save_file=config.get("save_file"),
                task_description=_task_desc,
            )
            train_logger.info("DONE.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Step 5 主训练：须用 torchrun 启动（DistributedDataParallel）")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件；默认 logs/task{idx}_时间戳.log；D4C_LOG_DIR 指定目录",
    )
    parser.add_argument("--auxiliary", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数，不传则用 config.epochs")
    parser.add_argument("--coef", type=float, default=0.5)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=None, help="全局 batch，不传则用 config；须能被 WORLD_SIZE 整除")
    parser.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数，不传则用 config.num_proc")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="跳过训练，仅加载 checkpoint 并在 valid 上输出 FINAL RESULTS（须已有 model.pth）",
    )
    args = parser.parse_args()
    if "RANK" not in os.environ:
        print(
            "错误: 请使用 torchrun 启动 Step 5，例如:\n"
            "  torchrun --standalone --nproc_per_node=2 run-d4c.py --auxiliary A --target B ...\n"
            "单卡: DDP_NPROC=1 bash sh/run_step5.sh --task N",
            file=sys.stderr,
        )
        sys.exit(1)
    _run_ddp(args)


if __name__ == "__main__":
    main()