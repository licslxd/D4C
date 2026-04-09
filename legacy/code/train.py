"""
LEGACY / NOT PART OF THE NEW MAINLINE

DEPRECATED — not part of the supported DDP pipeline.

This file is kept only for historical / experimental reference. There is no
guarantee of compatibility with the current mainline (python code/d4c.py). Do not use for reproduction.

正式 Step 5：python code/d4c.py step5 …（勿使用本文件）。
"""
import os
import sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_LEGACY_DIR, "..", "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
from base_utils import *
from paths_config import get_data_dir, get_t5_small_dir
from config import get_dataloader_num_workers, get_dataloader_prefetch_factor
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import logging
from transformers import T5Tokenizer
from torch import nn, optim
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from perf_monitor import PerfMonitor
from train_logging import (
    create_run_paths,
    setup_train_logging,
    log_run_header,
    log_config_snapshot,
    format_epoch_training_block,
    log_epoch_training_block,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    append_eval_run_summaries,
)

_t5b = get_t5_small_dir()
_t5_path = _t5b if os.path.exists(_t5b) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)
    
    
class Processor():
    def __init__(self, train):
        self.max_length = 25
        self.train = train
    def __call__(self, sample):
        if self.train: 
            user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
            item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
            raitng = torch.tensor(sample["rating"], dtype=torch.float)
            explanation = sample["explanation"]
            counterfactual = sample["counterfactual"]
            try:
                explanation_idx = tokenizer(explanation, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            except:
                explanation_idx = tokenizer("no explanation", padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            try:
                counterfactual_idx = tokenizer(counterfactual, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            except:
                counterfactual_idx = tokenizer("no counterfactual", padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)
            counterfactual_idx = torch.tensor(counterfactual_idx, dtype=torch.long)
            return {"user_idx": user_idx, "item_idx": item_idx, "rating": raitng, "explanation_idx": explanation_idx, "counterfactual_idx": counterfactual_idx}
        else:
            user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
            item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
            raitng = torch.tensor(sample["rating"], dtype=torch.float)
            explanation = sample["explanation"]
            try:
                explanation_idx = tokenizer(explanation, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            except:
                explanation_idx = tokenizer("no explanation", padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)
            return {"user_idx": user_idx, "item_idx": item_idx, "rating": raitng, "explanation_idx": explanation_idx}
            
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
    
class D4C(nn.Module):
    def __init__(self, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, user_profiles, item_profiles, edited_user_features, edited_item_features):
        super().__init__()
        ntypes = 2  #determine if counterfactual or original samples
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.type_embeddings = nn.Embedding(ntypes, emsize)
        self.user_profiles = nn.Parameter(user_profiles)  
        self.item_profiles = nn.Parameter(item_profiles)
        self.edited_user_features = edited_user_features
        self.edited_item_features = edited_item_features
        self.recommender = PETER_MLP(emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)  
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

    def gather(self, batch, device, train):
        if train: 
            user_idx, item_idx, rating, tgt_output, count_output = batch
            count_output = count_output.to(device)
            count_input = T5_shift_right(count_output)
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device).float()
            tgt_output = tgt_output.to(device)
            tgt_input = T5_shift_right(tgt_output)
            return user_idx, item_idx, rating, tgt_input, tgt_output, count_input, count_output
        else:
            user_idx, item_idx, rating, tgt_output = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device).float()
            tgt_output = tgt_output.to(device)
            tgt_input = T5_shift_right(tgt_output)
            return user_idx, item_idx, rating, tgt_input, tgt_output

    def forward(self, user, item, tgt_input):
        device = user.device
        types = torch.zeros_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1) 
        user_embed = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_embed = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        word_feature = self.word_embeddings(tgt_input)   # in shape (N,seqlen, emsize)
        user_feature = torch.mean(torch.stack((user_embed, user_profile, user_embed*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_embed, item_profile, item_embed*item_profile)), dim=0)
        src = torch.cat([type_feature, user_feature, item_feature, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = generate_count_mask(tgt_input.shape[1], device)
        hidden = self.transformer_encoder(src=src, mask=attn_mask)
        rating = self.recommender(hidden[:,1])
        context_dist = self.hidden2token(hidden[:,2]).unsqueeze(1).repeat(1, tgt_input.shape[1], 1) 
        word_dist = self.hidden2token(hidden[:,3:])
        return rating, context_dist, word_dist

    def take_counterfactuals(self, user, item, count_input):
        device = user.device
        types = torch.ones_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1)
        user_feature = self.edited_user_features[user].unsqueeze(dim=1)
        item_feature = self.edited_item_features[item].unsqueeze(dim=1)
        word_feature = self.word_embeddings(count_input)
        src = torch.cat([type_feature, user_feature, item_feature, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = generate_count_mask(count_input.shape[1], device)
        hidden = self.transformer_encoder(src=src, mask=attn_mask)
        context_dist = self.hidden2token(hidden[:,2]).unsqueeze(1).repeat(1, count_input.shape[1], 1)
        word_dist = self.hidden2token(hidden[:,3:])
        return context_dist, word_dist

    def recommend(self, user, item):
        types = torch.zeros_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1) 
        user_embed = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_embed = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_feature = torch.mean(torch.stack((user_embed, user_profile, user_embed*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_embed, item_profile, item_embed*item_profile)), dim=0)
        src = torch.cat([type_feature, user_feature, item_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden = self.transformer_encoder(src)
        rating = self.recommender(hidden[:,1])  
        return rating

    def generate(self, user, item):
        max_len = 25
        bos_idx = 0
        device = user.device
        batch_size = user.shape[0]
        types = torch.zeros_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1) 
        user_embed = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_embed = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)      
        user_feature = torch.mean(torch.stack((user_embed, user_profile, user_embed*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_embed, item_profile, item_embed*item_profile)), dim=0)
        
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)   # in shape (N,1)
        for i in range(max_len):
            word_feature = self.word_embeddings(decoder_input_ids) 
            src = torch.cat([type_feature, user_feature, item_feature, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)  # in shape: (N, 2+1, emsize)
            attn_mask = generate_count_mask(decoder_input_ids.shape[1], device)
            hidden = self.transformer_encoder(src=src, mask=attn_mask)     # in shape (N, 3, emsize)
            dist = self.hidden2token(hidden).softmax(dim=-1)
            output_id = dist[:,-1,:].topk(1).indices                       # in shape (N, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
        return decoder_input_ids[:,1:]  # removing <BOS>


def trainModel(model, train_dataloader, valid_dataloader, config):
    epochs = config.get("epochs")
    batch_size = config.get("batch_size", "?")
    learning_rate = config.get("learning_rate")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    coef = config.get("coef")
    weight = config.get("weight")
    device = config.get("device")
    ntoken = config.get("ntoken")
    log_file = config.get("log_file")
    save_file = config.get("save_file")
    logger = config.get("logger")
    prev_valid_loss = float("inf")
    enduration = 0
    n_steps = len(train_dataloader)
    train_info = f"[Train] batch_size={batch_size}, epochs={epochs}, steps_per_epoch={n_steps}"
    if logger:
        logger.info(train_info)
    else:
        print(train_info, flush=True)
    train_nw = getattr(train_dataloader, "num_workers", 0)
    valid_nw = getattr(valid_dataloader, "num_workers", 0)
    perf = PerfMonitor(
        device=device,
        log_file=log_file,
        num_proc=config.get("num_proc"),
        train_num_workers=train_nw,
        valid_num_workers=valid_nw,
        training_logger=logger,
    )
    perf.start()
    _model = get_underlying_model(model)
    try:
        for epoch in range(epochs):
            perf.epoch_start()
            model.train()
            avg_loss = 0
            for batch in tqdm(train_dataloader, total=len(train_dataloader)):
                user_idx, item_idx, rating, tgt_input, tgt_output, count_input, count_output = _model.gather(batch, device, True)
                pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input)
                count_context_dist, cont_word_dist = model.take_counterfactuals(user_idx, item_idx, count_input)
                loss_r = _model.rating_loss_fn(pred_rating, rating)
                loss_e = _model.exp_loss_fn(word_dist.view(-1, ntoken), tgt_output.reshape(-1))
                loss_c = _model.exp_loss_fn(context_dist.view(-1, ntoken), tgt_output.reshape(-1))
                loss_c_c = _model.exp_loss_fn(count_context_dist.view(-1, ntoken), count_output.reshape(-1))
                loss_c_w = _model.exp_loss_fn(cont_word_dist.view(-1, ntoken), count_output.reshape(-1))
                loss_con = weight * (loss_c_c + loss_c_w)
                loss_reg = coef * (loss_r + loss_c)
                loss = loss_e + loss_con + loss_reg
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss.item()
            avg_loss /= len(train_dataloader)
            lr_epoch = optimizer.param_groups[0]["lr"]
            rec = perf.epoch_end(epoch + 1, len(train_dataloader), emit_log=False)
            current_valid_loss = validModel(model, valid_dataloader, device)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                bleu_line=None,
            )
            log_epoch_training_block(logger, block)
            if current_valid_loss > prev_valid_loss:
                learning_rate = learning_rate * 0.5
                enduration += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            else:
                torch.save(model.state_dict(), save_file)
            prev_valid_loss = current_valid_loss
            if enduration >= 5:
                break
    finally:
        perf.finish()

def validModel(model, valid_dataloader, device):
    _model = get_underlying_model(model)
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output = _model.gather(batch, device, False)
            pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input)
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
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
            user_idx, item_idx, rating, tgt_input, tgt_output = _model.gather(batch, device, False)
            pred_ratings = model.recommend(user_idx, item_idx)
            pred_exps = model.generate(user_idx, item_idx)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件；默认 logs/task{idx}_时间戳.log，可用环境变量 D4C_LOG_DIR 指定目录",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_file", type=str, default= "model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--coef", type=float, default=0.5)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default="AM_Electronics")
    parser.add_argument("--target", type=str, default="AM_CDs")
    parser.add_argument("--weight", type=float, default=0.01)
    args = parser.parse_args()

    pairs = [("AM_Electronics", "AM_CDs"),
            ("AM_Movies", "AM_CDs"),
            ("AM_CDs", "AM_Electronics"), 
            ("AM_Movies", "AM_Electronics"),
            ("AM_CDs", "AM_Movies"),
            ("AM_Electronics", "AM_Movies"), 
            ("Yelp", "TripAdvisor"),
            ("TripAdvisor", "Yelp")]
    index = pairs.index((args.source, args.target)) + 1
    log_path, run_id = create_run_paths(index, args.log_file)
    _setup = setup_train_logging(
        log_file=log_path,
        task_idx=index,
        rank=0,
        world_size=1,
        run_id=run_id,
    )
    _logger = _setup["logger"]

    config = {"source": args.source,
            "target": args.target, 
            "device": args.device, 
            "log_file": log_path,
            "save_file": args.save_file,
            "learning_rate": args.learning_rate,
            "weight": args.weight,
            "coef": args.coef,
            "epochs": args.epochs,
            "seed": args.seed,
            "batch_size": 128, 
            "ntoken": len(tokenizer),
            "emsize": 768,
            "nhead": 2, 
            "nhid": 2048,
            "nlayers": 2, 
            "dropout": 0.2,
            "task_idx": index,
            "run_id": run_id,
            "logger": _logger,
            }
    train_path = os.path.join(get_data_dir(), "counterfactual", str(index))
    edited_user_features = torch.tensor(np.load(train_path+"/user_features.npy"), dtype=torch.float, device=config.get("device"))
    edited_item_features = torch.tensor(np.load(train_path+"/item_features.npy"), dtype=torch.float, device=config.get("device"))
    train_dataset = load_dataset("csv", data_files={"train": train_path+"/train.csv"})
    train_processor = Processor(train=True)
    encoded_data = train_dataset["train"].map(train_processor, batched=True)
    encoded_data.set_format(type="torch")
    dataset = TensorDataset(encoded_data['user_idx'],
                                encoded_data['item_idx'],
                                encoded_data['rating'],
                                encoded_data['explanation_idx'], 
                                encoded_data['counterfactual_idx'])
    # Legacy 入口：未走 FinalTrainingConfig，但 workers/prefetch 与 config 层 hardware 预设（D4C_HARDWARE_PRESET）及
    # D4C_DATALOADER_WORKERS_* / D4C_PREFETCH_* 等 ENV 共享同一套 getter。
    tw = get_dataloader_num_workers("train")
    vw = get_dataloader_num_workers("valid")
    pin_mem = torch.cuda.is_available()
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size"),
        shuffle=True,
        num_workers=tw,
        pin_memory=pin_mem,
        persistent_workers=tw > 0,
        prefetch_factor=get_dataloader_prefetch_factor(tw, split="train"),
    )

    path = os.path.join(get_data_dir(), config.get('target'))
    eval_dataset = load_dataset("csv", data_files={"valid": path+"/valid.csv", "test": path+"/test.csv"})
    user_profiles = torch.tensor(np.load(path+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
    item_profiles = torch.tensor(np.load(path+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
    eval_processor = Processor(train=False)
    encoded_data = eval_dataset.map(eval_processor, batched=True)
    encoded_data.set_format(type="torch")
    valid_dataset = TensorDataset(encoded_data["valid"]['user_idx'],
                                encoded_data["valid"]['item_idx'],
                                encoded_data["valid"]['rating'],
                                encoded_data["valid"]['explanation_idx'])
                                
    test_dataset = TensorDataset(encoded_data["test"]['user_idx'],
                                encoded_data["test"]['item_idx'],
                                encoded_data["test"]['rating'],
                                encoded_data["test"]['explanation_idx'])
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.get("batch_size"),
        shuffle=True,
        num_workers=vw,
        pin_memory=pin_mem,
        persistent_workers=vw > 0,
        prefetch_factor=get_dataloader_prefetch_factor(vw, split="valid"),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    nuser = max(train_dataset["train"]["user_idx"]) + 1
    nitem = max(train_dataset["train"]["item_idx"]) + 1
    model = D4C(nuser, nitem, 
                config["ntoken"],
                config["emsize"],
                config["nhead"], 
                config["nhid"], 
                config["nlayers"],
                config["dropout"], 
                user_profiles, 
                item_profiles, 
                edited_user_features, 
                edited_item_features).to(config["device"])

    log_run_header(
        _logger,
        {
            "run_id": run_id,
            "task_idx": index,
            "rank": 0,
            "world_size": 1,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(config["device"]),
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "save_file": os.path.abspath(str(config["save_file"])),
            "log_file": os.path.abspath(log_path),
            "source": config["source"],
            "target": config["target"],
        },
    )
    log_config_snapshot(_logger, config)

    trainModel(model, train_dataloader, valid_dataloader, config)
    model.load_state_dict(torch.load(config.get("save_file")))
    import time as _time
    _eval_t0 = _time.time()
    _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final = evalModel(model, test_dataloader, config.get("device"))
    _eval_elapsed = _time.time() - _eval_t0
    _eval_min, _eval_sec = divmod(int(_eval_elapsed), 60)
    _td = f"train.py 单卡 D4C Task {index}: {args.source} -> {args.target}"
    _lines = format_final_results_lines(final, task_description=_td, start_time=_eval_start_str)
    _lines.append(f"Eval elapsed: {_eval_min}m {_eval_sec}s ({_eval_elapsed:.1f}s)")
    log_final_results_block(_logger, _lines)
    finalize_run_log(_logger)
    append_eval_run_summaries(
        final,
        task_idx=index,
        run_id=run_id,
        pipeline="train_d4c",
        domain_from=args.source,
        domain_to=args.target,
        log_file=log_path,
        save_file=config.get("save_file"),
        task_description=_td,
        start_time=_eval_start_str,
        eval_elapsed=_eval_elapsed,
    )
    _logger.info("DONE.")