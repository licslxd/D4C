import os
import sys

# 必须在首次 import nltk 之前设置，否则 METEOR / word_tokenize 会触发 nltk.download 联网
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_NLTK_LOCAL = os.path.join(_REPO_ROOT, "pretrained_models", "nltk_data")
if os.path.isdir(_NLTK_LOCAL):
    os.environ.setdefault("NLTK_DATA", _NLTK_LOCAL)

from rouge import rouge
from nltk import word_tokenize
from bleu import compute_bleu
import logging
import numpy as np
import nltk

_logger = logging.getLogger(__name__)


class _FilterBertEmptyStderr:
    """bert_score 对空候选/参考逐条 print 到 stderr，刷屏；仅过滤这两类提示行。"""

    _SKIP_SUBSTR = (
        "Empty candidate sentence",
        "Empty reference sentence",
    )

    def __init__(self, real):
        self._real = real

    def write(self, s):
        if s and any(sub in s for sub in self._SKIP_SUBSTR):
            return len(s)
        return self._real.write(s)

    def flush(self):
        return self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)

if os.path.isdir(_NLTK_LOCAL) and _NLTK_LOCAL not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_LOCAL)


def _patch_nltk_download_offline_only():
    """HF evaluate 的 METEOR 在 _download_and_prepare 里无条件调用 nltk.download，离线仍会联网。

    若本地已有对应语料（与 meteor.py 中 wordnet / punkt / punkt_tab / omw-1.4 一致），则直接返回成功，
    避免 [nltk_data] urlopen / Name or service not known。
    """
    _FIND = {
        "wordnet": "corpora/wordnet",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "omw-1.4": "corpora/omw-1.4",
    }
    if getattr(nltk, "_d4c_offline_dl_patch", False):
        return
    _orig = nltk.download

    def _wrapped(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0] in _FIND:
            try:
                nltk.data.find(_FIND[args[0]])
                return True
            except LookupError:
                return False
        return _orig(*args, **kwargs)

    nltk.download = _wrapped
    nltk._d4c_offline_dl_patch = True


from torch import nn
import torch
import math


def get_underlying_model(model):
    """当 model 被 DistributedDataParallel 包装时返回原始模块。"""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model

def T5_shift_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert decoder_start_token_id is not None, (
        "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
        " See T5 docs for more information"
    )
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def compute_bleu1234_only(predictions, references):
    """
    仅计算 BLEU-1~4（与 evaluate_text 中词级 BLEU 口径一致），用于训练阶段按验证集 BLEU-4 选模，避免 METEOR/BERT 开销。
    """
    predictions_tokens = [word_tokenize(prediction) for prediction in predictions]
    references_tokens = [word_tokenize(reference) for reference in references]
    formatted_ref = [[ref] for ref in references_tokens]
    out = {}
    for order in (1, 2, 3, 4):
        try:
            bleu_n, _, _, _, _, _ = compute_bleu(
                formatted_ref, predictions_tokens, max_order=order, smooth=False
            )
            out[str(order)] = round(bleu_n * 100, 2)
        except Exception:
            out[str(order)] = 0.0
    return out


def evaluate_text(predictions, references):
    """
    Example:
        >>> predictions = ["good day", "need to work"]
        >>> references = ["nice day", "work from home"]
        >>> evlauate_text(predictions, references)
    """
    # compute bleu
    # compute rouge
    # compute distinct（语料级 n-gram distinct，×100；论文主表 DIST-1/DIST-2 口径；
    #   与 d4c_eval_metrics.extended_text_metrics_bundle 中的 distinct 定义不同，勿混读）
    # compute meteor

    def distinct_score(sentences, n):
        sentences = [word_tokenize(sentence) for sentence in sentences]
        unique_ngrams = set()
        total_ngrams = 0

        for sentence in sentences:
            ngrams = [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
            unique_ngrams.update(ngrams)
            total_ngrams += len(ngrams)

        distinct_score = len(unique_ngrams) / total_ngrams
        return distinct_score
    # dist score
    try:
        dist1 = round(distinct_score(predictions, 1) * 100, 2)
    except:
        dist1 = 0
    try:
        dist2 = round(distinct_score(predictions, 2) * 100, 2)
    except:
        dist2 = 0
    
    # bleu score
    predictions_tokens = [word_tokenize(prediction) for prediction in predictions]
    references_tokens = [word_tokenize(reference) for reference in references]
    formatted_ref = [[ref] for ref in references_tokens]
    try:
        bleu1, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=1, smooth=False)
        bleu1 = round(bleu1*100, 2)
    except:
        bleu1 = 0
    try:
        bleu2, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=2, smooth=False)
        bleu2 = round(bleu2*100, 2)
    except:
        bleu2 = 0
    try:
        bleu3, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=3, smooth=False)
        bleu3 = round(bleu3*100, 2)
    except:
        bleu3 = 0
    try:
        bleu4, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=4, smooth=False)
        bleu4 = round(bleu4*100,2)
    except:
        bleu4 = 0
    
    # rouge score
    score = rouge(predictions, references)
    rouge_s = {k: round(v * 100, 2) for (k, v) in score.items()}
    
    
    # meteor score (离线：使用本地 cache_dir，无缓存时跳过)
    try:
        import evaluate
        _cache = os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "evaluate_meteor")
        os.makedirs(_cache, exist_ok=True)
        # 必须早于 evaluate.load：meteor 指标会无条件 nltk.download，需改为仅检测本地
        _patch_nltk_download_offline_only()
        # 优先从 hf_cache 中的本地 meteor.py 加载，避免 evaluate.load("meteor") 在离线时打印 Hub 解析提示
        _meteor_dir = os.path.join(
            _REPO_ROOT, "pretrained_models", "hf_cache", "modules", "evaluate_modules", "metrics", "evaluate-metric--meteor"
        )
        _meteor_script = None
        if os.path.isdir(_meteor_dir):
            for _entry in sorted(os.listdir(_meteor_dir)):
                _candidate = os.path.join(_meteor_dir, _entry, "meteor.py")
                if os.path.isfile(_candidate):
                    _meteor_script = _candidate
                    break
        if _meteor_script:
            meteor = evaluate.load(_meteor_script, cache_dir=_cache)
        else:
            meteor = evaluate.load("meteor", cache_dir=_cache)
        meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
        meteor_score = round(meteor_score * 100, 2)
    except Exception:
        _logger.exception("METEOR failed")
        meteor_score = 0.0

    # bert_score (离线：使用本地 bertscore 脚本 + hf_cache 中的 microsoft/deberta-xlarge-mnli)
    # SKIP_BERTSCORE=1 时跳过（加载 deberta-xlarge-mnli 并逐条推理是 eval 最大瓶颈）
    _skip_bert = os.environ.get("SKIP_BERTSCORE", "1") == "1"
    if not _skip_bert:
      try:
        import evaluate
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _hf_cache = os.path.join(_base, "pretrained_models", "hf_cache")
        _bs_script = os.path.join(_base, "pretrained_models", "evaluate_bertscore", "bertscore.py")
        _deberta_snapshot = os.path.join(_hf_cache, "models--microsoft--deberta-xlarge-mnli", "snapshots", "5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c")
        _model_type = _deberta_snapshot if os.path.exists(_deberta_snapshot) else "microsoft/deberta-xlarge-mnli"
        if _model_type == _deberta_snapshot:
            for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
                if _var not in os.environ:
                    os.environ[_var] = _hf_cache
                    break
        bertscore = evaluate.load(_bs_script)
        try:
            _bert_bs = int(os.environ.get("BERTSCORE_BATCH_SIZE", "64"))
        except ValueError:
            _bert_bs = 64
        _bert_bs = max(1, _bert_bs)
        _n_empty_cand = sum(1 for p in predictions if not (p or "").strip())
        _n_empty_ref = sum(1 for r in references if not (r or "").strip())
        if _n_empty_cand or _n_empty_ref:
            _logger.warning(
                "evaluate_text: 空候选句 %d 条、空参考句 %d 条；BERTScore 按 0 计（bert_score 逐条 stderr 已抑制）",
                _n_empty_cand,
                _n_empty_ref,
            )
        _prev_err = sys.stderr
        sys.stderr = _FilterBertEmptyStderr(_prev_err)
        try:
            bert_score = bertscore.compute(
                predictions=predictions,
                references=references,
                model_type=_model_type,
                num_layers=48 if _model_type == _deberta_snapshot else None,
                lang="en",
                batch_size=_bert_bs,
            )
        finally:
            sys.stderr = _prev_err
        bert_score = round(np.mean(bert_score["f1"])*100, 2)
      except Exception as e:
        import warnings
        warnings.warn(f"BERTScore 计算失败，已回退为 0.0。原因: {e}", UserWarning)
        bert_score = 0.0
    result = {
            "rouge": {"1":rouge_s["rouge_1/f_score"], "2":rouge_s["rouge_2/f_score"], "l":rouge_s["rouge_l/f_score"]},
            "bleu": {"1":bleu1, "2":bleu2, "3":bleu3, "4":bleu4}, 
            "dist": {"1":dist1, "2":dist2},
            "meteor": meteor_score}
    if not _skip_bert:
        result["bert"] = bert_score
    return result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]`` (batch_first)
        """
        if x.dim() == 3:
            # batch_first: (N, L, D) -> add pe (1, L, D)
            seq_len = x.size(1)
            x = x + self.pe[:seq_len].transpose(0, 1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_count_mask(tgt_len, device):
    src_len = 3
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    mask[0, 2] = False
    mask[1, 2] = False
    return mask

def generate_new_mask(tgt_len, device):
    src_len = 3
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    mask[0, 2] = False
    mask[1, 2] = False
    return mask

def generate_domain_mask(tgt_len, device):
    src_len = 5  # Set src_len to 5
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[:src_len, :src_len] = False  # No masking for the first 5 positions
    return mask

def generate_peter_mask(tgt_len, device):
    src_len = 2
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    return mask

def generate_square_mask(seqlen, device):
    mask = torch.triu(torch.ones((seqlen, seqlen), device=device), diagonal=1) == 1
    return mask

def generate_peter_noui_mask(tgt_len, device):
    src_len = 2
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    mask[2:,:2] = True
    return mask

def compute_entropy(generated_dist):
    log_probabilities = torch.log(generated_dist + 1e-9)  # Ensure no log(0) by adding a small epsilon
    entropies = -torch.sum(generated_dist * log_probabilities, dim=-1)  # Shape (N, seqlen)
    sample_entropies = torch.mean(entropies, dim=-1)  # Shape (N,)
    return sample_entropies

def filter_by_entropy(entropy_values, percentile=0.75):
    entropy_tensor = torch.tensor(entropy_values)
    threshold = torch.quantile(entropy_tensor, percentile)
    filtered_indices = torch.where(entropy_tensor <= threshold)[0]
    return filtered_indices.tolist()
