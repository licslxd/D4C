"""
手写解码控制器：与 HF generate 解耦，统一 nucleus / greedy + 尾部退火 + 终端约束。

由 ``step5_engine.Model`` 调用；有效参数字典见 ``build_generate_kwargs_effective_v2``。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace as dc_replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


PAPER_DECODE_CONTROLLER_SCHEMA = "d4c_decode_controller/1.0"


@dataclass
class GenerateConfig:
    """单次生成运行时的解码配置（由 FinalTrainingConfig / yaml 填充）。"""

    strategy: str = "greedy"
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 0
    min_len: int = 0
    soft_max_len: int = 0
    hard_max_len: int = 25
    eos_boost_start: int = 9999
    eos_boost_value: float = 0.0
    tail_temperature: float = -1.0
    tail_top_p: float = -1.0
    forbid_eos_after_open_quote: bool = True
    forbid_eos_after_open_bracket: bool = True
    forbid_bad_terminal_tokens: bool = True
    bad_terminal_token_ids: Tuple[int, ...] = ()
    token_repeat_window: int = 4
    token_repeat_max: int = 2
    decode_seed: Optional[int] = None


def merge_generate_config_with_override(base: GenerateConfig, override: Mapping[str, Any]) -> GenerateConfig:
    """将 override 合并进 base（仅允许 GenerateConfig 已有字段）；用于单次 generate 临时覆盖，不回写模型状态。"""
    if not override:
        return base
    names = {f.name for f in fields(GenerateConfig)}
    unknown = set(override.keys()) - names
    if unknown:
        raise ValueError(f"cfg_override 含未知字段 {sorted(unknown)}；允许: {sorted(names)}")
    return dc_replace(base, **dict(override))


def coerce_generate_cfg_override(
    base: GenerateConfig, cfg_override: Optional[Union[GenerateConfig, Mapping[str, Any]]]
) -> Optional[GenerateConfig]:
    """与 generate / generate_with_token_logprobs 对齐：None、GenerateConfig 或 dict 映射为单次解码用 GenerateConfig。"""
    if cfg_override is None:
        return None
    if isinstance(cfg_override, GenerateConfig):
        return cfg_override
    return merge_generate_config_with_override(base, cfg_override)


@dataclass
class GenerationState:
    decoder_input_ids: torch.Tensor
    active: torch.Tensor
    step: int
    recent_tokens: List[List[int]] = field(default_factory=list)


def _effective_tail_scalar(base: float, tail: float, step: int, soft: int, hard: int) -> float:
    if tail < 0 or hard <= soft or step < soft:
        return float(base)
    alpha = (float(step) - float(soft)) / float(max(1, hard - soft))
    alpha = min(1.0, max(0.0, alpha))
    return float(base) * (1.0 - alpha) + float(tail) * alpha


def _unbalanced_delimiters(text: str) -> bool:
    """轻量括号/引号未闭合检测（字符级，供 forbid_eos）。"""
    s = text or ""
    p = 0
    b = 0
    quote = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            quote = not quote
            continue
        if quote:
            continue
        if ch == "(":
            p += 1
        elif ch == ")":
            p = max(0, p - 1)
        elif ch == "[":
            b += 1
        elif ch == "]":
            b = max(0, b - 1)
    return quote or p > 0 or b > 0


def prepare_logits(hidden_last: torch.Tensor, hidden2token: torch.nn.Module) -> torch.Tensor:
    return hidden2token(hidden_last)


def apply_repetition_penalty_logits(
    logits: torch.Tensor, decoder_input_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    appeared = torch.zeros(logits.size(0), logits.size(1), device=logits.device, dtype=torch.bool)
    appeared.scatter_(1, decoder_input_ids, True)
    adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(appeared, adjusted, logits)


def apply_no_repeat_ngram_logits(
    logits: torch.Tensor, decoder_input_ids: torch.Tensor, ngram_size: int
) -> None:
    if ngram_size <= 0:
        return
    B, V = logits.shape
    for b in range(B):
        seq = decoder_input_ids[b].tolist()
        cur_len = len(seq)
        if cur_len + 1 < ngram_size:
            continue
        generated = set()
        for i in range(cur_len - ngram_size + 1):
            generated.add(tuple(seq[i : i + ngram_size]))
        prefix = tuple(seq[cur_len - (ngram_size - 1) : cur_len])
        banned = {ng[-1] for ng in generated if ng[:-1] == prefix}
        if not banned:
            continue
        idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
        logits[b, idx] = torch.finfo(logits.dtype).min


def apply_token_repeat_suppression(
    logits: torch.Tensor,
    recent_rows: Sequence[Sequence[int]],
    *,
    window: int,
    max_same: int,
) -> None:
    if window <= 0 or max_same <= 0:
        return
    B = logits.size(0)
    for b in range(B):
        hist = list(recent_rows[b])[-window:]
        if len(hist) < max_same:
            continue
        t = hist[-1]
        if sum(1 for x in hist if x == t) >= max_same:
            logits[b, t] = torch.finfo(logits.dtype).min


def apply_min_len_eos_mask(
    logits: torch.Tensor,
    *,
    eos_id: int,
    gen_so_far: int,
    min_len: int,
) -> None:
    if eos_id < 0 or min_len <= 0:
        return
    if gen_so_far < min_len:
        logits[:, eos_id] = torch.finfo(logits.dtype).min


def apply_unbalanced_delimiter_eos_mask(
    logits: torch.Tensor,
    *,
    eos_id: int,
    decoded_texts: Sequence[str],
    cfg: GenerateConfig,
) -> None:
    if eos_id < 0 or not decoded_texts:
        return
    if not (cfg.forbid_eos_after_open_bracket or cfg.forbid_eos_after_open_quote):
        return
    for b in range(logits.size(0)):
        t = decoded_texts[b] if b < len(decoded_texts) else ""
        if _unbalanced_delimiters(t):
            logits[b, eos_id] = torch.finfo(logits.dtype).min


def forbid_eos_if_bad_tail_token(
    logits: torch.Tensor,
    *,
    eos_id: int,
    tail_token_ids: torch.Tensor,
    bad_ids: Tuple[int, ...],
) -> None:
    if eos_id < 0 or not bad_ids:
        return
    # tail_token_ids: (B,)
    for b in range(logits.size(0)):
        tid = int(tail_token_ids[b].item())
        if tid in bad_ids:
            logits[b, eos_id] = torch.finfo(logits.dtype).min


def apply_sampling_schedule(
    cfg: GenerateConfig, step: int
) -> Tuple[float, float]:
    hard = max(1, int(cfg.hard_max_len))
    soft = int(cfg.soft_max_len)
    if soft <= 0:
        soft = max(1, int(hard * 0.65))
    soft = min(soft, hard - 1) if hard > 1 else 1
    t_base = float(cfg.temperature)
    p_base = float(cfg.top_p)
    t_tail = float(cfg.tail_temperature) if cfg.tail_temperature >= 0 else t_base
    p_tail = float(cfg.tail_top_p) if cfg.tail_top_p >= 0 else p_base
    eff_t = _effective_tail_scalar(t_base, t_tail, step, soft, hard)
    eff_p = _effective_tail_scalar(p_base, p_tail, step, soft, hard)
    eff_t = max(eff_t, 1e-8)
    eff_p = min(1.0, max(1e-6, eff_p))
    return eff_t, eff_p


def apply_eos_boost(
    logits: torch.Tensor,
    *,
    eos_id: int,
    step: int,
    cfg: GenerateConfig,
) -> None:
    if eos_id < 0 or cfg.eos_boost_value == 0.0:
        return
    if step >= int(cfg.eos_boost_start):
        logits[:, eos_id] = logits[:, eos_id] + float(cfg.eos_boost_value)


def sample_next_token(
    logits: torch.Tensor,
    *,
    strategy: str,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (output_id (B,1), entropy (B,), log_probs 分布用于 gather)。"""
    B = logits.size(0)
    device = logits.device
    dtype = logits.dtype
    st = strategy.lower()
    if st == "nucleus":
        logits_scaled = logits / temperature
        log_probs = F.log_softmax(logits_scaled, dim=-1)
        probs = F.softmax(logits_scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        nucleus = sorted_probs.masked_fill(mask, 0.0)
        nucleus = nucleus / (nucleus.sum(dim=-1, keepdim=True) + 1e-12)
        ent = -(nucleus * nucleus.clamp_min(1e-12).log()).sum(dim=-1)
        sampled_inner = torch.multinomial(nucleus, 1, generator=generator)
        next_ids = torch.gather(sorted_idx, -1, sampled_inner)
        return next_ids, ent, log_probs
    log_probs = F.log_softmax(logits, dim=-1)
    next_ids = logits.argmax(dim=-1, keepdim=True)
    ent = torch.zeros(B, device=device, dtype=dtype)
    return next_ids, ent, log_probs


def update_generation_state(
    state: GenerationState,
    output_id: torch.Tensor,
    eos_id: int,
) -> None:
    state.decoder_input_ids = torch.cat([state.decoder_input_ids, output_id], dim=-1)
    state.step += 1
    rows = output_id.squeeze(-1).tolist()
    for b, tid in enumerate(rows):
        if b >= len(state.recent_tokens):
            state.recent_tokens.append([])
        state.recent_tokens[b].append(int(tid))
    if eos_id >= 0:
        active = output_id.squeeze(-1) != eos_id
        state.active = state.active & active


def build_candidate_generation_specs(
    base: GenerateConfig,
    family: str,
    *,
    k_cli: int,
    include_diverse: bool,
) -> List[Tuple[str, Optional[GenerateConfig]]]:
    """返回 (family_tag, cfg_override|None)；None 表示使用模型当前默认 GenerateConfig。"""
    fam = (family or "balanced").strip().lower()
    out: List[Tuple[str, Optional[GenerateConfig]]] = []
    if fam == "mixed":
        for _ in range(2):
            c = dc_replace(
                base,
                temperature=max(1e-8, float(base.temperature) * 0.82),
                top_p=min(1.0, float(base.top_p) * 0.92),
            )
            out.append(("conservative", c))
        for _ in range(2):
            out.append(("balanced", None))
        if include_diverse:
            c = dc_replace(
                base,
                temperature=float(base.temperature) * 1.1,
                top_p=min(0.98, float(base.top_p) * 1.03),
            )
            out.append(("diverse", c))
        return out
    k = max(1, int(k_cli))
    for j in range(k):
        if fam == "conservative":
            c = dc_replace(
                base,
                temperature=max(1e-8, float(base.temperature) * (0.88 + 0.02 * j)),
                top_p=min(1.0, float(base.top_p) * (0.94 + 0.005 * j)),
            )
            out.append(("conservative", c))
        elif fam == "diverse":
            c = dc_replace(
                base,
                temperature=float(base.temperature) * (1.02 + 0.03 * j),
                top_p=min(0.99, float(base.top_p) + 0.02 * j),
            )
            out.append(("diverse", c))
        else:
            c = dc_replace(
                base,
                temperature=max(1e-8, float(base.temperature) * (0.97 + 0.02 * (j - k / 2.0))),
            )
            out.append(("balanced", c))
    return out


def build_generate_kwargs_effective_v2(cfg: GenerateConfig, *, eos_token_id: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schema_version": PAPER_DECODE_CONTROLLER_SCHEMA,
        "strategy": cfg.strategy,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "repetition_penalty": cfg.repetition_penalty,
        "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
        "min_len": cfg.min_len,
        "soft_max_len": cfg.soft_max_len,
        "hard_max_len": cfg.hard_max_len,
        "eos_boost_start": cfg.eos_boost_start,
        "eos_boost_value": cfg.eos_boost_value,
        "tail_temperature": cfg.tail_temperature,
        "tail_top_p": cfg.tail_top_p,
        "forbid_eos_after_open_quote": cfg.forbid_eos_after_open_quote,
        "forbid_eos_after_open_bracket": cfg.forbid_eos_after_open_bracket,
        "forbid_bad_terminal_tokens": cfg.forbid_bad_terminal_tokens,
        "bad_terminal_token_ids": list(cfg.bad_terminal_token_ids),
        "token_repeat_window": cfg.token_repeat_window,
        "token_repeat_max": cfg.token_repeat_max,
        "decode_seed": cfg.decode_seed,
    }
    if eos_token_id >= 0:
        out["eos_token_id"] = eos_token_id
    return out
