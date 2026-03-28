# -*- coding: utf-8 -*-
"""
训练入口的运行期覆盖（来自 argparse），显式在内存中传递，不写回 os.environ。

供 ``build_resolved_training_config`` / ``collect_training_runtime_overrides_from_args`` 使用；不写 os.environ。
"""

from __future__ import annotations

from typing import Any, Dict


def collect_training_runtime_overrides_from_args(args: Any) -> Dict[str, Any]:
    """
    从训练 CLI namespace 收集「非 None」字段，键名稳定，便于日志与后续统一 resolve。

    注意：不包含 torchrun 的 LOCAL_RANK 等；仅训练超参/调度相关。
    """
    out: Dict[str, Any] = {}
    pairs = [
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("coef", "coef"),
        ("adv", "adv"),
        ("num_proc", "num_proc"),
        ("per_device_batch_size", "per_device_batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("scheduler_initial_lr", "scheduler_initial_lr"),
        ("learning_rate", "learning_rate"),
        ("warmup_steps", "warmup_steps"),
        ("warmup_ratio", "warmup_ratio"),
        ("warmup_epochs", "warmup_epochs"),
        ("min_lr_ratio", "min_lr_ratio"),
        ("lr_scheduler", "lr_scheduler"),
        ("eval_batch_size", "eval_batch_size"),
        ("quick_eval_max_samples", "quick_eval_max_samples"),
        ("full_eval_every", "full_eval_every"),
        ("early_stop_patience_full", "early_stop_patience_full"),
        ("early_stop_patience_loss", "early_stop_patience_loss"),
        ("min_epochs", "min_epochs"),
        ("early_stop_patience", "early_stop_patience"),
        ("bleu4_max_samples", "bleu4_max_samples"),
        ("adversarial_start_epoch", "adversarial_start_epoch"),
        ("adversarial_warmup_epochs", "adversarial_warmup_epochs"),
        ("adversarial_coef_target", "adversarial_coef_target"),
    ]
    for key, attr in pairs:
        if not hasattr(args, attr):
            continue
        v = getattr(args, attr)
        if v is not None:
            out[key] = v
    return out
