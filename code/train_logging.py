# -*- coding: utf-8 -*-
"""
训练/评估统一日志：标准 logging、按 run 分文件、DDP 仅 rank0 写文件。
环境变量：
  D4C_LOG_DIR      日志目录（默认 <项目根>/logs）
  D4C_CONSOLE_LEVEL 控制台级别，默认 INFO（非 rank0 恒为 WARNING）
  D4C_FILE_LEVEL    文件级别，默认 DEBUG（仅 rank0 有 FileHandler）
  D4C_MIRROR_LOG=1  同时镜像到 code/log.out（旧行为，默认关闭）
  D4C_LOG_PRETTY=0  关闭多行缩进 JSON（默认开启 RUN_META / RUN_CONFIG 多行缩进，便于阅读）
  D4C_LOG_STRUCTURED_CONSOLE=1  结构化块（RUN_*）同时打到控制台；默认仅写入日志文件，避免与 FileHandler
                                并存且 stderr 被 tee/重定向到同一文件时出现重复行。
  D4C_EVAL_SUMMARY=0           关闭 eval 自动汇总（默认开启）：写入每任务与全局共 6 个文件（txt/jsonl/csv）。
  D4C_EVAL_SUMMARY_GLOBAL_DIR  全局汇总目录（默认 <项目根>/log），内含 eval_runs_all.{txt,jsonl,csv}
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from paths_config import D4C_ROOT, get_log_task_dir

LOGGER_NAME = "d4c"
RUN_END_LINE = "========== RUN END =========="


class _StreamSuppressFileOnlyFilter(logging.Filter):
    """带 d4c_file_only 的记录不输出到 StreamHandler，仍由 FileHandler 写入（若存在）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "d4c_file_only", False)


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{rnd}"


def _log_dir() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("D4C_LOG_DIR") or os.path.join(D4C_ROOT, "logs")))


def create_run_paths(
    task_idx: int,
    explicit_log_file: Optional[str] = None,
) -> Tuple[str, str]:
    """
    返回 (log_path, run_id)。
    explicit_log_file 为有效路径且不是占位符 log.out 时，直接使用该路径；否则生成
    logs/task{task_idx}_{YYYYMMDD_HHMMSS}.log
    """
    run_id = generate_run_id()
    ex = (explicit_log_file or "").strip()
    if ex and ex != "log.out":
        return os.path.abspath(os.path.expanduser(ex)), run_id
    base = _log_dir()
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"task{task_idx}_{ts}.log")
    return path, run_id


def _parse_level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def _pretty_log_enabled() -> bool:
    """默认 True（多行缩进）；设 D4C_LOG_PRETTY=0/false/no/off 为单行 JSON。"""
    v = os.environ.get("D4C_LOG_PRETTY", "").strip().lower()
    if not v:
        return True
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def _structured_console_enabled() -> bool:
    """为 True 时结构化 RUN_* 也镜像到控制台（默认 False）。"""
    v = os.environ.get("D4C_LOG_STRUCTURED_CONSOLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _structured_log_extra(logger: logging.Logger) -> Dict[str, Any]:
    """
    若存在 FileHandler 且未要求控制台镜像，则标记 d4c_file_only，由 StreamHandler 过滤器抑制，
    避免 tee/2>&1 与文件为同一路径时大块 JSON 重复。
    """
    if _structured_console_enabled():
        return {}
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return {"d4c_file_only": True}
    return {}


def _json_safe_sorted_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k in sorted(data.keys()):
        v = data[k]
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = repr(v)
    return safe


def setup_train_logging(
    *,
    log_file: Optional[str],
    task_idx: int,
    rank: int = 0,
    world_size: int = 1,
    run_id: Optional[str] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> Dict[str, Any]:
    """
    配置名为 d4c 的 logger：控制台 INFO（rank0）/ WARNING（其他 rank）；文件仅 rank0。
    """
    if run_id is None:
        run_id = generate_run_id()
    console_level = _parse_level(console_level or os.environ.get("D4C_CONSOLE_LEVEL", "INFO"))
    file_level = _parse_level(file_level or os.environ.get("D4C_FILE_LEVEL", "DEBUG"))

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    if rank == 0:
        sh.setLevel(console_level)
    else:
        sh.setLevel(logging.WARNING)
    logger.addHandler(sh)

    log_path = log_file
    if rank == 0 and log_path:
        sh.addFilter(_StreamSuppressFileOnlyFilter())
        d = os.path.dirname(os.path.abspath(log_path))
        if d:
            os.makedirs(d, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return {"logger": logger, "run_id": run_id, "log_path": log_path, "rank": rank, "world_size": world_size}


def log_run_header(logger: logging.Logger, meta: Dict[str, Any]) -> None:
    """RUN_META：默认多行缩进 JSON（键有序）；D4C_LOG_PRETTY=0 时为单行。"""
    safe = _json_safe_sorted_dict(meta)
    pretty = _pretty_log_enabled()
    payload = json.dumps(safe, ensure_ascii=False, indent=2 if pretty else None)
    extra = _structured_log_extra(logger)
    if pretty:
        logger.info("RUN_META\n%s", payload, extra=extra)
    else:
        logger.info("RUN_META %s", payload, extra=extra)


def log_config_snapshot(
    logger: logging.Logger,
    config: Dict[str, Any],
    *,
    exclude_keys: Tuple[str, ...] = ("logger",),
) -> None:
    """RUN_CONFIG：与 RUN_META 相同展示规则；不可 JSON 序列化的值用 repr。"""
    trimmed = {k: v for k, v in config.items() if k not in exclude_keys}
    safe = _json_safe_sorted_dict(trimmed)
    pretty = _pretty_log_enabled()
    payload = json.dumps(safe, ensure_ascii=False, indent=2 if pretty else None)
    extra = _structured_log_extra(logger)
    if pretty:
        logger.info("RUN_CONFIG\n%s", payload, extra=extra)
    else:
        logger.info("RUN_CONFIG %s", payload, extra=extra)


def log_run_snapshot(
    logger: logging.Logger,
    meta: Dict[str, Any],
    config: Dict[str, Any],
    *,
    exclude_keys: Tuple[str, ...] = ("logger",),
) -> None:
    """单次 RUN_SNAPSHOT：meta（运行/CLI 上下文）+ config（完整训练配置），避免重复键与双份 RUN_*。"""
    trimmed = {k: v for k, v in config.items() if k not in exclude_keys}
    body = {"meta": _json_safe_sorted_dict(meta), "config": _json_safe_sorted_dict(trimmed)}
    pretty = _pretty_log_enabled()
    payload = json.dumps(body, ensure_ascii=False, indent=2 if pretty else None)
    extra = _structured_log_extra(logger)
    if pretty:
        logger.info("RUN_SNAPSHOT\n%s", payload, extra=extra)
    else:
        logger.info("RUN_SNAPSHOT %s", payload, extra=extra)


def format_epoch_line(
    epoch: int,
    time_str: str,
    lr: float,
    train_loss: float,
    valid_loss: Optional[float] = None,
    adv_loss: Optional[float] = None,
) -> str:
    parts = [
        f"epoch={epoch}",
        f"time={time_str}",
        f"lr={lr:.6g}",
        f"train_loss={train_loss:.4f}",
    ]
    if valid_loss is not None:
        parts.append(f"valid_loss={valid_loss:.4f}")
    if adv_loss is not None:
        parts.append(f"adv_loss={adv_loss:.4f}")
    return " | ".join(parts)


def format_final_results_lines(
    final: Dict[str, Any],
    *,
    task_description: Optional[str] = None,
) -> List[str]:
    """构建 FINAL RESULTS 文本行（无 log 前缀；指标行用制表符缩进）。

    task_description: 可选，在评估结果块最上方增加一行「任务说明：…」（位于 FINAL RESULTS 分隔线之上）。
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tab = "\t"
    lines: List[str] = []
    if task_description:
        lines.append(f"任务说明：{task_description}")
    lines.extend(
        [
            "------------------------------------------FINAL RESULTS------------------------------------------",
            current_time,
            "[Recommendation]",
        ]
    )
    lines.extend(
        [
            f"{tab}MAE = {final['recommendation']['mae']} | RMSE = {final['recommendation']['rmse']} ",
            "[Explanation]",
            f"{tab}ROUGE: {final['explanation']['rouge']['1']}, {final['explanation']['rouge']['2']}, {final['explanation']['rouge']['l']} ",
            f"{tab}BLEU: {final['explanation']['bleu']['1']}, {final['explanation']['bleu']['2']}, {final['explanation']['bleu']['3']}, {final['explanation']['bleu']['4']} ",
            f"{tab}DIST: {final['explanation']['dist']['1']}, {final['explanation']['dist']['2']},",
            f"{tab}METEOR: {final['explanation']['meteor']} ",
            f"{tab}BERT: {final['explanation']['bert']} ",
        ]
    )
    return lines


def _write_plain_log_block(logger: Optional[logging.Logger], text: str) -> None:
    """写入多行纯文本（不经 Formatter），同步到 FileHandler 与控制台 StreamHandler。"""
    if not text.endswith("\n"):
        text = text + "\n"
    if logger is None:
        print(text, end="", flush=True)
        return
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                h.stream.write(text)
                h.flush()
            except Exception:
                pass
        elif isinstance(h, logging.StreamHandler):
            try:
                h.stream.write(text)
                h.flush()
            except Exception:
                pass


def log_final_results_block(logger: Optional[logging.Logger], lines: list) -> None:
    """FINAL RESULTS 多行块（纯文本，无时间戳/级别前缀；写入文件 + rank0 控制台）。"""
    text = "\n".join(lines)
    _write_plain_log_block(logger, text)


def finalize_run_log(logger: Optional[logging.Logger], extra: Optional[str] = None) -> None:
    _write_plain_log_block(logger, RUN_END_LINE + "\n")
    if logger is not None:
        if extra:
            logger.info("%s", extra)
    elif extra:
        print(extra, flush=True)


def broadcast_run_paths_ddp(
    log_path: Optional[str],
    run_id: Optional[str],
    rank: int,
) -> Tuple[str, str]:
    """分布式初始化后由 rank0 生成路径并广播到各 rank。"""
    import torch.distributed as dist

    if not dist.is_initialized():
        return log_path or "", run_id or ""
    obj = [log_path, run_id] if rank == 0 else [None, None]
    dist.broadcast_object_list(obj, src=0)
    return obj[0], obj[1]


# --- eval 结果自动汇总（每任务 + 全局，纯文本 / JSONL / CSV）---

_EVAL_SUMMARY_TXT = "eval_runs.txt"
_EVAL_SUMMARY_JSONL = "eval_runs.jsonl"
_EVAL_SUMMARY_CSV = "eval_runs.csv"
_EVAL_SUMMARY_GLOBAL_TXT = "eval_runs_all.txt"
_EVAL_SUMMARY_GLOBAL_JSONL = "eval_runs_all.jsonl"
_EVAL_SUMMARY_GLOBAL_CSV = "eval_runs_all.csv"

_EVAL_SUMMARY_CSV_FIELDS: Tuple[str, ...] = (
    "ts",
    "run_id",
    "task_idx",
    "pipeline",
    "domain_from",
    "domain_to",
    "log_file",
    "save_file",
    "task_description",
    "mae",
    "rmse",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "dist_1",
    "dist_2",
    "meteor",
    "bert",
)


def _eval_summary_enabled() -> bool:
    v = os.environ.get("D4C_EVAL_SUMMARY", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _global_eval_summary_dir() -> str:
    g = os.environ.get("D4C_EVAL_SUMMARY_GLOBAL_DIR", "").strip()
    if g:
        return os.path.abspath(os.path.expanduser(g))
    return os.path.join(D4C_ROOT, "log")


def flatten_final_metrics_for_summary(final: Dict[str, Any]) -> Dict[str, float]:
    """将 FINAL RESULTS 指标摊平为可写入 CSV/JSON 的标量。"""
    r = final["recommendation"]
    e = final["explanation"]
    rg = e["rouge"]
    bl = e["bleu"]
    di = e["dist"]

    def _f(x: Any) -> float:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)

    return {
        "mae": _f(r["mae"]),
        "rmse": _f(r["rmse"]),
        "rouge_1": _f(rg["1"]),
        "rouge_2": _f(rg["2"]),
        "rouge_l": _f(rg["l"]),
        "bleu_1": _f(bl["1"]),
        "bleu_2": _f(bl["2"]),
        "bleu_3": _f(bl["3"]),
        "bleu_4": _f(bl["4"]),
        "dist_1": _f(di["1"]),
        "dist_2": _f(di["2"]),
        "meteor": _f(e["meteor"]),
        "bert": _f(e["bert"]),
    }


def _append_text(path: str, text: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    _append_text(path, line)


def _append_csv_row(path: str, row: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    need_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_EVAL_SUMMARY_CSV_FIELDS), extrasaction="ignore")
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _EVAL_SUMMARY_CSV_FIELDS})


def append_eval_run_summaries(
    final: Dict[str, Any],
    *,
    task_idx: int,
    run_id: str,
    pipeline: str,
    domain_from: str,
    domain_to: str,
    log_file: Optional[str] = None,
    save_file: Optional[str] = None,
    task_description: Optional[str] = None,
) -> None:
    """将一次 eval 的指标追加到每任务目录与全局目录下的 .txt / .jsonl / .csv（共 6 个文件）。

    每任务：get_log_task_dir(task_idx) 下 eval_runs.{txt,jsonl,csv}
    全局：D4C_EVAL_SUMMARY_GLOBAL_DIR（默认 <项目根>/log）下 eval_runs_all.{txt,jsonl,csv}

    设 D4C_EVAL_SUMMARY=0 可关闭。失败时静默忽略，不影响主训练/评估流程。
    """
    if not _eval_summary_enabled():
        return
    try:
        metrics = flatten_final_metrics_for_summary(final)
    except Exception:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    desc_one_line = (task_description or "").replace("\n", " ").strip()
    row: Dict[str, Any] = {
        "ts": ts,
        "run_id": run_id or "",
        "task_idx": task_idx,
        "pipeline": pipeline,
        "domain_from": domain_from,
        "domain_to": domain_to,
        "log_file": os.path.abspath(os.path.expanduser(log_file)) if log_file else "",
        "save_file": os.path.abspath(os.path.expanduser(save_file)) if save_file else "",
        "task_description": desc_one_line,
        **metrics,
    }

    lines_block = format_final_results_lines(final, task_description=task_description)
    plain_sep = (
        "================================================================================\n"
        f"{ts} | run_id={run_id} | task_idx={task_idx} | pipeline={pipeline}\n"
        f"{domain_from} -> {domain_to}\n"
        f"log_file={row['log_file']}\n"
        f"save_file={row['save_file']}\n"
        "--------------------------------------------------------------------------------\n"
        + "\n".join(lines_block)
        + "\n================================================================================\n"
    )

    try:
        task_dir = get_log_task_dir(task_idx)
        _append_text(os.path.join(task_dir, _EVAL_SUMMARY_TXT), plain_sep)
        _append_jsonl(os.path.join(task_dir, _EVAL_SUMMARY_JSONL), row)
        _append_csv_row(os.path.join(task_dir, _EVAL_SUMMARY_CSV), row)

        gdir = _global_eval_summary_dir()
        _append_text(os.path.join(gdir, _EVAL_SUMMARY_GLOBAL_TXT), plain_sep)
        _append_jsonl(os.path.join(gdir, _EVAL_SUMMARY_GLOBAL_JSONL), row)
        _append_csv_row(os.path.join(gdir, _EVAL_SUMMARY_GLOBAL_CSV), row)
    except Exception:
        pass
