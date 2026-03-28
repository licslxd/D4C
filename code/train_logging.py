# -*- coding: utf-8 -*-
"""
训练/评估统一日志：标准 logging、按 run 分文件、DDP 仅 rank0 写文件。
环境变量：
  D4C_LOG_DIR      日志目录（默认 <项目根>/logs）
  D4C_CONSOLE_LEVEL 控制台级别，默认 INFO（非 rank0 恒为 WARNING）
  D4C_FILE_LEVEL    文件级别，默认 DEBUG（仅 rank0 有 FileHandler）
  D4C_LOG_CONSOLE=1 rank0 在写日志文件时仍向 stdout 镜像（默认关闭：仅 FileHandler 写文件，避免与 shell 重定向/tee 双写）
  D4C_MIRROR_LOG=1  同时镜像到 code/log.out（旧行为，默认关闭）
  D4C_LOG_PRETTY=0  关闭多行缩进 JSON（默认开启 RUN_META / RUN_CONFIG 多行缩进，便于阅读）
  D4C_LOG_STRUCTURED_CONSOLE=1  结构化块（RUN_*）同时打到控制台；默认仅写入日志文件，避免与 FileHandler
                                并存且 stderr 被 tee/重定向到同一文件时出现重复行。
  D4C_EVAL_SUMMARY=0           关闭 eval 自动汇总（默认开启）：写入每任务与全局共 6 个文件（txt/jsonl/csv）。
  D4C_EVAL_SUMMARY_GLOBAL_DIR  全局汇总目录（显式设置时优先，路径原样使用）。未设置时：优先按 D4C_LOG_GROUP
                                为 <项目根>/log/<group>/eval；否则按 D4C_CHECKPOINT_GROUP；再否则按 SUBDIR 启发式
                                （step3_/step5_ 前缀）；否则 <项目根>/log/eval。
                                内含 eval_runs_all.{txt,jsonl,csv}
  每任务 eval 汇总：get_log_task_dir(task) 下的 eval/ 子目录内 eval_runs.{txt,jsonl,csv}（训练主日志在 runs/ 子目录，由 shell 约定）
  D4C_STEP3_ALL_SHELL_LOG  （仅 shell）run_step3_optimized.sh --all 前台 tee 汇总路径；未设置时默认为 log/step3_optimized_all_<秒级时间戳>.log
  D4C_LOG_SILENT_STDIO_WARN=1  关闭 setup_train_logging 对 stdout/stderr 与 --log_file 同路径的告警
  D4C_DUAL_TRAIN_LOG=1       双文件：--log_file 为 train.log（细粒度：Step/Grad/Checkpoint/Timing/数据告警），
                              另写同目录 nohup.log（RUN_*、epoch 块、DDP 心跳、性能汇总）。可用 D4C_SUMMARY_LOG 覆盖摘要路径。
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import string
import sys
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from paths_config import get_d4c_root, get_log_task_dir

LOGGER_NAME = "d4c"
RUN_END_LINE = "========== RUN END =========="

# 日志路由：双文件模式下 train.log 仅 detail+both，nohup.log 仅 summary+both；单文件无 Filter，等价于 both。
ROUTE_DETAIL = "detail"
ROUTE_SUMMARY = "summary"
ROUTE_BOTH = "both"


class _D4cRouteFilter(logging.Filter):
    """仅放行 d4c_route 属于 allowed 或 both 的记录。"""

    def __init__(self, allowed: FrozenSet[str]) -> None:
        super().__init__()
        self._allowed = allowed

    def filter(self, record: logging.LogRecord) -> bool:
        r = getattr(record, "d4c_route", ROUTE_BOTH)
        if r == ROUTE_BOTH:
            return True
        return r in self._allowed


def _dual_log_enabled() -> bool:
    v = os.environ.get("D4C_DUAL_TRAIN_LOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _summary_log_path(detail_path: str) -> str:
    ex = os.environ.get("D4C_SUMMARY_LOG", "").strip()
    if ex:
        return os.path.abspath(os.path.expanduser(ex))
    return os.path.join(os.path.dirname(os.path.abspath(detail_path)), "nohup.log")


def _plain_route_visible(handler: logging.Handler, route: str) -> bool:
    allowed = getattr(handler, "_d4c_routes_allowed", None)
    if allowed is None:
        return True
    if route == ROUTE_BOTH:
        return True
    return route in allowed


class _StreamSuppressFileOnlyFilter(logging.Filter):
    """带 d4c_file_only 的记录不输出到 StreamHandler，仍由 FileHandler 写入（若存在）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "d4c_file_only", False)


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{rnd}"


def _log_dir() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("D4C_LOG_DIR") or os.path.join(get_d4c_root(), "logs")))


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


def _console_mirror_enabled() -> bool:
    """rank0 写日志文件时是否仍附加 StreamHandler（默认 False，仅文件写入）。"""
    v = os.environ.get("D4C_LOG_CONSOLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def logger_has_file_handler(logger: Optional[logging.Logger]) -> bool:
    """用于判断当前是否由 FileHandler 写文件（避免 print 与 logging 双份）。"""
    if logger is None:
        return False
    return any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def _structured_log_extra(logger: logging.Logger) -> Dict[str, Any]:
    """
    若同时存在 FileHandler 与 StreamHandler 且未要求结构化块上控制台，则标记 d4c_file_only，
    由 StreamHandler 过滤器抑制，避免 tee/2>&1 与文件为同一路径时大块 JSON 重复。
    """
    if _structured_console_enabled():
        return {}
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if has_file and has_stream:
        return {"d4c_file_only": True}
    return {}


def log_route_extra(logger: logging.Logger, route: str) -> Dict[str, Any]:
    """合并结构化控制台抑制与路由字段（供 train_diagnostics 等复用）。"""
    e = _structured_log_extra(logger)
    e["d4c_route"] = route
    return e


def _realpath_resolved(path: str) -> str:
    try:
        return os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    except OSError:
        return os.path.abspath(os.path.expanduser(path))


def _warn_if_stdio_points_to_log_file(logger: logging.Logger, log_path: str) -> None:
    """若 stdout/stderr 已重定向到与 log_path 同一文件，则告警（易导致双 fd 写 train.log 乱序）。"""
    v = os.environ.get("D4C_LOG_SILENT_STDIO_WARN", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return
    target = _realpath_resolved(log_path)
    for stream_name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        try:
            if stream.isatty():
                continue
            nm = getattr(stream, "name", None)
            if not isinstance(nm, str) or not nm or nm.startswith("<"):
                continue
            if _realpath_resolved(nm) == target:
                logger.warning(
                    "[D4C] %s 与 --log_file 解析为同一路径 (%s)，可能导致重复写入或乱序；"
                    "请避免将终端重定向到 train.log，或关闭 D4C_LOG_CONSOLE。",
                    stream_name,
                    target,
                )
        except Exception:
            pass


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
    配置名为 d4c 的 logger：rank0 默认仅 FileHandler 写文件（不重定向终端也可单份）；
    设 D4C_LOG_CONSOLE=1 时 rank0 额外附加 StreamHandler；非 rank0 仅 StreamHandler（WARNING+）。
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

    log_path = log_file
    mirror = _console_mirror_enabled()
    # rank0 且有日志路径时默认不挂 StreamHandler，避免与 FileHandler 双写同一文件（含 shell 重定向）
    want_stream = (rank != 0) or (not (rank == 0 and log_path)) or mirror

    stream_handler: Optional[logging.StreamHandler] = None
    if want_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        if rank == 0:
            sh.setLevel(console_level)
        else:
            sh.setLevel(logging.WARNING)
        if rank == 0 and log_path:
            sh.addFilter(_StreamSuppressFileOnlyFilter())
        logger.addHandler(sh)
        stream_handler = sh

    summary_log_path: Optional[str] = None
    dual_log = False
    if rank == 0 and log_path:
        d = os.path.dirname(os.path.abspath(log_path))
        if d:
            os.makedirs(d, exist_ok=True)
        use_dual = _dual_log_enabled()
        sp: Optional[str] = _summary_log_path(log_path) if use_dual else None
        if use_dual and sp and _realpath_resolved(sp) != _realpath_resolved(log_path):
            dual_log = True
            summary_log_path = sp
            sd = os.path.dirname(os.path.abspath(sp))
            if sd:
                os.makedirs(sd, exist_ok=True)
            allow_d: FrozenSet[str] = frozenset({ROUTE_DETAIL, ROUTE_BOTH})
            allow_s: FrozenSet[str] = frozenset({ROUTE_SUMMARY, ROUTE_BOTH})
            fh_d = logging.FileHandler(log_path, encoding="utf-8")
            fh_d.setLevel(file_level)
            fh_d.setFormatter(fmt)
            fh_d._d4c_routes_allowed = allow_d  # type: ignore[attr-defined]
            fh_d.addFilter(_D4cRouteFilter(allow_d))
            logger.addHandler(fh_d)
            fh_s = logging.FileHandler(sp, encoding="utf-8")
            fh_s.setLevel(file_level)
            fh_s.setFormatter(fmt)
            fh_s._d4c_routes_allowed = allow_s  # type: ignore[attr-defined]
            fh_s.addFilter(_D4cRouteFilter(allow_s))
            logger.addHandler(fh_s)
            if stream_handler is not None:
                stream_handler.addFilter(_D4cRouteFilter(allow_s))
            _warn_if_stdio_points_to_log_file(logger, log_path)
        else:
            if use_dual and sp and _realpath_resolved(sp) == _realpath_resolved(log_path):
                logger.warning(
                    "[D4C] D4C_DUAL_TRAIN_LOG=1 但摘要路径与主日志相同，已回退为单文件写入。"
                )
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            _warn_if_stdio_points_to_log_file(logger, log_path)

    return {
        "logger": logger,
        "run_id": run_id,
        "log_path": log_path,
        "rank": rank,
        "world_size": world_size,
        "summary_log_path": summary_log_path,
        "dual_log": dual_log,
    }


def log_run_header(logger: logging.Logger, meta: Dict[str, Any]) -> None:
    """RUN_META：默认多行缩进 JSON（键有序）；D4C_LOG_PRETTY=0 时为单行。"""
    safe = _json_safe_sorted_dict(meta)
    pretty = _pretty_log_enabled()
    payload = json.dumps(safe, ensure_ascii=False, indent=2 if pretty else None)
    extra = log_route_extra(logger, ROUTE_SUMMARY)
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
    extra = log_route_extra(logger, ROUTE_SUMMARY)
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
    extra = log_route_extra(logger, ROUTE_SUMMARY)
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


def _fmt_duration_sec(t: float) -> str:
    if t >= 3600:
        return f"{t/3600:.1f}h"
    if t >= 60:
        return f"{t/60:.1f}m"
    return f"{t:.1f}s"


def format_epoch_training_block(
    *,
    time_str: str,
    epoch: int,
    epoch_time_s: float,
    total_time_s: float,
    step_time_s: float,
    gpu_util: str,
    gpu_mem: str,
    cpu_used: str,
    cpu_total: str,
    cpu_util: str,
    lr: float,
    train_loss: float,
    valid_loss: Optional[float] = None,
    adv_loss: Optional[float] = None,
    adversarial_coef: Optional[float] = None,
    bleu_line: Optional[str] = None,
    lr_schedule_detail: Optional[str] = None,
) -> str:
    """每 epoch 一块多行纯文本（无 logging 时间戳/级别前缀），与 perf_monitor 的 rec 字段对齐。

    第一行为 time_str，第二行为「Epoch n」，其后指标行统一缩进 4 空格；块末多一个空行与下一 epoch 分隔。
    """
    et = _fmt_duration_sec(epoch_time_s)
    tt = _fmt_duration_sec(total_time_s)
    step_ms = step_time_s * 1000
    line3 = f"epoch_time={et}\t|\ttotal={tt}\t\t|\tstep={step_ms:.0f}ms |"
    line4 = f"GPU={gpu_util}\t|\tMem={gpu_mem}\t|\tCPU={cpu_used}/{cpu_total} {cpu_util}"
    parts5: List[str] = [f"lr={lr:.6g}", f"train_loss={train_loss:.4f}"]
    if valid_loss is not None:
        parts5.append(f"valid_loss={valid_loss:.4f}")
    if adv_loss is not None:
        parts5.append(f"adv_loss={adv_loss:.4f}")
    if adversarial_coef is not None:
        parts5.append(f"adversarial_coef={adversarial_coef:.6g}")
    line5 = "\t|\t".join(parts5)
    detail = [line3, line4, line5]
    if lr_schedule_detail is not None:
        detail.append(lr_schedule_detail.rstrip())
    if bleu_line is not None:
        detail.append(bleu_line.rstrip())
    indent = "    "
    lines: List[str] = [time_str, f"Epoch {epoch}"] + [indent + ln for ln in detail]
    # 块末空行：与下一 epoch 的时间戳分隔，便于阅读
    return "\n".join(lines) + "\n\n"


def log_epoch_training_block(logger: Optional[logging.Logger], text: str) -> None:
    """写入 format_epoch_training_block 生成的多行块（双文件模式下写入 nohup 侧摘要）。"""
    _write_plain_log_block(logger, text, route=ROUTE_SUMMARY)


def format_final_results_lines(
    final: Dict[str, Any],
    *,
    task_description: Optional[str] = None,
    start_time: Optional[str] = None,
) -> List[str]:
    """构建 FINAL RESULTS 文本行（无 log 前缀；指标行用制表符缩进）。

    task_description: 可选，在评估结果块最上方增加一行「任务说明：…」（位于 FINAL RESULTS 分隔线之上）。
    start_time: 可选，eval 开始时间字符串；未传则用当前时间。
    """
    current_time = start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        ]
    )
    if "bert" in final.get("explanation", {}):
        lines.append(f"{tab}BERT: {final['explanation']['bert']} ")
    return lines


def _write_plain_log_block(
    logger: Optional[logging.Logger],
    text: str,
    *,
    route: str = ROUTE_BOTH,
) -> None:
    """写入多行纯文本（不经 Formatter）：按 d4c 路由写入对应 FileHandler / StreamHandler。"""
    if not text.endswith("\n"):
        text = text + "\n"
    if logger is None:
        print(text, end="", flush=True)
        return
    for h in logger.handlers:
        # FileHandler 继承 StreamHandler；与 logger.info 一样走 Handler 锁，避免与 emit 交错
        if isinstance(h, logging.StreamHandler) and _plain_route_visible(h, route):
            try:
                h.acquire()
                try:
                    h.stream.write(text)
                    h.flush()
                finally:
                    h.release()
            except Exception:
                pass


def log_final_results_block(logger: Optional[logging.Logger], lines: list) -> None:
    """FINAL RESULTS 多行块（双文件模式下写入摘要侧）。"""
    text = "\n".join(lines)
    _write_plain_log_block(logger, text, route=ROUTE_SUMMARY)


def flush_preset_load_events(logger: Optional[logging.Logger]) -> None:
    """将 import config 阶段记录的 presets YAML 加载结果刷入训练日志（摘要路由）。"""
    if logger is None:
        return
    try:
        import config as cfg

        ev = getattr(cfg, "PRESET_LOAD_EVENTS", None) or []
        for line in ev:
            logger.info("[PresetYAML] %s", line, extra=log_route_extra(logger, ROUTE_SUMMARY))
    except Exception:
        pass


def finalize_run_log(logger: Optional[logging.Logger], extra: Optional[str] = None) -> None:
    _write_plain_log_block(logger, RUN_END_LINE + "\n", route=ROUTE_BOTH)
    if logger is not None:
        if extra:
            logger.info("%s", extra, extra=log_route_extra(logger, ROUTE_BOTH))
    elif extra:
        print(extra, flush=True)


# 误写入 train.log 的常见 shell 行（历史 bug 或手工重定向）
_TRAIN_LOG_SHELL_MARKERS = (
    "---------- Task ",
    "========== 跳过 Task ",
)
_EPOCH_HEAD_LINE_RE = re.compile(r"^Epoch (\d+)\s*$")


def audit_train_log_file(path: str) -> Dict[str, Any]:
    """轻量自检（启发式）：是否混入 shell 包装行、Epoch 行序列是否严格递增 1。

    适用于单次连续训练；同文件若含多段 train/eval，可能出现重复 Epoch 编号，结果仅供参考。
    """
    result: Dict[str, Any] = {
        "path": path,
        "shell_hits": [],
        "epoch_numbers": [],
        "epoch_sequence_gaps": [],
    }
    if not path or not os.path.isfile(path):
        result["error"] = "not_a_file"
        return result
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            lines = fp.readlines()
    except OSError as e:
        result["error"] = str(e)
        return result
    for lineno, ln in enumerate(lines, 1):
        for marker in _TRAIN_LOG_SHELL_MARKERS:
            if marker in ln:
                result["shell_hits"].append(
                    {"line": lineno, "marker": marker, "snippet": ln.strip()[:160]}
                )
                break
    for ln in lines:
        m = _EPOCH_HEAD_LINE_RE.match(ln.strip())
        if m:
            result["epoch_numbers"].append(int(m.group(1)))
    nums = result["epoch_numbers"]
    if len(nums) >= 2:
        for a, b in zip(nums, nums[1:]):
            if b != a + 1:
                result["epoch_sequence_gaps"].append({"after_epoch": a, "next_seen": b})
    result["epoch_line_count"] = len(nums)
    result["epoch_max"] = max(nums) if nums else None
    return result


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

_EVAL_SUMMARY_SUBDIR = "eval"  # 与 runs/…/train.log 分层：…/<group>/eval/eval_runs.*

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
    "eval_elapsed_s",
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
    _root = get_d4c_root()
    log_group = os.environ.get("D4C_LOG_GROUP", "").strip().lower()
    if log_group in ("step3", "step5"):
        return os.path.join(_root, "log", log_group, _EVAL_SUMMARY_SUBDIR)
    group = os.environ.get("D4C_CHECKPOINT_GROUP", "").strip().lower()
    if group in ("step3", "step5"):
        return os.path.join(_root, "log", group, _EVAL_SUMMARY_SUBDIR)
    sub = os.environ.get("D4C_CHECKPOINT_SUBDIR", "").strip().lower()
    if sub.startswith("step3_"):
        return os.path.join(_root, "log", "step3", _EVAL_SUMMARY_SUBDIR)
    if sub.startswith("step5_"):
        return os.path.join(_root, "log", "step5", _EVAL_SUMMARY_SUBDIR)
    return os.path.join(_root, "log", _EVAL_SUMMARY_SUBDIR)


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

    d = {
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
    }
    if "bert" in e:
        d["bert"] = _f(e["bert"])
    return d


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
    start_time: Optional[str] = None,
    eval_elapsed: Optional[float] = None,
) -> None:
    """将一次 eval 的指标追加到每任务目录与全局目录下的 .txt / .jsonl / .csv（共 6 个文件）。

    每任务：get_log_task_dir(task_idx)/eval/ 下 eval_runs.{txt,jsonl,csv}（如 log/<task>/step3/eval/、log/<task>/step5/eval/）
    全局：D4C_EVAL_SUMMARY_GLOBAL_DIR（显式路径不自动加子目录），否则为 log/step3/eval、log/step5/eval
          或 log/eval 下 eval_runs_all.{txt,jsonl,csv}

    设 D4C_EVAL_SUMMARY=0 可关闭。失败时静默忽略，不影响主训练/评估流程。
    """
    if not _eval_summary_enabled():
        return
    try:
        metrics = flatten_final_metrics_for_summary(final)
    except Exception:
        return

    ts = start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        "eval_elapsed_s": round(eval_elapsed, 1) if eval_elapsed is not None else "",
    }

    lines_block = format_final_results_lines(final, task_description=task_description, start_time=start_time)
    if eval_elapsed is not None:
        _m, _s = divmod(int(eval_elapsed), 60)
        lines_block.append(f"Eval elapsed: {_m}m {_s}s ({eval_elapsed:.1f}s)")
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
        task_eval_dir = os.path.join(get_log_task_dir(task_idx), _EVAL_SUMMARY_SUBDIR)
        _append_text(os.path.join(task_eval_dir, _EVAL_SUMMARY_TXT), plain_sep)
        _append_jsonl(os.path.join(task_eval_dir, _EVAL_SUMMARY_JSONL), row)
        _append_csv_row(os.path.join(task_eval_dir, _EVAL_SUMMARY_CSV), row)

        gdir = _global_eval_summary_dir()
        _append_text(os.path.join(gdir, _EVAL_SUMMARY_GLOBAL_TXT), plain_sep)
        _append_jsonl(os.path.join(gdir, _EVAL_SUMMARY_GLOBAL_JSONL), row)
        _append_csv_row(os.path.join(gdir, _EVAL_SUMMARY_GLOBAL_CSV), row)
    except Exception:
        pass
