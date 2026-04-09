#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验 presets/**/*.yaml 结构与类型（与 code/config.py 加载规则对齐）。
不修改 config 行为；失败时 exit 1，标准错误输出可读信息。

用法（仓库根）:
  python3 scripts/check_presets.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _die(msg: str) -> None:
    print(f"[check_presets] {msg}", file=sys.stderr)


def _preset_stem_ok(stem: str) -> Tuple[bool, str]:
    if not stem:
        return False, "文件名为空"
    if stem.startswith("."):
        return False, "预设名不能以 . 开头"
    if not re.match(r"^[A-Za-z0-9_-]+$", stem):
        return False, "预设名须仅含字母、数字、下划线、连字符"
    return True, ""


def _glob_yaml(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml"))


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 D4C presets YAML")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="仓库根（含 presets/、code/）；默认为本脚本上级目录",
    )
    args = ap.parse_args()
    repo = (args.repo_root or Path(__file__).resolve().parent.parent).resolve()
    code_dir = repo / "code"
    if not code_dir.is_dir():
        _die(f"未找到 code/ 目录: {code_dir}")
        return 1

    sys.path.insert(0, str(code_dir))
    try:
        import yaml
    except ImportError:
        _die("需要 PyYAML：pip install pyyaml")
        return 1

    from config import (
        _coerce_training_preset_top_level,
        _normalize_task_row_yaml,
        _validate_hardware_presets,
        _validate_training_presets,
    )

    failed = False

    def load_mapping(path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"YAML 解析失败: {e}") from e

    # --- training ---
    tdir = repo / "presets" / "training"
    tpaths = _glob_yaml(tdir)
    if not tpaths:
        print("[check_presets] presets/training: 无 .yaml/.yml，跳过（运行时使用内置 TRAINING_PRESETS）")
    else:
        merged: Dict[str, Any] = {}
        training_any_error = False
        for path in tpaths:
            stem = path.stem
            ok, reason = _preset_stem_ok(stem)
            if not ok:
                _die(f"training 非法预设名 {path.name}: {reason}")
                training_any_error = True
                continue
            try:
                raw = load_mapping(path)
                if raw is None:
                    raise ValueError("文件为空或仅注释")
                if not isinstance(raw, dict):
                    raise TypeError(f"根须为 dict，当前为 {type(raw).__name__}")
                merged[stem] = _coerce_training_preset_top_level(raw)
            except Exception as e:
                _die(f"training {path.relative_to(repo)}: {e}")
                training_any_error = True
        if merged:
            try:
                _validate_training_presets(merged, name="presets/training")
            except Exception as e:
                _die(f"training 结构校验: {e}")
                training_any_error = True
        if training_any_error:
            failed = True

    # --- hardware (dataloader / ddp / threads) ---
    rdir = repo / "presets" / "hardware"
    rpaths = _glob_yaml(rdir)
    if not rpaths:
        print("[check_presets] presets/hardware: 无 .yaml/.yml，跳过（运行时使用内置 HARDWARE_PRESETS）")
    else:
        rmerged: Dict[str, Any] = {}
        hardware_any_error = False
        for path in rpaths:
            stem = path.stem
            ok, reason = _preset_stem_ok(stem)
            if not ok:
                _die(f"hardware 非法预设名 {path.name}: {reason}")
                hardware_any_error = True
                continue
            try:
                raw = load_mapping(path)
                if raw is None:
                    raise ValueError("文件为空或仅注释")
                if not isinstance(raw, dict):
                    raise TypeError(f"根须为 dict，当前为 {type(raw).__name__}")
                rmerged[stem] = raw
            except Exception as e:
                _die(f"hardware {path.relative_to(repo)}: {e}")
                hardware_any_error = True
        if rmerged:
            try:
                _validate_hardware_presets(rmerged, name="presets/hardware")
            except Exception as e:
                _die(f"hardware 结构校验: {e}")
                hardware_any_error = True
        if hardware_any_error:
            failed = True

    # --- tasks ---
    kdir = repo / "presets" / "tasks"
    kpaths = _glob_yaml(kdir)
    if not kpaths:
        print("[check_presets] presets/tasks: 无 .yaml/.yml，跳过（运行时使用内置 task_configs）")
    else:
        merged_tasks: Dict[int, Any] = {}
        tasks_any_error = False
        for path in kpaths:
            try:
                raw = load_mapping(path)
                if not isinstance(raw, dict):
                    raise TypeError(f"根须为 dict，当前为 {type(raw).__name__}")
                for k, v in raw.items():
                    tid = int(k)
                    if tid < 1 or tid > 8:
                        raise ValueError(f"任务号须在 1..8，收到 {tid!r}（键 {k!r}）")
                    merged_tasks[tid] = _normalize_task_row_yaml(
                        v, ctx=f"{path.name} task {tid}"
                    )
            except Exception as e:
                _die(f"tasks {path.relative_to(repo)}: {e}")
                tasks_any_error = True
        need = set(range(1, 9))
        if not tasks_any_error:
            if set(merged_tasks.keys()) != need:
                _die(
                    "tasks 合并后须恰好包含任务 1..8，当前键: "
                    f"{sorted(merged_tasks.keys())}"
                )
                tasks_any_error = True
        if tasks_any_error:
            failed = True

    # --- eval_profiles ---
    epdir = repo / "presets" / "eval_profiles"
    epaths = _glob_yaml(epdir)
    if not epaths:
        _die("presets/eval_profiles: 无 .yaml/.yml（新版主线必须存在）")
        failed = True
    else:
        eval_any_error = False
        allowed = {"hardware_preset", "decode_preset", "rerank_preset", "eval_batch_size", "num_return_sequences"}
        forbidden = {"train_batch_size", "per_device_train_batch_size", "gradient_accumulation_steps"}
        for path in epaths:
            try:
                raw = load_mapping(path)
                if not isinstance(raw, dict):
                    raise TypeError(f"根须为 dict，当前为 {type(raw).__name__}")
                extra = set(raw.keys()) - allowed
                if extra:
                    raise ValueError(f"含非法字段: {sorted(extra)}")
                bad = sorted(k for k in forbidden if k in raw)
                if bad:
                    raise ValueError(f"包含 training 字段（禁止）: {bad}")
                if "eval_batch_size" not in raw:
                    raise ValueError("缺少 eval_batch_size")
            except Exception as e:
                _die(f"eval_profiles {path.relative_to(repo)}: {e}")
                eval_any_error = True
        if eval_any_error:
            failed = True

    # --- rerank ---
    rrdir = repo / "presets" / "rerank"
    rrpaths = _glob_yaml(rrdir)
    rerank_any_error = False
    for path in rrpaths:
        try:
            raw = load_mapping(path)
            if isinstance(raw, dict) and "num_return_sequences" in raw:
                raise ValueError("num_return_sequences 已禁止，必须迁移到 presets/eval_profiles/*.yaml")
        except Exception as e:
            _die(f"rerank {path.relative_to(repo)}: {e}")
            rerank_any_error = True
    if rerank_any_error:
        failed = True

    if failed:
        _die("校验失败（见上文）")
        return 1
    print("[check_presets] 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
