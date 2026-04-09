#!/usr/bin/env bash
# Shell 层 tee/nohup **汇总**路径（与 code/d4c_core/path_layout.py 一致）。
# - runs/global/vN/meta/shell_logs/ — 跨任务编排器汇总
# - runs/task{T}/vN/meta/shell_logs/ — 单任务编排器汇总

d4c_shell_logs_global() {
    local root="$1"
    local iter="${D4C_ITER:-v1}"
    local d="$root/runs/global/$iter/meta/shell_logs"
    mkdir -p "$d"
    printf '%s' "$d"
}

d4c_shell_logs_task() {
    local root="$1" task="$2"
    local iter="${D4C_ITER:-v1}"
    local d="$root/runs/task${task}/$iter/meta/shell_logs"
    mkdir -p "$d"
    printf '%s' "$d"
}
