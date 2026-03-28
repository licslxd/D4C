#!/usr/bin/env bash
# shellcheck shell=bash
# D4C — 训练入口公共库：fail-fast 校验与派发（由 scripts/train_ddp.sh source）
#
# 设计约束：
# - --gpus → 仅 CUDA_VISIBLE_DEVICES；--ddp-nproc → 仅 DDP_NPROC + 透传子脚本
# - D4C_TRAIN_PRESET / D4C_RUNTIME_PRESET 来自 CLI；任务默认表仅在 Python TASK_DEFAULTS
#
set -euo pipefail

d4c_train_usage() {
    cat <<'EOF'
用法: bash scripts/train_ddp.sh [选项]

  --step 3|4|5           单步：调用 sh/run_step{3,4,5}_optimized.sh（4/5 须 --step3-subdir）
  --pipeline 3,4,5      按 3→4→5 顺序执行子集（例 3,4 或 4,5 或 3,4,5；去重后排序）
  --task N               任务 1–8（必填）
  --step3-subdir NAME    无 Step 3 的 pipeline（如 4,5）或单步 4/5 时必填；含 Step 3 时可省略（Step 4/5 用本轮 Step 3 产生的最新 step3_opt_*）
  --gpus LIST            仅设置 CUDA_VISIBLE_DEVICES（例 0,1）
  --ddp-nproc K          设置 DDP_NPROC，并 --ddp-nproc 透传子脚本；未传则用环境变量或默认 2 参与校验
  --train-preset NAME    → D4C_TRAIN_PRESET
  --runtime-preset NAME  → D4C_RUNTIME_PRESET
  --batch-size N         转发 --batch-size；若指定则须能被 DDP_NPROC 整除

约束：--step 与 --pipeline 二选一。

示例:
  bash scripts/train_ddp.sh --step 3 --task 2 --ddp-nproc 2 --gpus 0,1
  bash scripts/train_ddp.sh --pipeline 3,4,5 --task 4 --ddp-nproc 2 --gpus 0,1 --batch-size 1024
  bash scripts/train_ddp.sh --pipeline 4,5 --task 4 --step3-subdir step3_opt_20260325_1503 --ddp-nproc 2

兼容：可直接使用 sh/run_step*_optimized.sh（见 docs/D4C_RUNTIME_SPEC.md）。
EOF
}

d4c_train_die() {
    echo "[train_ddp] 错误: $*" >&2
    exit 2
}

d4c_train_log() {
    echo "[train_ddp] $*"
}

# --- fail-fast 校验（对外语义名与函数一致） ---
d4c_train_validate_task() {
    local t="$1"
    [[ "$t" =~ ^[1-8]$ ]] || d4c_train_die "validate_task: task 须为 1-8，收到: ${t:-<空>}"
}

d4c_train_validate_gpus() {
    local g="$1"
    [[ -n "$g" ]] || d4c_train_die "validate_gpus: --gpus 不能为空"
    [[ "$g" =~ ^[0-9]+(,[0-9]+)*$ ]] || d4c_train_die "validate_gpus: 须为数字逗号列表，例如 0,1，收到: $g"
}

d4c_train_validate_ddp_nproc() {
    local n="$1"
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || d4c_train_die "validate_ddp_nproc: 须为正整数，收到: ${n:-<空>}"
}

# 若已设置 CUDA_VISIBLE_DEVICES，则可见 GPU 数 >= DDP_NPROC
d4c_train_validate_gpu_vs_ddp() {
    local nproc="$1"
    local cvd="${CUDA_VISIBLE_DEVICES:-}"
    [[ -z "$cvd" ]] && return 0
    local n=0
    n=$(echo "$cvd" | awk -F',' '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}')
    [[ "$n" -ge "$nproc" ]] || d4c_train_die "validate_gpu_vs_ddp: 可见 GPU 数 ($n) 须 >= DDP_NPROC ($nproc)，CUDA_VISIBLE_DEVICES=$cvd"
}

d4c_train_validate_batch_world_size() {
    local bs="$1"
    local nproc="$2"
    [[ -z "$bs" ]] && return 0
    [[ "$bs" =~ ^[1-9][0-9]*$ ]] || d4c_train_die "validate_batch_world_size: --batch-size 须为正整数，收到: $bs"
    ((bs % nproc == 0)) || d4c_train_die "validate_batch_world_size: batch-size ($bs) 须能被 DDP_NPROC ($nproc) 整除"
}

# Step 4/5 上游：checkpoints/<task>/step3_optimized/<subdir>/
d4c_train_validate_step3_subdir() {
    local root="$1"
    local task="$2"
    local sub="$3"
    [[ -n "$sub" ]] || d4c_train_die "validate_step3_subdir: Step 4/5 需要非空的 --step3-subdir"
    local p="${root}/checkpoints/${task}/step3_optimized/${sub}"
    [[ -d "$p" ]] || d4c_train_die "validate_step3_subdir: 上游目录不存在或不是目录: $p"
}

d4c_train_validate_step() {
    local s="$1"
    case "$s" in
        3|4|5) ;;
        *) d4c_train_die "validate_step: 须为 3、4 或 5，收到: ${s:-<空>}" ;;
    esac
}

# 解析 pipeline 规格 → 去重后按 3,4,5 顺序输出空格分隔步号；设置全局 _d4c_pl_has3 _d4c_pl_has4 _d4c_pl_has5
d4c_train_parse_pipeline_spec() {
    local spec="$1"
    [[ -n "$spec" ]] || d4c_train_die "validate_pipeline: --pipeline 不能为空"
    local have3=0 have4=0 have5=0
    local IFS=','
    local -a parts
    IFS=',' read -ra parts <<<"$spec"
    local p
    for raw in "${parts[@]}"; do
        p="${raw//[[:space:]]/}"
        [[ -n "$p" ]] || d4c_train_die "validate_pipeline: 含空段，原始: $spec"
        case "$p" in
            3) have3=1 ;;
            4) have4=1 ;;
            5) have5=1 ;;
            *) d4c_train_die "validate_pipeline: 每段须为 3、4 或 5，收到: $p" ;;
        esac
    done
    _d4c_pl_has3=$have3
    _d4c_pl_has4=$have4
    _d4c_pl_has5=$have5
    _d4c_pl_ordered=""
    [[ "$have3" -eq 1 ]] && _d4c_pl_ordered+="3 "
    [[ "$have4" -eq 1 ]] && _d4c_pl_ordered+="4 "
    [[ "$have5" -eq 1 ]] && _d4c_pl_ordered+="5 "
    _d4c_pl_ordered="${_d4c_pl_ordered% }"
    [[ -n "$_d4c_pl_ordered" ]] || d4c_train_die "validate_pipeline: 至少须包含一步"
}

d4c_train_latest_step3_subdir_or_die() {
    local root="$1"
    local tid="$2"
    local base="${root}/checkpoints/${tid}/step3_optimized"
    [[ -d "$base" ]] || d4c_train_die "Step 3 完成后未找到目录: $base"
    local latest
    latest=$(ls -1td "${base}"/step3_opt_* 2>/dev/null | head -1) || true
    [[ -n "$latest" ]] || d4c_train_die "未在 $base 下找到 step3_opt_*（Step 3 是否成功写完 checkpoint？）"
    basename "$latest"
}

d4c_train_print_launch_summary() {
    local mode_label="$1"
    local mode_val="$2"
    local task="$3"
    local step3_hint="$4"
    local eff_ddp="$5"
    local bs="${6:-}"
    d4c_train_log "========== launch =========="
    d4c_train_log "task:              ${task}"
    d4c_train_log "${mode_label}:      ${mode_val}"
    d4c_train_log "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    d4c_train_log "DDP_NPROC (effective): ${eff_ddp}"
    d4c_train_log "D4C_TRAIN_PRESET:  ${D4C_TRAIN_PRESET:-<unset>}"
    d4c_train_log "D4C_RUNTIME_PRESET: ${D4C_RUNTIME_PRESET:-<unset>}"
    d4c_train_log "batch-size (CLI):  ${bs:-<omit>}"
    d4c_train_log "step3-subdir:      ${step3_hint}"
    d4c_train_log "============================"
}

d4c_train_main() {
    local d4c_root="$1"
    shift || true

    local step=""
    local pipeline=""
    local task=""
    local gpus=""
    local ddp_cli=""
    local train_preset=""
    local runtime_preset=""
    local batch_size=""
    local step3_subdir=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                d4c_train_usage
                exit 0
                ;;
            --step)
                step="${2:-}"
                shift 2 || d4c_train_die "--step 缺少值"
                ;;
            --pipeline)
                pipeline="${2:-}"
                shift 2 || d4c_train_die "--pipeline 缺少值"
                ;;
            --task)
                task="${2:-}"
                shift 2 || d4c_train_die "--task 缺少值"
                ;;
            --gpus=*)
                gpus="${1#*=}"
                shift
                ;;
            --gpus)
                gpus="${2:-}"
                shift 2 || d4c_train_die "--gpus 缺少值"
                ;;
            --ddp-nproc=*)
                ddp_cli="${1#*=}"
                shift
                ;;
            --ddp-nproc)
                ddp_cli="${2:-}"
                shift 2 || d4c_train_die "--ddp-nproc 缺少值"
                ;;
            --train-preset=*)
                train_preset="${1#*=}"
                shift
                ;;
            --train-preset)
                train_preset="${2:-}"
                shift 2 || d4c_train_die "--train-preset 缺少值"
                ;;
            --runtime-preset=*)
                runtime_preset="${1#*=}"
                shift
                ;;
            --runtime-preset)
                runtime_preset="${2:-}"
                shift 2 || d4c_train_die "--runtime-preset 缺少值"
                ;;
            --batch-size=*)
                batch_size="${1#*=}"
                shift
                ;;
            --batch-size)
                batch_size="${2:-}"
                shift 2 || d4c_train_die "--batch-size 缺少值"
                ;;
            --step3-subdir=*)
                step3_subdir="${1#*=}"
                shift
                ;;
            --step3-subdir)
                step3_subdir="${2:-}"
                shift 2 || d4c_train_die "--step3-subdir 缺少值"
                ;;
            *)
                d4c_train_die "未知参数: $1（见 bash scripts/train_ddp.sh --help）"
                ;;
        esac
    done

    [[ -n "$step" && -n "$pipeline" ]] && d4c_train_die "不能同时指定 --step 与 --pipeline"
    [[ -z "$step" && -z "$pipeline" ]] && {
        d4c_train_usage >&2
        d4c_train_die "须指定 --step 或 --pipeline"
    }

    d4c_train_validate_task "$task"

    local effective_ddp="${ddp_cli:-${DDP_NPROC:-2}}"
    d4c_train_validate_ddp_nproc "$effective_ddp"

    if [[ -n "$gpus" ]]; then
        d4c_train_validate_gpus "$gpus"
        export CUDA_VISIBLE_DEVICES="$gpus"
    fi

    d4c_train_validate_gpu_vs_ddp "$effective_ddp"

    if [[ -n "$batch_size" ]]; then
        d4c_train_validate_batch_world_size "$batch_size" "$effective_ddp"
    fi

    if [[ -n "$ddp_cli" ]]; then
        export DDP_NPROC="$ddp_cli"
    fi

    if [[ -n "$train_preset" ]]; then
        export D4C_TRAIN_PRESET="$train_preset"
    fi
    if [[ -n "$runtime_preset" ]]; then
        export D4C_RUNTIME_PRESET="$runtime_preset"
    fi

    local sh_dir="${d4c_root}/sh"
    local extra_bs=()
    [[ -n "$batch_size" ]] && extra_bs=(--batch-size "$batch_size")
    local extra_ddp=()
    [[ -n "$ddp_cli" ]] && extra_ddp=(--ddp-nproc "$ddp_cli")

    local summary_step3_display=""
    local mode_label="step"
    local mode_val=""

    if [[ -n "$pipeline" ]]; then
        d4c_train_parse_pipeline_spec "$pipeline"
        mode_label="pipeline"
        mode_val="$_d4c_pl_ordered (ordered)"

        # 无 Step 3 却含 4 或 5：必须已有 step3 目录
        if [[ "$_d4c_pl_has3" -eq 0 ]]; then
            if [[ "$_d4c_pl_has4" -eq 1 || "$_d4c_pl_has5" -eq 1 ]]; then
                d4c_train_validate_step3_subdir "$d4c_root" "$task" "$step3_subdir"
            fi
            summary_step3_display="$step3_subdir"
        else
            if [[ -n "$step3_subdir" ]]; then
                summary_step3_display="${step3_subdir} (用户指定；Step 4/5 将使用 Step 3 完成后最新 step3_opt_*)"
            else
                summary_step3_display="(Step 3 完成后自动解析最新 step3_opt_*)"
            fi
        fi

        d4c_train_print_launch_summary "$mode_label" "$mode_val" "$task" "$summary_step3_display" "$effective_ddp" "${batch_size:-}"

        local cur_sub="$step3_subdir"
        local s
        for s in $_d4c_pl_ordered; do
            case "$s" in
                3)
                    bash "${sh_dir}/run_step3_optimized.sh" --task "$task" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp[@]+"${extra_ddp[@]}"}"
                    cur_sub="$(d4c_train_latest_step3_subdir_or_die "$d4c_root" "$task")"
                    d4c_train_log "Step 3 完成，解析 step3-subdir: $cur_sub"
                    ;;
                4)
                    d4c_train_validate_step3_subdir "$d4c_root" "$task" "$cur_sub"
                    bash "${sh_dir}/run_step4_optimized.sh" --task "$task" --step3-subdir "$cur_sub" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp[@]+"${extra_ddp[@]}"}"
                    ;;
                5)
                    d4c_train_validate_step3_subdir "$d4c_root" "$task" "$cur_sub"
                    bash "${sh_dir}/run_step5_optimized.sh" --task "$task" --step3-subdir "$cur_sub" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp[@]+"${extra_ddp[@]}"}"
                    ;;
            esac
        done
        exit 0
    fi

    # 单步
    d4c_train_validate_step "$step"
    mode_val="$step"
    case "$step" in
        3)
            summary_step3_display="(n/a for step 3)"
            ;;
        4|5)
            d4c_train_validate_step3_subdir "$d4c_root" "$task" "$step3_subdir"
            summary_step3_display="$step3_subdir"
            ;;
    esac

    d4c_train_print_launch_summary "$mode_label" "$mode_val" "$task" "$summary_step3_display" "$effective_ddp" "${batch_size:-}"

    case "$step" in
        3)
            exec bash "${sh_dir}/run_step3_optimized.sh" --task "$task" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp[@]+"${extra_ddp[@]}"}"
            ;;
        4)
            exec bash "${sh_dir}/run_step4_optimized.sh" --task "$task" --step3-subdir "$step3_subdir" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp[@]+"${extra_ddp[@]}"}"
            ;;
        5)
            exec bash "${sh_dir}/run_step5_optimized.sh" --task "$task" --step3-subdir "$step3_subdir" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp[@]+"${extra_ddp[@]}"}"
            ;;
    esac
}
