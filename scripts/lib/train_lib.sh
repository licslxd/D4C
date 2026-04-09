#!/usr/bin/env bash
# shellcheck shell=bash
# D4C — Shell 编排：GPU/CVD/DDP 校验后调用 python code/d4c.py（由 scripts/entrypoints/train_ddp.sh source）
# 业务语义只来自 CLI / preset YAML；不向环境注入 D4C_TRAIN_PRESET / TRAIN_*。
#
set -euo pipefail

d4c_train_usage() {
    cat <<'EOF'
说明: Shell 编排层（GPU/CVD/DDP 校验后调用 python code/d4c.py）。
日常实验请优先: python code/d4c.py step3|step4|step5|eval|pipeline …

用法: bash scripts/entrypoints/train_ddp.sh [选项]

  --step 3|4|5           单步：python code/d4c.py step{3,4,5} …（4/5 须 --from-run）
  --pipeline 3,4,5      去重排序后执行；若恰好为 Step3→4→5 则**唯一推荐**调用
                        python code/d4c.py pipeline（与 Python 主线一致）。
                        其它组合（如 3、4 或 4,5）由本脚本按顺序调用 d4c.py step{3,4,5}，
                        属分步 Shell orchestrator，**非** d4c.py pipeline 语义。
  --task N               任务 1–8（必填）
  --iter vN              迭代目录（默认 v1）
  --from-run NAME        无 Step 3 的 pipeline 或单步 4/5 时必填（Step3 run 目录名）
  --gpus LIST            仅设置 CUDA_VISIBLE_DEVICES
  --ddp-nproc K          设置 DDP_NPROC，并映射为 d4c.py --ddp-world-size
  --preset NAME          训练预设 YAML stem（Step3/Step4 上下文；默认 step3）；**不**再使用 --train-preset / D4C_TRAIN_PRESET
  --hardware-preset NAME  转发 d4c.py --hardware-preset
  --batch-size N         转发 --batch-size（仅 step3/step5；step4 推理 batch 仅来自 --eval-profile）
  --eval-profile STEM    Step4 / 含 4 或 5 的 pipeline / 单步 5：**必填**（勿依赖 shell 隐式默认）
  --step4-run NAME       仅单步 step5 且 --step5-run 为 auto 时必填（如 2_1）

约束：--step 与 --pipeline 二选一。

示例:
  bash scripts/entrypoints/train_ddp.sh --step 3 --task 2 --iter v1 --ddp-nproc 2 --gpus 0,1
  bash scripts/entrypoints/train_ddp.sh --pipeline 3,4,5 --task 4 --iter v1 --ddp-nproc 2 --gpus 0,1 --batch-size 1024 --eval-profile eval_fast_single_gpu
  bash scripts/entrypoints/train_ddp.sh --pipeline 4,5 --task 4 --iter v1 --from-run 2 --ddp-nproc 2 --eval-profile eval_balanced_2gpu

多 seed: bash scripts/orchestration/step5_multi_seed.sh …
EOF
}

d4c_train_die() {
    echo "[train_ddp] 错误: $*" >&2
    exit 2
}

d4c_train_log() {
    echo "[train_ddp] $*"
}

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

d4c_train_validate_gpu_vs_ddp() {
    local nproc="$1"
    [[ -z "$nproc" ]] && return 0
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
    [[ -z "$nproc" ]] && return 0
    [[ "$bs" =~ ^[1-9][0-9]*$ ]] || d4c_train_die "validate_batch_world_size: --batch-size 须为正整数，收到: $bs"
    ((bs % nproc == 0)) || d4c_train_die "validate_batch_world_size: batch-size ($bs) 须能被 DDP_NPROC ($nproc) 整除"
}

d4c_train_validate_from_run() {
    local root="$1"
    local task="$2"
    local iter="$3"
    local name="$4"
    [[ -n "$name" ]] || d4c_train_die "validate_from_run: Step 4/5 需要非空的 --from-run（Step3 run 目录名）"
    local p="${root}/runs/task${task}/${iter}/train/step3/${name}"
    [[ -d "$p" ]] || d4c_train_die "validate_from_run: Step3 run 目录不存在: $p"
}

d4c_train_validate_step() {
    local s="$1"
    case "$s" in
        3|4|5) ;;
        *) d4c_train_die "validate_step: 须为 3、4 或 5，收到: ${s:-<空>}" ;;
    esac
}

d4c_train_parse_pipeline_spec() {
    local spec="$1"
    [[ -n "$spec" ]] || d4c_train_die "validate_pipeline: --pipeline 不能为空"
    local have3=0 have4=0 have5=0
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

d4c_train_latest_from_run_or_die() {
    local root="$1"
    local tid="$2"
    local iter="$3"
    local base="${root}/runs/task${tid}/${iter}/train/step3"
    [[ -d "$base" ]] || d4c_train_die "Step 3 完成后未找到目录: $base"
    local latest
    latest=$(ls -1d "${base}"/run* 2>/dev/null | sort -V | tail -1) || true
    [[ -n "$latest" ]] || d4c_train_die "未在 $base 下找到 run*（Step 3 是否成功写完？）"
    basename "$latest"
}

# 同一迭代下最新 Step4 run 目录名（用于 pipeline 子集 4→5 衔接 --step4-run）
d4c_train_latest_step4_run_or_die() {
    local root="$1"
    local tid="$2"
    local iter="$3"
    local base="${root}/runs/task${tid}/${iter}/train/step4"
    [[ -d "$base" ]] || d4c_train_die "Step4 目录不存在: $base"
    local latest
    latest=$(ls -1d "${base}"/*/ 2>/dev/null | sort -V | tail -1) || true
    [[ -n "$latest" ]] || d4c_train_die "未在 $base 下找到 step4 run 子目录"
    basename "$latest"
}

d4c_train_print_launch_summary() {
    local mode_label="$1"
    local mode_val="$2"
    local task="$3"
    local iter="$4"
    local from_hint="$5"
    local eff_ddp="$6"
    local bs="${7:-}"
    local preset_disp="${8:-}"
    local ep_disp="${9:-}"
    d4c_train_log "========== launch =========="
    d4c_train_log "task:              ${task}"
    d4c_train_log "iter:              ${iter}"
    d4c_train_log "${mode_label}:      ${mode_val}"
    d4c_train_log "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    d4c_train_log "DDP_NPROC (effective): ${eff_ddp}"
    d4c_train_log "--preset (CLI):    ${preset_disp}"
    d4c_train_log "--eval-profile:    ${ep_disp}"
    d4c_train_log "D4C_HARDWARE_PRESET(env): ${D4C_HARDWARE_PRESET:-<unset>}"
    d4c_train_log "batch-size (CLI):  ${bs:-<omit>}"
    d4c_train_log "from-run (hint):   ${from_hint}"
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
    local preset_name=""
    local hardware_preset=""
    local batch_size=""
    local from_run=""
    local iteration_id=""
    local eval_profile=""
    local step4_run_cli=""

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
            --iter)
                iteration_id="${2:-}"
                shift 2 || d4c_train_die "--iter 缺少值"
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
            --preset=*)
                preset_name="${1#*=}"
                shift
                ;;
            --preset)
                preset_name="${2:-}"
                shift 2 || d4c_train_die "--preset 缺少值"
                ;;
            --hardware-preset=*)
                hardware_preset="${1#*=}"
                shift
                ;;
            --hardware-preset)
                hardware_preset="${2:-}"
                shift 2 || d4c_train_die "--hardware-preset 缺少值"
                ;;
            --batch-size=*)
                batch_size="${1#*=}"
                shift
                ;;
            --batch-size)
                batch_size="${2:-}"
                shift 2 || d4c_train_die "--batch-size 缺少值"
                ;;
            --eval-profile=*)
                eval_profile="${1#*=}"
                shift
                ;;
            --eval-profile)
                eval_profile="${2:-}"
                shift 2 || d4c_train_die "--eval-profile 缺少值"
                ;;
            --from-run=*)
                from_run="${1#*=}"
                shift
                ;;
            --from-run)
                from_run="${2:-}"
                shift 2 || d4c_train_die "--from-run 缺少值"
                ;;
            --step4-run=*)
                step4_run_cli="${1#*=}"
                shift
                ;;
            --step4-run)
                step4_run_cli="${2:-}"
                shift 2 || d4c_train_die "--step4-run 缺少值"
                ;;
            *)
                d4c_train_die "未知参数: $1（见 bash scripts/entrypoints/train_ddp.sh --help）"
                ;;
        esac
    done

    [[ -n "$step" && -n "$pipeline" ]] && d4c_train_die "不能同时指定 --step 与 --pipeline"
    [[ -z "$step" && -z "$pipeline" ]] && {
        d4c_train_usage >&2
        d4c_train_die "须指定 --step 或 --pipeline"
    }

    d4c_train_validate_task "$task"

    local effective_ddp=""
    if [[ -n "${ddp_cli:-}" ]]; then
        effective_ddp="$ddp_cli"
    elif [[ -n "${DDP_NPROC:-}" ]]; then
        effective_ddp="$DDP_NPROC"
    fi
    if [[ -n "$effective_ddp" ]]; then
        d4c_train_validate_ddp_nproc "$effective_ddp"
    fi

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

    export D4C_ITER="${iteration_id:-${D4C_ITER:-v1}}"
    local it="$D4C_ITER"

    local py_d4c="${d4c_root}/code/d4c.py"
    local preset_train="${preset_name:-step3}"

    local extra_bs=()
    [[ -n "$batch_size" ]] && extra_bs=(--batch-size "$batch_size")
    local extra_ddp_ws=()
    [[ -n "$effective_ddp" ]] && extra_ddp_ws=(--ddp-world-size "$effective_ddp")
    local extra_hw=()
    [[ -n "$hardware_preset" ]] && extra_hw=(--hardware-preset "$hardware_preset")
    local iter_args=(--iter "$it")

    local summary_from_display=""
    local mode_label="step"
    local mode_val=""

    if [[ -n "$pipeline" ]]; then
        d4c_train_parse_pipeline_spec "$pipeline"
        mode_label="pipeline"
        mode_val="$_d4c_pl_ordered (ordered)"

        if [[ "$_d4c_pl_has4" -eq 1 || "$_d4c_pl_has5" -eq 1 ]]; then
            [[ -n "$eval_profile" ]] || d4c_train_die "pipeline 含 Step4 或 Step5 时必须提供 --eval-profile"
        fi

        if [[ "$_d4c_pl_has3" -eq 0 ]]; then
            if [[ "$_d4c_pl_has4" -eq 1 || "$_d4c_pl_has5" -eq 1 ]]; then
                d4c_train_validate_from_run "$d4c_root" "$task" "$it" "$from_run"
            fi
            summary_from_display="$from_run"
        else
            if [[ -n "$from_run" ]]; then
                summary_from_display="${from_run}（用户指定；Step 4/5 将使用 Step 3 完成后最新 run*）"
            else
                summary_from_display="(Step 3 完成后自动解析最新 run*)"
            fi
        fi

        d4c_train_print_launch_summary "$mode_label" "$mode_val" "$task" "$it" "$summary_from_display" "$effective_ddp" "${batch_size:-}" "$preset_train" "$eval_profile"

        # 唯一推荐：完整 3→4→5 与 Python run_pipeline 对齐
        if [[ "$_d4c_pl_has3" -eq 1 && "$_d4c_pl_has4" -eq 1 && "$_d4c_pl_has5" -eq 1 && "$_d4c_pl_ordered" == "3 4 5" ]]; then
            exec python "$py_d4c" pipeline --task "$task" --preset "$preset_train" \
                "${iter_args[@]}" --eval-profile "$eval_profile" \
                "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp_ws[@]+"${extra_ddp_ws[@]}"}" \
                "${extra_hw[@]+"${extra_hw[@]}"}"
        fi

        local cur_sub="$from_run"
        local cur_s4=""
        local s
        for s in $_d4c_pl_ordered; do
            case "$s" in
                3)
                    python "$py_d4c" step3 --task "$task" --preset "$preset_train" \
                        "${iter_args[@]}" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp_ws[@]+"${extra_ddp_ws[@]}"}" \
                        "${extra_hw[@]+"${extra_hw[@]}"}"
                    cur_sub="$(d4c_train_latest_from_run_or_die "$d4c_root" "$task" "$it")"
                    d4c_train_log "Step 3 完成，解析 from-run: $cur_sub"
                    ;;
                4)
                    d4c_train_validate_from_run "$d4c_root" "$task" "$it" "$cur_sub"
                    python "$py_d4c" step4 --task "$task" --preset "$preset_train" \
                        "${iter_args[@]}" --from-run "$cur_sub" --eval-profile "$eval_profile" \
                        "${extra_ddp_ws[@]+"${extra_ddp_ws[@]}"}" \
                        "${extra_hw[@]+"${extra_hw[@]}"}"
                    cur_s4="$(d4c_train_latest_step4_run_or_die "$d4c_root" "$task" "$it")"
                    d4c_train_log "Step 4 完成，step4-run: $cur_s4"
                    ;;
                5)
                    d4c_train_validate_from_run "$d4c_root" "$task" "$it" "$cur_sub"
                    [[ -n "$cur_s4" ]] || d4c_train_die "Step5 前须先完成 Step4（或改用 python code/d4c.py pipeline）"
                    python "$py_d4c" step5 --task "$task" --preset step5 \
                        "${iter_args[@]}" --from-run "$cur_sub" --step5-run auto \
                        --step4-run "$cur_s4" \
                        --eval-profile "$eval_profile" \
                        "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp_ws[@]+"${extra_ddp_ws[@]}"}" \
                        "${extra_hw[@]+"${extra_hw[@]}"}"
                    ;;
            esac
        done
        exit 0
    fi

    d4c_train_validate_step "$step"
    mode_val="$step"
    case "$step" in
        3)
            summary_from_display="(n/a for step 3)"
            ;;
        4|5)
            d4c_train_validate_from_run "$d4c_root" "$task" "$it" "$from_run"
            summary_from_display="$from_run"
            ;;
    esac

    case "$step" in
        4)
            [[ -n "$eval_profile" ]] || d4c_train_die "单步 step4 须 --eval-profile"
            ;;
        5)
            [[ -n "$eval_profile" ]] || d4c_train_die "单步 step5（train+eval）须 --eval-profile；若仅训练请用 scripts/entrypoints/step5.sh --train-only"
            [[ -n "$step4_run_cli" ]] || d4c_train_die "单步 step5 且 --step5-run auto 时须 --step4-run（与 d4c.py 合同一致）"
            ;;
    esac

    d4c_train_print_launch_summary "$mode_label" "$mode_val" "$task" "$it" "$summary_from_display" "$effective_ddp" "${batch_size:-}" "$preset_train" "$eval_profile"

    case "$step" in
        3)
            exec python "$py_d4c" step3 --task "$task" --preset "$preset_train" \
                "${iter_args[@]}" "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp_ws[@]+"${extra_ddp_ws[@]}"}" \
                "${extra_hw[@]+"${extra_hw[@]}"}"
            ;;
        4)
            exec python "$py_d4c" step4 --task "$task" --preset "$preset_train" \
                "${iter_args[@]}" --from-run "$from_run" --eval-profile "$eval_profile" \
                "${extra_ddp_ws[@]+"${extra_ddp_ws[@]}"}" \
                "${extra_hw[@]+"${extra_hw[@]}"}"
            ;;
        5)
            exec python "$py_d4c" step5 --task "$task" --preset step5 \
                "${iter_args[@]}" --from-run "$from_run" --step5-run auto \
                --step4-run "$step4_run_cli" \
                --eval-profile "$eval_profile" \
                "${extra_bs[@]+"${extra_bs[@]}"}" "${extra_ddp_ws[@]+"${extra_ddp_ws[@]}"}" \
                "${extra_hw[@]+"${extra_hw[@]}"}"
            ;;
    esac
}
