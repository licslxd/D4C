# D4C

**一句话**：面向跨域推荐场景的 **可反事实解释生成**（PyTorch + Transformers）。  
**主入口（唯一推荐）**：在仓库根目录执行 **`python code/d4c.py …`**。

更完整的运行规范、Shell 参数与路径约定见 **`docs/D4C_Scripts_and_Runtime_Guide.md`**；**配置与预设**见 **`docs/PRESETS.md`**；离线数据与集群补充见根目录 **`D4C_离线完整指南.md`**。

---

## 快速开始（Quick Start）

以下命令均在 **项目根**（含 `code/`、`presets/` 的目录）执行。

```bash
# Step3 — 域对抗（默认 train + 收尾 eval）；产物在 runs/task{T}/vN/train/step3/<run>/（auto 默认 1、2、…）
python code/d4c.py step3 --task 4 --preset step3 --iter v1

# Step4 — 反事实推理（eval 语义；须 --eval-profile；产物在 train/step4/<step4-run>/）
python code/d4c.py step4 --task 4 --preset step3 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu

# Step5 — 主模型训练（非 --train-only 须 --eval-profile；--step5-run auto 时须 --step4-run）
python code/d4c.py step5 --task 4 --preset step5 --iter v1 --from-run 2 --step4-run 2_1 --step5-run auto \
  --eval-profile eval_fast_single_gpu

# Eval — Step5 权重 valid 评测；产物在 runs/.../eval/<run>/
# 推荐：--eval-profile（编排 hardware+decode+eval_batch_size；见 presets/eval_profiles/）
python code/d4c.py eval --task 4 --preset step5 --iter v1 --from-run 2 --step5-run 2_1_1 \
  --eval-profile eval_balanced_2gpu

# Pipeline — 串联 Step3 → Step4 → Step5（须 --eval-profile；Step5 预设固定为 step5）
python code/d4c.py pipeline --task 4 --preset step3 --iter v1 --eval-profile eval_fast_single_gpu
```

帮助：`python code/d4c.py -h` 及各子命令 `python code/d4c.py step3 -h`。

**Task4 可复制模板（非主线入口）**：`scripts/templates/`（见该目录 `README.md`）。

---

## 目录结构（你需要知道的）

| 路径 | 作用 |
|------|------|
| **`code/d4c.py`** | 主线入口：解析预设、校验、打印摘要、`torchrun` 调度各阶段 runner |
| **`code/d4c_core/`** | 配置解析、校验、manifest、dispatch、路径解析 |
| **`code/executors/`** | Step3/4/5 业务实现（engine + torchrun 入口） |
| **`presets/`** | `tasks/`、`training/`、`hardware/`、`decode/`、`rerank/`、`eval_profiles/` YAML，见 **PRESETS.md** |
| **`scripts/entrypoints/`** | 单阶段 Shell 编排（`step3.sh`…`train_ddp.sh`）；仅调用 `d4c.py`，不承载训练业务默认 |
| **`scripts/orchestration/`** | 多 seed、nohup 监控等场景编排 |
| **`scripts/lib/`** | Shell 公共库（路径预测、`train_lib.sh` 等） |
| **`scripts/entrypoints/train_ddp.sh`** | 官方 Bash 批量入口（GPU/DDP 校验 → `d4c.py`；完整 3→4→5 时调用 `d4c.py pipeline`） |
| **`docs/`** | 主指南、`PRESETS.md`、其它说明 |
| **`runs/`** | **正式实验产物根**：`task{T}/vN/train|eval|rerank|matrix|analysis|meta`（默认被 `.gitignore`） |
| **`cache/`** | HF datasets 等缓存（`cache/task{T}/hf/`，默认被 `.gitignore`） |
| **`checkpoints/`、`log/`** | 旧版布局目录；**新版主线不再写入**（若本地仍存在可仅作 legacy） |
| **`legacy/`** | 考古：`legacy/code/`、`legacy/sh/`、`legacy/tools/`；见 **`legacy/README.md`** |

数据与预训练权重目录见离线指南；本 README 不展开论文背景。

### `runs/` 内 global、task、`multi_seed` 边界

- **`runs/global/vN/meta/`** 只放**跨任务**、无法归属单一 `task{T}` 的产物：`eval_registry_all.*`、多任务编排器的 `shell_logs`、跨任务批处理摘要等。**禁止**写入单任务 eval/rerank 结果、单任务 analysis pack、bad cases、`multi_seed`、单任务 shell 日志。
- **`runs/task{T}/vN/meta/`** 放该任务一切可归属元数据：`eval_registry.*`、`shell_logs`、`multi_seed/<run>/`（Step5 多 seed 的 shell tee）等。
- **Step5 `multi_seed`**：默认目录为 `runs/task{T}/vN/meta/multi_seed/<run>/`。显式指定：`bash scripts/orchestration/step5_multi_seed.sh … --multi-seed-run 5` 或 `export D4C_MULTI_SEED_RUN_ID=5`。

**示例路径**

- `runs/global/v1/meta/eval_registry_all.jsonl`
- `runs/task4/v1/meta/eval_registry.jsonl`
- `runs/task4/v1/meta/multi_seed/1/train_seed_3407.log`

---

## 预设（presets）摘要

分层与完整合并顺序见 **`docs/PRESETS.md`**。要点：

- **task / training / hardware / decode / rerank** 为 merge 层；**`--eval-profile`** 为**编排层**（选 preset + 少量评测字段，不是与 hardware 同级的 YAML 合并层）；**CLI 非空参数最后覆盖**。
- **`--hardware-preset`** → `presets/hardware/*.yaml`（注入 `D4C_HARDWARE_PROFILE_JSON`）。
- **`--eval-profile`** → `presets/eval_profiles/*.yaml`（**step4 必填**；eval* 与 step5 内置评测推荐；eval* 亦可显式 `--hardware-preset` + `--decode-preset`）。
- **step3 / step4** 不加载 decode 预设文件。
- 训练 batch 只改 `presets/training/*.yaml`；**step4 / eval*** 的推理全局 batch 只改对应 **`eval_profile` 的 `eval_batch_size`**（不再使用 training 的 `train_batch_size` 作为 step4 推理 batch）。

详见 **`docs/PRESETS.md`**（含示例与「该改哪里」）。

### 子进程环境与 `TRAIN_*`

`d4c_core.runners` 在启动 **torchrun** 前会清洗父 shell 中的 `TRAIN_*`、`EVAL_BATCH_SIZE`、`MAX_PARALLEL_CPU` 以及**全部** `D4C_*`，再**显式注入** `D4C_HARDWARE_PROFILE_JSON`、`D4C_HARDWARE_PRESET`、线程环境、decode/rerank JSON、指纹 `D4C_TRAINING_SEMANTIC_FINGERPRINT` / `D4C_GENERATION_SEMANTIC_FINGERPRINT`、评测侧 `D4C_EVAL_PROFILE_NAME`（**step4** / eval* 使用 `--eval-profile` 时）、step4 另注入 `D4C_GLOBAL_EVAL_BATCH_SIZE` / `D4C_EVAL_PER_GPU_BATCH_SIZE`，以及 `D4C_STAGE_RUN_DIR` / `D4C_EVAL_RUN_DIR` 等。**请勿依赖父 shell `export` 驱动主线**；请以 **`python code/d4c.py` 的 CLI + `presets/`**（评测首推 **`--eval-profile`**）为准。

---

## 输出产物

| 产物 | 位置 | 说明 |
|------|------|------|
| **训练/评测 run 根** | `runs/task{T}/vN/train/step3/<run>/`、`.../train/step4/<run>/`、`.../train/step5/<run>/`、`.../eval/<run>/`、`.../rerank/<run>/` | 以启动时打印的 `stage_run_dir` / `eval_run_dir` 为准；`manifest.json` 含 `paths` 与 `run_lineage` |
| **日志** | Step3：**`logs/train.log`** + 同目录 **`logs/eval.log`**（收尾 eval / `--eval-only`）；Step5：**`logs/train.log`**；Step4：**`train/step4/<run>/logs/step4.log`**；独立 eval 命令：**`eval/<run>/logs/eval.log`** | 与 `log_dir` 一致 |
| **Manifest** | **与当次 run 同目录的 `manifest.json`** | 每次 `torchrun` **前**写入（可关）；**`runtime_env`** 为唯一运行环境记录；**`hyperparameters.full_bleu_eval_resolved`** 给出当次 full BLEU 调度 |
| **`config_resolved.json`** | **Step3 / Step5 等训练 run 根目录** | 解析后的 **`FinalTrainingConfig`** 快照；含 **`full_bleu_eval_resolved`** 与 **`runtime_env`**（唯一运行环境记录；OMP/MKL/TOKENIZERS/CUDA） |
| **Eval 指标与预测** | `eval`/`rerank` run 目录下 `metrics.json`（含 **`eval_performance.summary` 耗时分解**）、`predictions.*`、`eval_digest.log`（**[Eval Timing Summary]**）；任务级汇总注册表见 `runs/task{T}/vN/meta/eval_registry.{txt,jsonl,csv}`，全局汇总见 `runs/global/vN/meta/eval_registry_all.*` | 旧名 `eval_runs*` 已废弃 |
| **Phase 汇总** | `runs/.../vN/matrix/<run>/phase1_summary.*`、`phase2_rerank_summary.*`、`matrix_manifest.json` | 由 `eval-summary` / `rerank-summary` / `eval-matrix` / `eval-rerank-matrix` 写出；JSON 内 `metrics_semantics` 区分 `repo_metrics` 与 `paper_metrics` 列 |
| **AI 分析包** | `runs/task{T}/vN/analysis/packNN/`（仅任务树内） | `eval` / `eval-rerank` / `eval-matrix` / `eval-rerank-matrix` 结束后默认导出（`--analysis-pack off` 可关） |
| **Step5 multi_seed shell tee** | `runs/task{T}/vN/meta/multi_seed/<run>/train_seed_*.log` | `scripts/orchestration/step5_multi_seed.sh`；与 `train/step5/<run>/logs/train.log` 并存 |
| **Step5 训练 CSV** | **`train/step4/{step5_run 去掉末段}/factuals_counterfactuals.csv`**（如 `step5/2_1_1` → `step4/2_1/`）；软链到 Step5 run | 详见 **PRESETS.md**、**D4C_Scripts_and_Runtime_Guide** |
| **Step5 权重** | **`.../train/step5/<run>/model/model.pth`** | `--model-path` 须指向该布局下的 `model.pth` |

关闭 manifest JSON：`export D4C_WRITE_RUN_MANIFEST=0`（stdout 摘要仍会打印）。

### 第 1–4 组实验（新版）对比用最小字段

脚本或笔记中建议固定记录（可从当次 `manifest.json`、`metrics.json`、`phase*_summary.json`、`matrix_manifest.json` 抽取）：

- `task_id`、`iteration_id`
- `step3_run`、`step4_run`、`step5_run`（目录 slug）
- `eval_run_dir` 或 `rerank_run_dir`（及其中 `metrics_json_path`）
- `rerank_enabled`、`decode_preset`、`rerank_preset`、`training_semantic_fingerprint`、`generation_semantic_fingerprint`、`metrics_schema_version`（**4.0**）
- `repo_metrics`、`paper_metrics`（**分开读**，勿混用扁平旧 `metrics` 当论文表）

---

## 排障入口（按顺序）

1. **`python code/d4c.py -h`** 与子命令 `-h`  
2. **终端 stdout**：`[Resolved Inputs/Outputs]`、`[Manifest]` 行  
3. **`manifest.json`**（与本次 run 目录同路径）  
4. **`train.log` / `eval.log`（step3 同目录）** / **`step4.log`** / 独立 eval run 的 **`eval.log`**（按阶段）  
5. **`docs/D4C_Scripts_and_Runtime_Guide.md`**（路径、Shell、`DDP_NPROC` 等）  
6. **`docs/PRESETS.md`**（配置从哪来、为何与预期不一致）  

高级调试（**非新手必会**）：`D4C_DISPATCH_DETAIL=1` 可打印 `torchrun` 实际入口脚本路径，见主指南。

---

## Shell / 集群（可选）

日常交互式实验 **直接用 `d4c.py`** 即可。需要 **多步/多任务批量** 时优先 **`scripts/entrypoints/train_ddp.sh`**；单阶段编排见 **`scripts/entrypoints/`**。说明见主指南 §1。

---

## 许可与引用

以仓库内随附的许可与论文引用说明为准（如有单独文件请以其为准）。
