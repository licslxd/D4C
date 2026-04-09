# D4C 预设体系（v3 / manifest 4.3）

配置分为两类：**真正参与 merge 的配置层** 与 **编排层 `eval_profile`**。  
**CLI 显式传入的非空参数一律最后覆盖** preset 合并结果。`manifest.json`（**schema 4.3**）记录 `consumed_presets`、`config_before_cli`、`training_semantic_fingerprint`（训练主身份）、`generation_semantic_fingerprint`（生成/评测语义，未消费时可为 null）。

## `runtime_env`（manifest / `config_resolved.json`）

**唯一**承载 OMP/MKL/TOKENIZERS/CUDA_VISIBLE_DEVICES 等编排层观测，四段结构：

- **`thread_env_requested`** / **`thread_env_effective`**：解析链得到的请求值与生效值（字符串化标量）。
- **`launcher_env_requested`** / **`launcher_env_effective`**：如 `CUDA_VISIBLE_DEVICES`（launcher-only，不参与 semantic fingerprint）。

`hyperparameters` **不再**重复写入线程或 CUDA 镜像字段；排障请读 `runtime_env` 或 `[startup_runtime_env]` / `[Resolved Outputs]` 中的同口径摘要。

## 真正参与 merge 的配置层

| 层 | 目录 | 职责 |
|----|------|------|
| task | `presets/tasks/` | 任务事实：`auxiliary` / `target` / `lr` / `coef` / `adv` |
| training | `presets/training/` | 训练语义：`train_batch_size`、`epochs`、调度等 |
| hardware | `presets/hardware/` | `num_proc`、`ddp_world_size`、DataLoader workers、OMP/MKL、`tokenizers_parallelism`、**`cuda_visible_devices`（launcher-only，不进语义 JSON）** 等 |
| decode | `presets/decode/` | 生成策略（与 `default.yaml` 浅合并） |
| rerank | `presets/rerank/` | rerank 权重与编排字段（仅 eval-rerank*） |

### 新版字段合同（强制）

- `training` 允许：`train_batch_size`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`epochs`、训练损失/调度/label 长度等；禁止：`eval_batch_size`、`num_return_sequences`、hardware 字段。
- `hardware` 允许：`ddp_world_size`、`num_proc`、workers/prefetch、`omp_num_threads`、`mkl_num_threads`、`tokenizers_parallelism`、`cuda_visible_devices`（字符串，如 `"0,1"`；由 `d4c.py` 注入子进程 `CUDA_VISIBLE_DEVICES`，**不参与** `D4C_HARDWARE_PROFILE_JSON` 与 training/generation semantic fingerprint）等；禁止：训练/评测 batch 与 `num_return_sequences`。
- `eval_profile` 允许：`hardware_preset`、`decode_preset`、`rerank_preset`（可选）、`eval_batch_size`、`num_return_sequences`；禁止：`train_batch_size`、`per_device_train_batch_size`、`gradient_accumulation_steps`。
- `decode` 仅 decode 语义，禁止 batch 字段。
- `rerank` 仅 rerank 打分语义，禁止 `num_return_sequences` 与 batch 字段。

## 编排层：`eval_profile`（不是 merge 主链的一层）

目录：`presets/eval_profiles/`。**职责**：选择 `hardware_preset` / `decode_preset` /（仅 rerank 命令）`rerank_preset`，并提供 **profile-owned** 字段：

- `eval_batch_size`
- `num_return_sequences`（**仅** `eval-rerank` / `eval-rerank-matrix`；plain `eval` 若写此键会报错）

YAML **仅允许**上述引用键与 profile-owned 键；禁止把整段 hardware/decode/rerank 参数块拷进 profile。

解析顺序：**先** `resolve_eval_profile`（若提供 `--eval-profile`），**再**按下列顺序加载各 preset YAML 并合并，**最后** CLI 覆盖。

## 合并顺序（按命令）

- **step3**：`task → training → hardware → CLI`（**不**加载 decode 预设文件）
- **step4**（**eval 语义侧**，与 eval 共用 eval_batch strict 合同）：**须** `resolve_eval_profile`（CLI **`--eval-profile` 必填**）→ `task → training → hardware → CLI`（**不**加载 decode 预设文件）。  
  推理全局 batch **仅**来自 `eval_profile.eval_batch_size`（须能被选中 hardware 的 `ddp_world_size` 整除）；**禁止**用 `training.train_batch_size` 作为 step4 推理 batch。
- **step5（默认，含训练后 valid）**：`task → training → hardware → decode → CLI`  
  - **`--train-only`**：不加载 decode 预设文件；使用内置占位 decode 标量仅满足训练侧 JSON 注入；**不**写入 `generation_semantic_fingerprint`，`manifest.generation_semantic_resolved` 为 null。
- **eval / eval-matrix**：`resolve eval_profile?` → `task → training（上下文）→ hardware → decode → profile_owned → CLI`  
  - 未使用 `--eval-profile` 时，须同时提供 `--hardware-preset` 与 `--decode-preset`。
- **eval-rerank / eval-rerank-matrix**：在上式之后加载 **rerank**，再 **CLI**。

## CLI 要点

- `--preset`：训练 YAML。
- `--hardware-preset`：`presets/hardware/<stem>.yaml`（子进程经 `D4C_HARDWARE_PROFILE_JSON` 消费）。
- `--decode-preset`：decode 叠加层（step5 非 train-only 时默认 `decode_greedy_default`）。
- `--eval-profile`：`presets/eval_profiles/<stem>.yaml`（**step4 必填**；eval 系推荐；**selector**，与 hardware/decode 不同级）。
- `eval_batch_size` 与 `num_return_sequences` 只允许写在 `eval_profile`，不再提供 public CLI 覆盖入口。

## Step5 主线默认（减法重构）

- `presets/training/step5.yaml` 主线默认 `train_label_max_length=64`。
- 训练标签 padding 主线为 `train_dynamic_padding=true`（batch 内动态补齐），不再走固定 128 长度全局 pad。
- `loss_weight_repeat_ul` / `loss_weight_terminal_clean` / `terminal_clean_span` 仅允许在 **training preset** 配置；`decode preset` 不再注入训练损失。
- rerank 仅定位为 `eval-rerank*` 可选增强路径，非 train/eval 主线必经。
- manifest / config_resolved / train log 会显式记录 `train_label_max_length`、dynamic padding 策略和训练辅助损失权重；运行环境仅见 **`runtime_env`**（见上文）。

## 指纹与实验身份

- **`training_semantic_fingerprint`**：训练主实验身份（`training_payload` 不含 `eval_batch_size` + `hardware_profile` 语义切片 + DDP + `train_label_max_length`）；**不含** decode、rerank、eval_profile 名；**不含** `omp_num_threads` / `mkl_num_threads` / `tokenizers_parallelism` / `cuda_visible_devices` / `runtime_env`（上述属编排层 launcher env，见 manifest `runtime_env`）。
- **`generation_semantic_fingerprint`**：生成/评测侧（decode 摘要、eval batch、rerank 摘要、`num_return_sequences` 等）。step3 为空；**step4** 含 **`eval_batch_size`**（来自 eval_profile）；step5 `--train-only` 为空。

同一 Step5 训练可换不同 decode 再跑 `eval`，**训练指纹不变、生成指纹变**。

## 官方 stem 速查

**hardware**：`default`、`hw_1gpu_fast`、`hw_1gpu_throughput`、`hw_2gpu_balanced`、`hw_2gpu_quality`、`hw_debug_smoke`

**decode**：`decode_greedy_default`、`decode_balanced_v2`、`decode_diverse_v2`、`decode_conservative_v2`、`decode_rerank_quality_v2`

**rerank**：`rerank_v3_default`、`rerank_v3_quality_first`、`rerank_v3_softclean`（及 `default.yaml` 底）

**eval_profile**：`eval_fast_single_gpu`、`eval_balanced_2gpu`、`eval_rerank_quality`、`eval_rerank_probe`

## 子进程与单一真相源

- 训练语义：`D4C_EFFECTIVE_TRAINING_PAYLOAD_JSON`（`schema_version: 2`，含 `auxiliary` / `target`）。
- 硬件：`D4C_HARDWARE_PROFILE_JSON` + `D4C_HARDWARE_PRESET`（hardware stem）。
- 指纹：`D4C_TRAINING_SEMANTIC_FINGERPRINT`、`D4C_GENERATION_SEMANTIC_FINGERPRINT`（有则注入）、`D4C_RUNTIME_DIAGNOSTICS_FINGERPRINT`。
- 评测编排名：`D4C_EVAL_PROFILE_NAME`（**step4** / eval* 使用 `--eval-profile` 时）；step4 另注入 `D4C_GLOBAL_EVAL_BATCH_SIZE`、`D4C_EVAL_PER_GPU_BATCH_SIZE`（与 manifest 中 `global_eval_batch_size` / `eval_per_gpu_batch_size` 一致）。
- 子进程 **不**依赖父 shell 的 `D4C_HARDWARE_PRESET` 去重新选 YAML；须由 `d4c.py` 注入 JSON。

## 矩阵运行追踪

`eval-matrix` / `eval-rerank-matrix` 在子 run 的 manifest 中写入 `matrix_session_id`、`matrix_cell_id`、`invoked_command`（如 `eval-matrix`）与 `cell_command`（如 `eval`）。`eval_profile_detail.orchestrator_yaml` 记录 selector 声明（若使用 profile）。

## 校验

```bash
python scripts/check_presets.py
```
