# sh/ 脚本参数说明

本文档汇总 `sh/` 目录下各 Bash 脚本的**必填参数**、**可选参数及其默认行为**，以及「由 Python / torchrun / config 决定」时的**具体效果**。

**统一入口**：**`scripts/train_ddp.sh`**（**`scripts/train_lib.sh`**）支持 **`--step 3|4|5`**、**`--pipeline 3,4,5`**（子集去重后按 3→4→5 执行）、**`--task`**、**`--gpus`**（仅 `CUDA_VISIBLE_DEVICES`）、**`--ddp-nproc`**（`DDP_NPROC` + 透传子脚本）、**`--train-preset` / `--runtime-preset`**、**`--batch-size`**、**`--step3-subdir`**；启动前 fail-fast（如可见 GPU 数 ≥ `DDP_NPROC`、`batch-size` 可整除、`step3` 目录存在等）。**`DDP_NPROC` 仅执行层**，与 **`D4C_NUM_PROC`** 区分见 **`docs/D4C_RUNTIME_SPEC.md`**。**`run_step3_to_step5_*.sh`** 仍可直接使用，且已对 Step 4 透传 DDP。

---

## `checkpoints/` 与 `log/` 路径（全局）

- **项目根 `D4C_ROOT`**：各脚本通过 `sh/` 相对路径解析得到，默认即含 `data/`、`code/`、`checkpoints/`、`log/` 的目录；可用环境变量 **`D4C_ROOT`** 覆盖（须保持与数据、预训练模型等路径一致）。

- **Checkpoint 根目录**：`<D4C_ROOT>/checkpoints/`。单任务目录由 **`code/paths_config.get_checkpoint_task_dir(task)`** 决定，依赖 **`D4C_CHECKPOINT_GROUP`**、**`D4C_CHECKPOINT_SUBDIR`**：

  | GROUP | SUBDIR | `checkpoints/<task>/…` |
  |-------|--------|------------------------|
  | 空 | 空 | `checkpoints/<task>/` |
  | 空 | 非空 | `checkpoints/<task>/<SUBDIR>/` |
  | 非空 | 空 | `checkpoints/<task>/<GROUP>/` |
  | 非空 | 非空 | `checkpoints/<task>/<GROUP>/<SUBDIR>/` |

  常见文件：**`model.pth`**（Step 3 / 5）；**`factuals_counterfactuals.csv`**（Step 4）。

- **日志根目录**：`<D4C_ROOT>/log/`。**`get_log_task_dir(task)`** 解析优先级见 **`paths_config`**：**`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR`**（仅影响日志，语义与 checkpoint 两变量对称；双非空时目录为 **`log/<task>/<D4C_LOG_GROUP>/`**）→ 否则 **`D4C_LOG_STEP`**（非空时为 **`log/<task>/<STEP>/`**）→ 否则仍按 **`D4C_CHECKPOINT_*`**（与 checkpoint **不完全对称**：GROUP 与 SUBDIR 均非空时，权重在 `…/<GROUP>/<SUBDIR>/`，主日志固定在 **`log/<task>/<GROUP>/`**，不按 SUBDIR 再分层；eval 在同级 **`eval/`**）。

  **Step 3 / Step 5 / Step 4 主日志文件（`--log_file`）**：在 **`get_log_task_dir(task)`** 下使用 **`runs/<秒级时间戳>/train.log`**（`_LOG_TS` 为 `date +%Y%m%d_%H%M%S`）；**`D4C_LOG_USE_TIMESTAMP=0`** 时为 **`runs/run/train.log`**（固定覆盖）。Step 3 传入 **`AdvTrain.py --log_file`**；Step 5 传入 **`run-d4c.py --log_file`**；Step 4 传入 **`generate_counterfactual.py --log_file`**（**`PerfMonitor`** / **`append_log_dual`**）。**不要**再对 `train.log` **`tee`**；**`run_step3_optimized.sh --all` 前台**的 **`tee`** 默认写 **`log/step3_optimized_all_*.log`**；路径可用 **`D4C_STEP3_ALL_SHELL_LOG`** 覆盖。

  **后台单任务**（`--daemon` / `--bg`）：`nohup` 重定向到与 `train.log` **同目录**的 **`nohup.log`**；子进程默认 **`D4C_CONSOLE_LEVEL=WARNING`**，减少与文件日志重复（可用环境变量覆盖）。

- **扁平日志**：Step 1+2、`run_step3_to_step5_*.sh` 使用 **`log/` 根下带前缀的文件名**；**`run_step3_optimized.sh --all` 前台**另有 **`log/step3_optimized_all_*.log`**（默认，可由 **`D4C_STEP3_ALL_SHELL_LOG`** 覆盖；shell **`tee`**，与各任务 **`train.log`** 分离）；Step 3 / Step 4 在 **`--all` 且 `--daemon`** 时另有终端汇总 **`step3_optimized_daemon_*.log`** / **`step4_optimized_daemon_*.log`**；Step 3 **`--eval-only`** 时为 **`step3_optimized_eval_daemon_*.log`**。**`run_step5_all.sh`** 整段输出 **`tee`** 到 **`log/step5_all_*.log`**。**`run_step5_optimized.sh`** 仅单任务、**无** `--all`，**`--daemon`** 时无全任务汇总文件名（每任务日志在 **`log/<task>/step5_optimized/runs/…`**）。

- **Step 3 / 4 / 5 正式入口**：**`run_step3_optimized.sh`**（默认 **`GROUP=step3_optimized`**、**`SUBDIR=step3_opt_<时间戳>`**，主日志 **`log/<task>/step3_optimized/`**）→ **`run_step4_optimized.sh`**（**必填 `--step3-subdir`**，与 Step 3 目录名一致；主日志默认 **`step4_optimized`**）→ **`run_step5_optimized.sh`**（嵌套 **`step3_optimized/<NAME>/step5/step5_opt_*`**，主日志默认 **`step5_optimized`**）。**`run_step5_optimized.sh`** 默认 **`export HF_EVALUATE_OFFLINE=1`**（若环境中已设置则保留）。**`--eval-only`** 与 **`--train-only`** **互斥**。

- **torchrun 解析**：上述脚本优先 **`torchrun`**（**`run_step5_all.sh`** 经 **`run_step5_optimized.sh`** 间接相同）；若无该命令但可 **`import torch`**，则回退 **`python -m torch.distributed.run`**。

- **镜像日志**：**`D4C_MIRROR_LOG=1`** 时，`append_log_dual` 可额外写入 **`code/log.out`**；主文件仍以脚本传入的 `--log_file`（`**runs/…/train.log**`）为准。

---

## 一、参数顺序说明

- **可选参数之间顺序无要求**：脚本遍历整段命令行参数，谁先谁后均可。
- **带值的选项必须成对出现**：`--选项 值` 必须紧挨，例如 `--task 2`、`--from 4`。不能写成 `2 --task`。
- **`run_step3_to_step5_all.sh`**、**`run_step5_all.sh`**：未知参数**报错退出**；**`--gpus`** 显式拒绝。其它 `run_step*_optimized.sh` 对未识别 token 仍可能静默忽略（勿依赖误拼参数被提示）。

---

## 二、各脚本必填与默认

**小节顺序**：与流水线一致 — **Step 1+2** → **Step 3**（含 **`run_step3_optimized.sh`**）→ **Step 4**（含 **`run_step4_optimized.sh`**）→ **Step 5**（含 **`run_step5_optimized.sh`**、`run_step5_all.sh`）→ **串联脚本**；**`smoke_test_ddp.sh`** 为独立自检脚本，见本节末尾。

### run_step1_step2.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无 | 可零参数运行：`bash run_step1_step2.sh` |
| 可选 | `--embed-batch-size N` | 不传则由 config 决定（见下文） |
| 可选 | `--cuda-device N` | 传给 `compute_embeddings.py`；不传则默认 **cuda:0**（无 CUDA 时 CPU） |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

---

### run_step3_optimized.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--all` 或 `--task N` | 二选一，必须指定其一 |
| 可选 | `--from N` | 仅配合 `--all`，默认 1 |
| 可选 | `--skip N,M,...` | 默认不跳过 |
| 可选 | `--eval-only` | 仅 `AdvTrain.py` eval（跳过 train）；与 `--train-only` 互斥；默认关 |
| 可选 | `--train-only` | 仅 train，跳过训练后的 `AdvTrain.py` eval；与 `--eval-only` 互斥；默认关 |
| 可选 | `--batch-size N` | 不传则由 config 决定 |
| 可选 | `--epochs N` | 不传则由 config 决定 |
| 可选 | `--num-proc N` | 不传则由 config 决定 |
| 可选 | `--ddp-nproc K` | 不传则用环境变量 `DDP_NPROC`，再缺省为 2（**`=1` 为单卡 DDP smoke，仍为同一主路径**） |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

**`AdvTrain.py eval`**：与 train 相同，**仅**在 **`torchrun` / `python -m torch.distributed.run`** 下运行；不支持 `python AdvTrain.py eval`。

**任务超参（auxiliary / target / lr / coef / adv）**：shell 侧由 **`run_step3_optimized.sh`** 的 **`get_task_params`** 从 **`config.TASK_DEFAULTS`**（与 **`task_configs`** 同表）读取；**不含**命名预设合并。Python 训练入口由 **`build_resolved_training_config`** 按 **BASE → TASK_DEFAULTS → TRAINING_PRESETS → ENV → CLI** 解析。若 **`export D4C_TRAIN_PRESET=<键名>`**，**`TRAINING_PRESETS`** 可为**全局**一条 dict 或**按任务 1–8** 分条；后者下各任务的 **`adv` / `train_batch_size` / `epochs` / `full_eval_every_epochs` / `min_lr_ratio`** 可分别覆盖（**`run_step3_optimized.sh`** 每任务设 **`D4C_PRESET_TASK_ID`** 以解析 shell 侧 **`get_epochs`**）。详见 **`config.py`** 中 **`TRAINING_PRESETS`** 注释。

**Step 3 与结构化日志**：**`d4c_step3_logfile`**：**`LOGFILE=<get_log_task_dir(task)>/runs/<YYYYMMDDHHMMSS>/train.log`**；**`D4C_LOG_USE_TIMESTAMP=0`** 时为 **`…/runs/run/train.log`**，传给 **`AdvTrain.py --log_file`**；**`--all` 前台**下 **`tee`** 默认写入 **`log/step3_optimized_all_<秒级时间戳>.log`**（**`D4C_STEP3_ALL_SHELL_LOG`** 可覆盖），**不**与 `LOGFILE`（`train.log`）同路径。**`--daemon`** 单任务时另有同目录 **`nohup.log`**。**`D4C_MIRROR_LOG=1`** 时可再镜像 **`code/log.out`**。**`--all --daemon`** 时另有 **`log/step3_optimized_daemon_*.log`**（**`--eval-only`** 时为 **`step3_optimized_eval_daemon_*.log`**）。

**脚本内默认 export（可用环境变量覆盖）**：**`D4C_LR_SCHEDULER=warmup_cosine`**、**`D4C_WARMUP_RATIO`**（0.05）、**`D4C_QUICK_EVAL_MAX_SAMPLES`**（512）、**`TRAIN_EARLY_STOP_PATIENCE_FULL`**（4）、**`TRAIN_MIN_EPOCHS`**（8）、**`TRAIN_EARLY_STOP_PATIENCE`**（6）、**`TRAIN_BLEU4_MAX_SAMPLES`**（512）。**`D4C_FULL_EVAL_EVERY`** 未设置时由 **`build_resolved_training_config`** 内建分阶段 full BLEU 默认。**大 batch**：未出现 **`--batch-size`** 时：若 **`D4C_OPT_BATCH_SIZE`** 有值则注入；否则若 **`training_preset_is_per_task()`** 为真则不注入（由 **`AdvTrain`** 按任务解析）；否则注入 **`get_train_batch_size()`**（模块默认 **2048**）。**Bash 4.2 + `set -u`** 下对空数组展开做了兼容。

**Runtime 并发（可选）**：**`export D4C_RUNTIME_PRESET=<键名>`** 选用 **`config.RUNTIME_PRESETS`**（如 **`gpu01_single_12c`**、**`gpu01_ddp2_12c`**），解析链为 **runtime_base → preset → `MAX_PARALLEL_CPU` / `D4C_NUM_PROC` / `D4C_DATALOADER_WORKERS_*` 等 ENV → CLI（`--num-proc` 等）**；与训练 **`D4C_TRAIN_PRESET`** 独立。**`FinalTrainingConfig`** 含 **`max_parallel_cpu`**、**`runtime_preset_name`**、各 split 的 **`dataloader_num_workers_*`** 与 **`dataloader_prefetch_factor_*`**。OMP/MKL/Tokenizers 线程请在 shell 中设置（脚本头有推荐示例）。

**`AdvTrain.py` / `run-d4c.py` 的 `--learning_rate`**：可不传（Python 侧默认 **`None`**）；优化器初始学习率在 **`config.build_resolved_training_config`** 中按 **BASE → `TASK_DEFAULTS.lr` → 命名预设 `lr` → `D4C_INITIAL_LR` → `--learning_rate` → `--scheduler-initial-lr`**（后者覆盖前者）解析，结果写入冻结的 **`FinalTrainingConfig`**。CLI 由 **`training_runtime_inputs.collect_training_runtime_overrides_from_args`** 收集，不写回 **`os.environ`**。

---

### run_step4_optimized.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--step3-subdir NAME` | 与 **`checkpoints/<task>/step3_optimized/<NAME>/`** 一致；设置 **`D4C_CHECKPOINT_SUBDIR`**（**不**传给 Python） |
| **必填** | `--all` 或 `--task N` | 二选一 |
| 可选 | `--from N` | 仅配合 `--all`，默认 1 |
| 可选 | `--skip N,M,...` | 默认不跳过 |
| 可选 | `--batch-size N` | 不传则按上节规则注入；**全局 batch 须能被 `DDP_NPROC` 整除** |
| 可选 | `--num-proc N` | 不传则由 config 决定 |
| 可选 | `--ddp-nproc K` | 不传则用 **`DDP_NPROC`**，再缺省 **2** |
| 可选 | `--daemon` / `--bg` | **`--all`** 时汇总 **`log/step4_optimized_daemon_*.log`**；单任务 **`nohup.log`** 与 **`train.log`** 同目录 |

**环境与路径**：**`D4C_CHECKPOINT_GROUP=step3_optimized`**。若未预先设置 **`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR`**，则 **`D4C_LOG_STEP=step4_optimized`**（仅当该变量**未设置**时；若 **`export D4C_LOG_STEP=`** 为空字符串则主日志随 **`D4C_CHECKPOINT_*`**）。**DDP**：**`torchrun`** + **`generate_counterfactual.py`**。**`nohup` 子进程**若未带 **`--step3-subdir`**，会从已继承的 **`D4C_CHECKPOINT_SUBDIR`** 恢复。

---

### run_step5_optimized.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--task N` | 1–8 |
| **必填** | `--step3-subdir NAME` | 与 **`checkpoints/<task>/step3_optimized/<NAME>/`** 一致；须已有 **`factuals_counterfactuals.csv`** |
| 可选 | `--eval-only` | 仅 valid 上 FINAL RESULTS；**须** **`--nested-subdir`**；与 `--train-only` 互斥 |
| 可选 | `--train-only` | 训练后跳过收尾 valid；与 `--eval-only` 互斥 |
| 可选 | `--nested-subdir 内层名` | 训练时可选（默认 **`step5_opt_<YYYYMMDD_HHMM>`**）；**`--eval-only` 必填** |
| 可选 | `--batch-size N` | 不传则由 config / 脚本内默认注入（见下） |
| 可选 | `--epochs N` | 不传则由 config 决定（**`--eval-only`** 时不传给 Python） |
| 可选 | `--num-proc N` | 不传则由 config 决定 |
| 可选 | `--ddp-nproc K` | 不传则用 **`DDP_NPROC`**，再缺省为 2 |
| 可选 | `--daemon` / `--bg` | 单任务后台；同目录 **`nohup.log`** |

**已移除**：**`--all`**、**`--from`**、**`--skip`**；若传入 **`--all`** 脚本报错退出。

**Checkpoint**：**`…/step3_optimized/<NAME>/step5/<内层>/`**；**`D4C_CHECKPOINT_GROUP=step3_optimized`**，**`SUBDIR=<NAME>/step5/<内层>`**；内层目录内 csv 软链 **`../../factuals_counterfactuals.csv`**。

**脚本内默认 export**（与 Step 3 对齐，可覆盖）：**`D4C_LR_SCHEDULER`**、**`D4C_WARMUP_RATIO`**、**`TRAIN_*`**、**`D4C_QUICK_EVAL_MAX_SAMPLES`** 等。**`torchrun run-d4c.py`** 另传 **`--min-epochs`**、**`--early-stop-patience(-full)`**、**`--quick-eval-max-samples`**、**`--bleu4-max-samples`**、**`--warmup-ratio`**；仅当已设置 **`D4C_FULL_EVAL_EVERY`** 时再传 **`--full-eval-every`**。

**主日志**：未预先设置 **`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR` / `D4C_LOG_STEP`** 时 **`D4C_LOG_GROUP=step5_optimized`** → **`log/<task>/step5_optimized/runs/…/train.log`**；eval 汇总在 **`log/<task>/step5_optimized/eval/`** 等（见 **`train_logging`**）。

**`run_step3_to_step5_*.sh` / `run_step5_all.sh`**：每任务自动 **`--step3-subdir`** = **`checkpoints/<task>/step3_optimized/`** 下最新 **`step3_opt_*`**；**`--eval-only`** 时再 **`--nested-subdir`** = 该 Step 3 目录下 **`step5/step5_opt_*`** 最新内层名。

---

### run_step5_all.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无 | 默认对 task **1–8** 依次执行（可用 **`--from`** / **`--skip`** 缩小） |
| 可选 | `--from N` | 从任务 N 起跑，默认 **1** |
| 可选 | `--skip N,M,...` | 逗号分隔，跳过所列任务 |
| 可选 | `--eval-only` | 每任务 **`run_step5_optimized.sh --eval-only`**；**须** 各任务已有 **`…/step3_optimized/<NAME>/step5/step5_opt_*`**（脚本取最新内层）；与 `--train-only` 互斥 |
| 可选 | `--train-only` | 每任务传入 **`--train-only`**；与 `--eval-only` 互斥 |
| 可选 | `--batch-size` / `--epochs` / `--num-proc` / `--ddp-nproc` | 原样传给每任务的 **`run_step5_optimized.sh`** |
| 可选 | `--daemon` / `--bg` | **`nohup`** 整段后台；汇总仍写入 **`log/step5_all_*.log`** |

**每任务**：自动 **`--step3-subdir`** = **`checkpoints/<task>/step3_optimized/`** 下 **`ls -1td step3_opt_*`** 最新目录名；**`--eval-only`** 时再 **`--nested-subdir`** = 该 Step 3 目录下 **`step5/step5_opt_*`** 最新内层名。

**汇总日志**：每次运行（含前台）在 **`log/step5_all_<时间>.log`** 中 **`tee`** 各任务子进程输出；单任务结构化日志仍以 **`log/<task>/step5_optimized/runs/…/train.log`** 为准。

---

### run_step3_to_step5_single.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--task N` | N 为 1–8 |
| 可选 | `--from 3\|4\|5` | 默认 3（从 Step 3 开始） |
| 可选 | `--eval-only` | **Step 3** 与 **Step 5** 均只 eval；Step 4 不变；与 `--train-only` 互斥；默认关 |
| 可选 | `--train-only` | **Step 3** 与 **Step 5** 均带 `--train-only`（跳过收尾 eval）；与 `--eval-only` 互斥；默认关 |
| 可选 | `--batch-size` / `--epochs` / `--num-proc` / `--ddp-nproc` | 不传则传给子脚本或由 config / 默认 DDP 决定 |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

**Step 5 路径**：调用 **`run_step5_optimized.sh`** 时自动传入 **`--step3-subdir`**（**`checkpoints/<task>/step3_optimized/`** 下最新 **`step3_opt_*`**）；**`--eval-only`** 时再传 **`--nested-subdir`**（**`…/step3_optimized/<NAME>/step5/`** 下最新 **`step5_opt_*`**）。

---

### run_step3_to_step5_all.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无模式选项 | 固定跑任务 1–8（可被 `--from` / `--skip` 缩小） |
| 可选 | `--from N` | 默认 1 |
| 可选 | `--skip N,M,...` | 默认不跳过 |
| 可选 | `--eval-only` | 每个任务的 **Step 3** 与 **Step 5** 均只 eval；Step 4 不变；与 `--train-only` 互斥 |
| 可选 | `--train-only` | 每个任务的 **Step 3** 与 **Step 5** 均带 `--train-only`；与 `--eval-only` 互斥 |
| 可选 | `--batch-size` / `--epochs` / `--num-proc` / `--ddp-nproc` | 按子脚本转发；不传则由 config 或默认 DDP 决定 |
| 可选 | `--daemon` / `--bg` | 不传则前台执行；后台时 stdout/stderr 写入 **`log/step3_to_5_all_*.log`** |

**Step 5 路径**：每任务同上（最新 **`step3_opt_*`** / **`step5_opt_*`**）传入 **`run_step5_optimized.sh`**。

**未知参数**：**`run_step3_to_step5_all.sh`**、**`run_step5_all.sh`** 对未识别选项会**报错退出**（不再静默丢弃）。**`--gpus`** 显式拒绝。

---

### smoke_test_ddp.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无 | 在项目根执行：`bash sh/smoke_test_ddp.sh`；在 `code/` 内调用 `torchrun` |
| 环境 | `CUDA_VISIBLE_DEVICES` | 与单进程 `nproc_per_node=1` 搭配即可；**仍为 DDP** |
| 产物 | `D4C_CHECKPOINT_GROUP=smoke_ddp` | 每次运行生成唯一 **`D4C_CHECKPOINT_SUBDIR`**；可事后删除 `checkpoints/1/smoke_ddp/` 与 `log/1/smoke_ddp/` |

**注意**：不评 BLEU/指标；Step 5 仍跑 **1 个完整 epoch**（数据大时可能较慢）。Step 3 smoke train 使用 **`AdvTrain.py train --max-steps 2 --save-final-checkpoint`**（`--max-steps` 在 epoch 内提前退出时默认不会触发按 metric 写盘；最终 checkpoint 依赖该 flag）。train 结束后脚本会检查 **`checkpoints/1/smoke_ddp/<tag>/model.pth`** 存在后再跑 eval。

---

## 三、「由 Python / torchrun / config 决定」时的具体效果

当脚本未传入某参数时，会以**空字符串**或**不传**的形式交给下层。下层逻辑如下：

### 3.1 config.py 中的默认值

| 配置项 | 默认值 | 用途 |
|--------|--------|------|
| `embed_batch_size` | **1024** | Step 1+2 嵌入计算的 batch size |
| `train_batch_size` | **2048** | 模块默认；运行时以 **`get_train_batch_size()`** 为准（**`D4C_TRAIN_PRESET`** 可覆盖）。Step 3/4/5 训练与推理 batch；显存不足可 **`--batch-size`** 减小 |
| `eval_batch_size` | **2560**（可用环境变量 **`EVAL_BATCH_SIZE`** 覆盖） | Step 5 **eval** 阶段全局 batch；DDP 下每 rank = 该值 / `world_size`；OOM 时可减小 |
| `epochs` | **50** | 模块默认；运行时以 **`get_epochs()`** 为准（**`D4C_TRAIN_PRESET`** 可覆盖）。Step 3 域对抗、Step 5 主训练轮数 |
| `TRAINING_PRESETS` / **`D4C_TRAIN_PRESET`** | 见 **`config.py`** | 命名预设：可为**全局**一条 dict，或**按任务** `1..8` → 字段 dict（各任务可不同）。**`step3`** / **`step5`** 为同构按任务表（**`presets/training/step3.yaml`**、**`step5.yaml`**；batch **1024**、**30** epoch、每 **2** epoch full BLEU、**`min_lr_ratio=0.1`**、**`adv=0.005`**）。**`run_step3_optimized.sh`** 默认 **`D4C_TRAIN_PRESET=step3`**，**`run_step5_optimized.sh`** 默认 **`step5`**；二者通过 **`D4C_PRESET_TASK_ID`** 解析 epochs / batch；按任务预设时可不统一注入 **`--batch-size`** |
| `num_proc` | **min(有效 CPU 核数, MAX_PARALLEL_CPU)** | 仅 **`datasets.map`（Tokenize）** 等并行进程数 |
| `MAX_PARALLEL_CPU` | 环境变量，**默认 12** | 与常见 GPU 节点 `nproc≈12` 对齐；更大机器可 `export MAX_PARALLEL_CPU=16` |
| DataLoader workers | `get_dataloader_num_workers(train/valid/test)` | 与 `num_proc` **独立**，上限随 `MAX_PARALLEL_CPU` 收紧 |

其中「有效 CPU 核数」来自 `cpu_utils.effective_cpu_count()`：优先 `sched_getaffinity`（反映 cgroup/cpuset 限制），否则 `os.cpu_count()`；环境变量 `RUNNING_CPU_COUNT` 可显式覆盖（例如与 shell `nproc` 一致）。

---

### 3.2 各 Python 脚本的承接逻辑

| 脚本 | 参数 | 未传时的来源 | 实际取值 |
|------|------|--------------|----------|
| **run_preprocess_and_embed.py** | `--embed-batch-size` | `config.get_embed_batch_size()` | **1024** |
| | `--cuda-device` | 不传则 `compute_embeddings` 用默认 **0** | 单进程单 GPU |
| **compute_embeddings.py** | batch size | 环境变量 `EMBED_BATCH_SIZE` 或 `config.get_embed_batch_size()` | **1024**（由 run_preprocess 注入或 config） |
| | `--cuda-device` | 不传则 **cuda:0**（无 CUDA 则 CPU） | **无** DataParallel / DDP |
| **generate_counterfactual.py** | `--batch-size` | `config.get_train_batch_size()` | **2048**；**须 `torchrun`**，全局 batch 须能被 `WORLD_SIZE` 整除 |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |
| **AdvTrain.py** | `--batch-size` | `config.get_train_batch_size()` | **2048** |
| | `--epochs` | `config.get_epochs()` | **50** |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |
| | `--save-final-checkpoint` | 默认不传 | **关**；为真时训练函数 **`finally`** 中 rank0 无条件写 **`save_file`（含 `--max-steps` 提前退出）** |
| **run-d4c.py** | `--batch-size` | `config.get_train_batch_size()` | **2048** |
| | `--epochs` | `config.get_epochs()` | **50** |
| | eval 阶段 batch（无单独 CLI 时常走 config） | `config.get_eval_batch_size()` | **2560**（或 **`EVAL_BATCH_SIZE`**） |
| | `--eval-only` | 由 `run_step5_optimized.sh` 传入时 | 跳过训练，加载 checkpoint 后在 valid 上输出 FINAL RESULTS |
| | `--train-only` | 由 `run_step5_optimized.sh` 传入时 | 训练后跳过收尾 valid 评估（epoch 内 valid 仍可有） |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |

---

### 3.3 torchrun / DDP 相关

| 项目 | 未传时的来源 | 实际取值 |
|------|--------------|----------|
| **DDP 进程数** | 环境变量 `DDP_NPROC` | 脚本默认 **2**（`run_step3_optimized.sh` / `run_step5_optimized.sh` 中 `DDP_NPROC="${DDP_NPROC:-2}"`） |
| **可见 GPU** | `CUDA_VISIBLE_DEVICES` | 若未设置，使用系统默认可见 GPU |
| **全局 batch** | config 或 `--batch-size` | 默认 **2048**（`train_batch_size`）；须能被 `DDP_NPROC` 整除，否则脚本会报错 |

单卡时建议：`DDP_NPROC=1` 或 `--ddp-nproc 1`，并确保 `CUDA_VISIBLE_DEVICES` 只包含一块 GPU（**仍为 DDP 主路径**，非第二套非分布式实现）。

---

### 3.4 小结：不传任何可选参数时的实际效果

| Step | 参数 | 实际效果 |
|------|------|----------|
| Step 1+2 | 全不传 | embed batch=1024，单卡，`num_proc`=min(CPU 核数,12)（可调 `MAX_PARALLEL_CPU`） |
| Step 3/5 | 全不传 | batch=2048，epochs=50，DDP 进程=2，`num_proc`=min(CPU 核数,12) |
| Step 4 | 全不传 | batch=2048，**DDP 进程数=2**（`DDP_NPROC` / `--ddp-nproc`），`num_proc`=min(CPU 核数,12)；**绑卡**由 **`CUDA_VISIBLE_DEVICES`** + rank 决定 |

显存不足时，可显式减小 `--batch-size`（如 64、128、256）或 `--embed-batch-size`；**Step 4（`run_step4_optimized.sh`）与 Step 3/5** 多卡请 **`CUDA_VISIBLE_DEVICES`** 与 **`DDP_NPROC` / `--ddp-nproc`**。
