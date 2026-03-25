# sh/ 脚本参数说明

本文档汇总 `sh/` 目录下各 Bash 脚本的**必填参数**、**可选参数及其默认行为**，以及「由 Python / torchrun / config 决定」时的**具体效果**。

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

  **Step 3 / Step 5 / Step 4 主日志文件（`--log_file`）**：在 **`get_log_task_dir(task)`** 下使用 **`runs/<秒级时间戳>/train.log`**（`_LOG_TS` 为 `date +%Y%m%d_%H%M%S`）；**`D4C_LOG_USE_TIMESTAMP=0`** 时为 **`runs/run/train.log`**（固定覆盖）。Step 3 传入 **`AdvTrain.py --log_file`**；Step 5 传入 **`run-d4c.py --log_file`**；Step 4 传入 **`generate_counterfactual.py --log_file`**（**`PerfMonitor`** / **`append_log_dual`**）。前台**不对**该 `train.log` 再 `tee`。

  **后台单任务**（`--daemon` / `--bg`）：`nohup` 重定向到与 `train.log` **同目录**的 **`nohup.log`**；子进程默认 **`D4C_CONSOLE_LEVEL=WARNING`**，减少与文件日志重复（可用环境变量覆盖）。

- **扁平日志**：Step 1+2、`run_step3_to_step5_*.sh` 使用 **`log/` 根下带前缀的文件名**；Step 3 / Step 4 在 **`--all` 且 `--daemon`** 时另有终端汇总 **`step3_daemon_*.log`** / **`step4_daemon_*.log`**；Step 3 **`--eval-only`** 时为 **`step3_eval_daemon_*.log`**。**`run_step5_all.sh`** 整段输出 **`tee`** 到 **`log/step5_all_*.log`**（前台与 **`--daemon`** 均会生成该汇总文件）。**`run_step5.sh`** 仅单任务，**无** `--all`，不产生旧版 **`step5_daemon_*.log`** 命名。

- **`run_step3.sh` 的默认**：未手动设置 `D4C_CHECKPOINT_SUBDIR` 时，Step 3 会设 `GROUP=step3`、`SUBDIR=step3_<时间戳>`，**`train.log`** 在 **`log/<task>/step3/runs/…/`**。**`run_step5.sh`** **仅嵌套 checkpoint**（**`D4C_CHECKPOINT_GROUP=step3`**），但默认 **`export D4C_LOG_GROUP=step5`**（仅当调用前未设 **`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR` / `D4C_LOG_STEP`**），故 **`train.log`** 在 **`log/<task>/step5/runs/…/`**，与 Step 3 日志分离；不再使用 **`checkpoints/<task>/step5/step5_*`**。**`run_step5.sh`** 默认 **`export HF_EVALUATE_OFFLINE=1`**（若环境中已设置则保留原值）。**`--eval-only`** 与 **`--train-only`** **互斥**。

- **工程优化脚本（`*_optimized.sh`）**：与复现脚本**并行存在**，不修改后者默认值。**`run_step3_optimized.sh`** → **`run_step4_optimized.sh`**（须 **`--step3-subdir`** 与 Step3 的 **`step3_opt_*`** 一致）→ **`run_step5_optimized.sh`**。**`run_step3_to_step5_*.sh` / `run_step5_all.sh`** 仍只调用 **`run_step3.sh` / `run_step4.sh` / `run_step5.sh`**，不自动使用 optimized 目录。

- **torchrun 解析**：`run_step3.sh` / **`run_step4.sh`** / `run_step5.sh` 及 **`run_step*_optimized.sh`** 优先使用 **`torchrun`**（**`run_step5_all.sh`** 经 **`run_step5.sh`** 间接相同）；若未安装该命令但可 **`import torch`**，则回退 **`python -m torch.distributed.run`**。

- **镜像日志**：**`D4C_MIRROR_LOG=1`** 时，`append_log_dual` 可额外写入 **`code/log.out`**；主文件仍以脚本传入的 `--log_file`（`**runs/…/train.log**`）为准。

---

## 一、参数顺序说明

- **可选参数之间顺序无要求**：脚本遍历整段命令行参数，谁先谁后均可。
- **带值的选项必须成对出现**：`--选项 值` 必须紧挨，例如 `--task 2`、`--from 4`。不能写成 `2 --task`。
- **`run_step3_to_step5_all.sh`**、**`run_step5_all.sh`**：未知参数会被静默丢弃（`*) shift ;;`），其它脚本无此逻辑。

---

## 二、各脚本必填与默认

**小节顺序**：与流水线一致 — **Step 1+2** → **Step 3**（含 **`run_step3_optimized.sh`**）→ **Step 4**（含 **`run_step4_optimized.sh`**）→ **Step 5**（含 **`run_step5_optimized.sh`**、`run_step5_all.sh`）→ **串联脚本**。

### run_step1_step2.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无 | 可零参数运行：`bash run_step1_step2.sh` |
| 可选 | `--embed-batch-size N` | 不传则由 config 决定（见下文） |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

---

### run_step3.sh

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
| 可选 | `--ddp-nproc K` | 不传则用环境变量 `DDP_NPROC`，再缺省为 2 |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

**任务超参（auxiliary / target / lr / coef / adv）**：由 **`code/config.py`** 的 **`task_configs`** 经 **`format_step3_task_params_line`** 生成。若 **`export D4C_TRAIN_PRESET=<键名>`**，**`TRAINING_PRESETS`** 可为**全局**一条 dict 或**按任务 1–8** 分条；后者下各任务的 **`adv` / `train_batch_size` / `epochs` / `full_eval_every_epochs` / `min_lr_ratio`** 可分别覆盖（**`run_step3.sh`** 每任务设 **`D4C_PRESET_TASK_ID`** 以解析 shell 侧 **`get_epochs`**）。优先级低于 CLI 与 **`D4C_MIN_LR_RATIO` / `D4C_FULL_EVAL_EVERY`** 等。详见 **`config.py`** 中 **`TRAINING_PRESETS`** 注释。

**Step 3 与结构化日志**：`run_step3.sh` 使用 **`d4c_step3_logfile`**（与 **`d4c_step5_logfile`** 规则一致）：**`LOGFILE=<get_log_task_dir(task)>/runs/<YYYYMMDDHHMMSS>/train.log`**；**`D4C_LOG_USE_TIMESTAMP=0`** 时为 **`…/runs/run/train.log`**，传给 **`AdvTrain.py --log_file`**；前台**不对** `LOGFILE` 再 `tee`。**`--daemon`** 单任务时另有同目录 **`nohup.log`**。**`D4C_MIRROR_LOG=1`** 时可再镜像 **`code/log.out`**。**`--all --daemon`** 时另有 **`log/step3_daemon_*.log`**（**`--eval-only`** 时为 **`step3_eval_daemon_*.log`**）。

---

### run_step3_optimized.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **与 `run_step3.sh` 相同** | `--all` / `--task N` / `--eval-only` / `--train-only` / `--batch-size` / … | 全部原样转发给 **`run_step3.sh`** |

**本脚本在 `exec` 前设置（可用环境变量覆盖）**：**`D4C_TRAIN_MODE=optimized`**、**`D4C_LR_SCHEDULER=warmup_cosine`**、**`D4C_WARMUP_RATIO`**（默认 0.05）、**`D4C_QUICK_EVAL_MAX_SAMPLES`**（默认 512）、**`TRAIN_EARLY_STOP_PATIENCE_FULL`**（默认 4）、**`TRAIN_MIN_EPOCHS`**（默认 8）、**`TRAIN_EARLY_STOP_PATIENCE`**（默认 6）、**`TRAIN_BLEU4_MAX_SAMPLES`**（默认 512）。**`D4C_FULL_EVAL_EVERY`**：脚本**不再**默认 `export`；未设置且未使用命名预设中的 **`full_eval_every_epochs`** 时，由 Python 采用**分阶段 full BLEU**（默认 epoch≤10 每 5 轮、之后每 2 轮，见 `config.resolve_full_bleu_eval_training`）；需要固定间隔时显式 **`export D4C_FULL_EVAL_EVERY=N`**，或使用 **`D4C_TRAIN_PRESET`**（见上节）。**Checkpoint**：默认 **`D4C_CHECKPOINT_GROUP=step3_optimized`**、**`D4C_CHECKPOINT_SUBDIR=step3_opt_<分秒时间戳>`**（若调用前已 `export D4C_CHECKPOINT_SUBDIR` 则保留）。**日志**：默认 **`D4C_LOG_GROUP=step3_optimized`**（若已设置 **`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR` / `D4C_LOG_STEP`** 则不覆盖）。**大 batch**：未出现 **`--batch-size`** 时**追加** **`--batch-size`**，默认值为 **`python -c "from config import get_train_batch_size; …"`**（模块默认 **2048**；**`D4C_TRAIN_PRESET`** 可改变该默认），仍可用 **`D4C_OPT_BATCH_SIZE`** 覆盖。

---

### run_step4.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--all` 或 `--task N` | 二选一，必须指定其一 |
| 可选 | `--from N` | 仅配合 `--all`，默认 1 |
| 可选 | `--skip N,M,...` | 默认不跳过 |
| 可选 | `--batch-size N` | 不传则由 config 决定；**全局 batch 须能被 `DDP_NPROC` 整除** |
| 可选 | `--num-proc N` | 不传则由 config 决定 |
| 可选 | `--ddp-nproc K` | 不传则用环境变量 **`DDP_NPROC`**，再缺省为 **2** |
| 可选 | `--daemon` / `--bg` | 不传则前台执行；单任务时 **`nohup.log`** 与 **`train.log`** 同目录；**`--all`** 时另有 **`log/step4_daemon_*.log`** |

**Step 4 与 DDP**：默认 **`torchrun --standalone --nproc_per_node=$DDP_NPROC` `generate_counterfactual.py`**；须与可见 GPU 数一致；多卡 **`CUDA_VISIBLE_DEVICES`**。单卡：**`DDP_NPROC=1`**。手动单进程（不推荐）：`cd code && python generate_counterfactual.py …`，与上述 shell 默认 torchrun 路径不同。

**Step 4 与结构化日志**：脚本默认 **`export D4C_LOG_STEP=step4`**（仅当调用前**未设置**该变量；若 **`export D4C_LOG_STEP=`** 为空字符串则主日志改随 **`D4C_CHECKPOINT_*`** 同表）。主日志 **`get_log_task_dir(task)/runs/<时间戳>/train.log`**，经 **`generate_counterfactual.py --log_file`** 写入 **`PerfMonitor`**；**不对**该文件再 **`tee`**（与 Step 3/5 一致）。**`D4C_MIRROR_LOG=1`** 时可再镜像 **`code/log.out`**；未传 **`--log_file`** 时 Python 默认仍写当前工作目录 **`save.log`**。

---

### run_step4_optimized.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **与 `run_step4.sh` 相同** | `--all` / `--task N` / `--batch-size` / … | 全部原样转发给 **`run_step4.sh`**（**`--step3-subdir` 会剥离，不传给 Python**） |
| **必填（本脚本）** | `--step3-subdir NAME` | 与 **`checkpoints/<task>/step3_optimized/<NAME>/`** 一致；用于设置 **`D4C_CHECKPOINT_SUBDIR`** |

**本脚本在 `exec` 前设置**：**`D4C_TRAIN_MODE=optimized`**、**`D4C_CHECKPOINT_GROUP=step3_optimized`**、**`D4C_CHECKPOINT_SUBDIR=<NAME>`**。若未预先设置 **`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR`**，则 **`D4C_LOG_STEP=step4_optimized`**（主日志 **`log/<task>/step4_optimized/…`**）。未出现 **`--batch-size`** 时**追加** **`--batch-size`**，默认 **`config.get_train_batch_size()`**（**`D4C_OPT_BATCH_SIZE`** 仍可覆盖）。

---

### run_step5.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--task N` | 1–8 |
| **必填** | `--step3-subdir NAME` | 与 **`checkpoints/<task>/step3/<NAME>/`** 一致；须已有 **`factuals_counterfactuals.csv`** |
| 可选 | `--eval-only` | 仅 valid 上 FINAL RESULTS；**须** **`--nested-subdir`**；与 `--train-only` 互斥 |
| 可选 | `--train-only` | 训练后跳过收尾 valid；与 `--eval-only` 互斥 |
| 可选 | `--nested-subdir 内层名` | 训练时可选（默认 **`step5_<YYYYMMDD_HHMM>`**）；**`--eval-only` 必填** |
| 可选 | `--batch-size N` | 不传则由 config 决定 |
| 可选 | `--epochs N` | 不传则由 config 决定（**`--eval-only`** 时不传给 Python） |
| 可选 | `--num-proc N` | 不传则由 config 决定 |
| 可选 | `--ddp-nproc K` | 不传则用环境变量 `DDP_NPROC`，再缺省为 2 |
| 可选 | `--daemon` / `--bg` | 单任务后台；同目录 **`nohup.log`** |

**已移除**：**`--all`**、**`--from`**、**`--skip`**；若传入 **`--all`** 脚本报错退出。

**Checkpoint**：仅 **`…/step3/<NAME>/step5/<内层>/`**；**`D4C_CHECKPOINT_GROUP=step3`**，**`SUBDIR=<NAME>/step5/<内层>`**；csv 软链 **`../../factuals_counterfactuals.csv`**。

**`run_step3_to_step5_single.sh` / `run_step3_to_step5_all.sh`**：调用 Step 5 前自动取 **`step3_*`**（**`ls -1td`** 最新）；**`--eval-only`** 时再取 **`step5_*`** 内层最新。

**Step 5 与主日志**：默认 **`log/<task>/step5/runs/<YYYYMMDDHHMMSS>/train.log`**（**`get_log_task_dir`** 因 **`D4C_LOG_GROUP=step5`**）；**`D4C_LOG_USE_TIMESTAMP=0`** 时为 **`log/<task>/step5/runs/run/train.log`**。每任务 eval 汇总在 **`log/<task>/step5/eval/`**；全局 eval 汇总在 **`log/step5/eval/`**（**`train_logging`** 优先读 **`D4C_LOG_GROUP`**）。**`D4C_MIRROR_LOG=1`** 时可再写 **`code/log.out`**。

---

### run_step5_optimized.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **与 `run_step5.sh` 相同** | `--task N`、`--step3-subdir`、`--nested-subdir`、… | 参数表一致 |

**与 `run_step5.sh` 的差异**：物理目录 **`checkpoints/<task>/step3_optimized/<NAME>/`**（**`--step3-subdir`** 须为该目录下文件夹名，通常由 **`run_step3_optimized.sh`** 生成）；**`D4C_CHECKPOINT_GROUP=step3_optimized`**；训练默认内层 **`step5_opt_<YYYYMMDD_HHMM>`**（**`--eval-only`** 时仍须 **`--nested-subdir <已有内层>`**）。主日志默认 **`log/<task>/step5_optimized/`**；**`torchrun run-d4c.py`** 额外传入 **`--train-mode optimized`**、**`--min-epochs`**、**`--early-stop-patience-full`**、**`--quick-eval-max-samples`**、**`--bleu4-max-samples`**、**`--warmup-ratio`**；仅当已设置 **`D4C_FULL_EVAL_EVERY`** 时再传入 **`--full-eval-every`**（否则与 Step3 optimized 一致，由 Python 分阶段 full BLEU 或命名预设决定）。未传 **`--batch-size`** 且非 **`--eval-only`** 时默认 **`--batch-size`** 为 **`config.get_train_batch_size()`**（**`D4C_OPT_BATCH_SIZE`** 仍可覆盖）。

---

### run_step5_all.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无 | 默认对 task **1–8** 依次执行（可用 **`--from`** / **`--skip`** 缩小） |
| 可选 | `--from N` | 从任务 N 起跑，默认 **1** |
| 可选 | `--skip N,M,...` | 逗号分隔，跳过所列任务 |
| 可选 | `--eval-only` | 每任务 **`run_step5.sh --eval-only`**；**须** 各任务已有 **`…/step3/<NAME>/step5/step5_*`**（脚本取最新内层 **`step5_*`**）；与 `--train-only` 互斥 |
| 可选 | `--train-only` | 每任务传入 **`--train-only`**；与 `--eval-only` 互斥 |
| 可选 | `--batch-size` / `--epochs` / `--num-proc` / `--ddp-nproc` | 原样传给每任务的 **`run_step5.sh`** |
| 可选 | `--daemon` / `--bg` | **`nohup`** 整段后台；汇总仍写入 **`log/step5_all_*.log`** |

**每任务**：自动 **`--step3-subdir`** = **`checkpoints/<task>/step3/`** 下 **`ls -1td step3_*`** 最新目录名；**`--eval-only`** 时再 **`--nested-subdir`** = 该 Step 3 目录下 **`step5/step5_*`** 最新内层名。

**汇总日志**：每次运行（含前台）在 **`log/step5_all_<时间>.log`** 中 **`tee`** 各任务子进程输出；单任务结构化日志仍以 **`log/<task>/step5/runs/…/train.log`** 为准。

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

**Step 5 路径**：调用 **`run_step5.sh`** 时自动传入 **`--step3-subdir`**（**`checkpoints/<task>/step3/`** 下最新 **`step3_*`**）；**`--eval-only`** 时再传 **`--nested-subdir`**（**`…/step3/<NAME>/step5/`** 下最新 **`step5_*`**）。

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

**Step 5 路径**：每任务同上（最新 **`step3_*`** / **`step5_*`**）传入 **`run_step5.sh`**。

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
| `TRAINING_PRESETS` / **`D4C_TRAIN_PRESET`** | 见 **`config.py`** | 命名预设：可为**全局**一条 dict，或**按任务** `1..8` → 字段 dict（各任务可不同）。**`gb1024_ep30_fe2`** 为按任务表，默认 8 项相同（batch **1024**、**30** epoch、每 **2** epoch full BLEU、**`min_lr_ratio=0.1`**、**`adv=0.005`**）。**`run_step3.sh`** / **`run_step5*.sh`** 通过 **`D4C_PRESET_TASK_ID`** 解析 epochs / batch；按任务预设时 **`run_step3_optimized.sh`** 等可不统一注入 **`--batch-size`** |
| `num_proc` | **min(有效 CPU 核数, MAX_PARALLEL_CPU)** | 仅 **`datasets.map`（Tokenize）** 等并行进程数 |
| `MAX_PARALLEL_CPU` | 环境变量，**默认 12** | 与常见 GPU 节点 `nproc≈12` 对齐；更大机器可 `export MAX_PARALLEL_CPU=16` |
| DataLoader workers | `get_dataloader_num_workers(train/valid/test)` | 与 `num_proc` **独立**，上限随 `MAX_PARALLEL_CPU` 收紧 |

其中「有效 CPU 核数」来自 `cpu_utils.effective_cpu_count()`：优先 `sched_getaffinity`（反映 cgroup/cpuset 限制），否则 `os.cpu_count()`；环境变量 `RUNNING_CPU_COUNT` 可显式覆盖（例如与 shell `nproc` 一致）。

---

### 3.2 各 Python 脚本的承接逻辑

| 脚本 | 参数 | 未传时的来源 | 实际取值 |
|------|------|--------------|----------|
| **run_preprocess_and_embed.py** | `--embed-batch-size` | `config.get_embed_batch_size()` | **1024** |
| **compute_embeddings.py** | batch size | 环境变量 `EMBED_BATCH_SIZE` 或 `config.get_embed_batch_size()` | **1024**（由 run_preprocess 注入或 config） |
| **generate_counterfactual.py** | `--batch-size` | `config.get_train_batch_size()` | **2048** |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |
| **AdvTrain.py** | `--batch-size` | `config.get_train_batch_size()` | **2048** |
| | `--epochs` | `config.get_epochs()` | **50** |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |
| **run-d4c.py** | `--batch-size` | `config.get_train_batch_size()` | **2048** |
| | `--epochs` | `config.get_epochs()` | **50** |
| | eval 阶段 batch（无单独 CLI 时常走 config） | `config.get_eval_batch_size()` | **2560**（或 **`EVAL_BATCH_SIZE`**） |
| | `--eval-only` | 由 `run_step5.sh` 传入时 | 跳过训练，加载 checkpoint 后在 valid 上输出 FINAL RESULTS |
| | `--train-only` | 由 `run_step5.sh` 传入时 | 训练后跳过收尾 valid 评估（epoch 内 valid 仍可有） |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |

---

### 3.3 torchrun / DDP 相关

| 项目 | 未传时的来源 | 实际取值 |
|------|--------------|----------|
| **DDP 进程数** | 环境变量 `DDP_NPROC` | 脚本默认 **2**（`run_step3.sh` / `run_step5.sh` 中 `DDP_NPROC="${DDP_NPROC:-2}"`） |
| **可见 GPU** | `CUDA_VISIBLE_DEVICES` | 若未设置，使用系统默认可见 GPU |
| **全局 batch** | config 或 `--batch-size` | 默认 **2048**（`train_batch_size`）；须能被 `DDP_NPROC` 整除，否则脚本会报错 |

单卡时建议：`DDP_NPROC=1` 或 `--ddp-nproc 1`，并确保 `CUDA_VISIBLE_DEVICES` 只包含一块 GPU。

---

### 3.4 小结：不传任何可选参数时的实际效果

| Step | 参数 | 实际效果 |
|------|------|----------|
| Step 1+2 | 全不传 | embed batch=1024，单卡，`num_proc`=min(CPU 核数,12)（可调 `MAX_PARALLEL_CPU`） |
| Step 3/5 | 全不传 | batch=2048，epochs=50，DDP 进程=2，`num_proc`=min(CPU 核数,12) |
| Step 4 | 全不传 | batch=2048，**DDP 进程数=2**（`DDP_NPROC` / `--ddp-nproc`），`num_proc`=min(CPU 核数,12)；**绑卡**由 **`CUDA_VISIBLE_DEVICES`** + rank 决定 |

显存不足时，可显式减小 `--batch-size`（如 64、128、256）或 `--embed-batch-size`；**Step 4（`run_step4.sh`）与 Step 3/5** 多卡请 **`CUDA_VISIBLE_DEVICES`** 与 **`DDP_NPROC` / `--ddp-nproc`**。
