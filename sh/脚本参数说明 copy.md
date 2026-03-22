# sh/ 脚本参数说明

本文档汇总 `sh/` 目录下各 Bash 脚本的**必填参数**、**可选参数及其默认行为**，以及「由 Python / torchrun / config 决定」时的**具体效果**。

---

## `checkpoints/` 与 `log/` 路径（全局）

- **项目根 `D4C_ROOT`**：各脚本通过 `sh/` 相对路径解析得到，默认即含 `data/`、`code/`、`checkpoints/`、`log/` 的目录；可用环境变量 **`D4C_ROOT`** 覆盖（须保持与数据、预训练模型等路径一致）。

- **Checkpoint 根目录**：`<D4C_ROOT>/checkpoints/`。单任务目录由 **`code/paths_config.get_checkpoint_task_dir(task)`** 决定，依赖环境变量 **`D4C_CHECKPOINT_GROUP`**、**`D4C_CHECKPOINT_SUBDIR`**（与源码注释一致）：

  | GROUP | SUBDIR | 目录 |
  |-------|--------|------|
  | 空 | 空 | `checkpoints/<task>/` |
  | 空 | 非空 | `checkpoints/<task>/<SUBDIR>/` |
  | 非空 | 空 | `checkpoints/<task>/<GROUP>/` |
  | 非空 | 非空 | `checkpoints/<task>/<GROUP>/<SUBDIR>/` |

  常见文件：**`model.pth`**（Step 3 / 5 训练或 eval 加载）；**`factuals_counterfactuals.csv`**（Step 4 输出，与 `generate_counterfactual.py` 使用的 task 目录一致）。

- **日志根目录**：`<D4C_ROOT>/log/`。**`get_log_task_dir(task)`** 与 checkpoint **同一套** GROUP/SUBDIR 规则，只是把前缀换成 **`log/<task>/`**。Step 3 / Step 5 在常规前台或单任务后台运行时，主日志文件名为该目录下的 **`run.log`**，并由脚本传入 Python **`--log_file`**，与 `tee` 一致。

- **与「分层 `run.log`」并存的扁平日志**：Step 1+2、`run_step4.sh`、以及 `run_step3_to_step5_*.sh` 使用 **`log/` 下带前缀的文件名**（如 `step4_*.log`、`step3_to_5_all_*.log`），**不**经过 `get_log_task_dir`；Step 3 / Step 5 在 **`--all` 且 `--daemon`** 时，汇总日志为 **`log/step3_daemon_*.log`** / **`log/step5_daemon_*.log`**（仅 eval 时为 **`step3_eval_daemon_*.log`** / **`step5_eval_daemon_*.log`**）。

- **`run_step3.sh` / `run_step5.sh` 的默认**：未手动设置时，Step 3 会设 `GROUP=step3`、`SUBDIR=step3_<时间戳>`；Step 5 会设 `GROUP=step5`、`SUBDIR=step5_<时间戳>`。**`run_step5.sh --eval-only`** 要求**事先**设置 **`D4C_CHECKPOINT_SUBDIR`**（及按需 **`D4C_CHECKPOINT_GROUP`**），否则会报错退出（避免新建时间戳目录却找不到 `model.pth`）。

- **镜像日志**：**`D4C_MIRROR_LOG=1`** 时，`append_log_dual` 可额外写入 **`code/log.out`**；主文件仍以脚本传入的 `--log_file`（如 `run.log`）为准。

---

## 一、参数顺序说明

- **可选参数之间顺序无要求**：脚本遍历整段命令行参数，谁先谁后均可。
- **带值的选项必须成对出现**：`--选项 值` 必须紧挨，例如 `--task 2`、`--from 4`。不能写成 `2 --task`。
- **`run_step3_to_step5_all.sh`**：未知参数会被静默丢弃（`*) shift ;;`），其它脚本无此逻辑。

---

## 二、各脚本必填与默认

### run_step1_step2.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无 | 可零参数运行：`bash run_step1_step2.sh` |
| 可选 | `--embed-batch-size N` | 不传则由 config 决定（见下文） |
| 可选 | `--gpus 0,1` | 不传则单卡（默认 GPU 0） |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

---

### run_step3.sh / run_step5.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--all` 或 `--task N` | 二选一，必须指定其一 |
| 可选 | `--from N` | 仅配合 `--all`，默认 1 |
| 可选 | `--skip N,M,...` | 默认不跳过 |
| 可选 | `--eval-only` | **Step 3**：仅 `AdvTrain.py` eval；**Step 5**：仅 `run-d4c.py` 在 valid 上出 FINAL RESULTS，不训练；默认关 |
| 可选 | `--batch-size N` | 不传则由 config 决定 |
| 可选 | `--epochs N` | 不传则由 config 决定（**Step 5** 在 **`--eval-only`** 时由脚本省略，不传给 Python） |
| 可选 | `--num-proc N` | 不传则由 config 决定 |
| 可选 | `--gpus 0,1` | Step 3 用 torchrun 时忽略，用 `CUDA_VISIBLE_DEVICES` |
| 可选 | `--ddp-nproc K` | 不传则用环境变量 `DDP_NPROC`，再缺省为 2 |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

**Step 5 与 `--eval-only`**：须已有训练产物的 **`model.pth`**（路径由 `paths_config.get_checkpoint_task_dir` + 文件名决定，或由 `run-d4c.py` 的 **`--save-file`** 指定）。若未设置环境变量 **`D4C_CHECKPOINT_SUBDIR`**，**`run_step5.sh`** 会拒绝 **`--eval-only`**（避免自动新建带时间戳的子目录却找不到权重）；请先 **`export D4C_CHECKPOINT_SUBDIR=…`**（按需再设 **`D4C_CHECKPOINT_GROUP`**）指向已有目录。**`--all --daemon`** 且仅 eval 时，汇总日志文件名为 **`step5_eval_daemon_*.log`**。

**Step 3 与结构化日志**：`run_step3.sh` 为每次运行设置 **`LOGFILE=$(get_log_task_dir(<task>)/run.log)`**（见上文「`log/` 路径」），并传给 **`AdvTrain.py --log_file`**，与 **`tee`** 目标一致。`AdvTrain.py` 中由 Python 追加的内容（`[Tokenize]`、训练头、`Config (DDP)`、每 epoch 的 Loss/Valid 行、`PerfMonitor`、`FINAL RESULTS` 等）经 `append_log_dual` 写入该 **`--log_file`**；若启用 **`D4C_MIRROR_LOG=1`** 可再镜像到 **`code/log.out`**（路径相同时只写一次）。**`--all --daemon`** 时另有扁平汇总 **`log/step3_daemon_*.log`**（仅 eval 时为 **`step3_eval_daemon_*.log`**）。

**Step 5 与最终指标**：`run_step5.sh` 同样使用 **`log/<task>/…/run.log`** 作为 **`--log_file`** 与 **`tee`** 目标（规则同 Step 3）。`run-d4c.py` 的结构化输出写入该文件；**`D4C_MIRROR_LOG=1`** 时可再写 **`code/log.out`**。**`--all --daemon`** 时汇总为 **`log/step5_daemon_*.log`**（仅 eval 时为 **`step5_eval_daemon_*.log`**）。

---

### run_step4.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--all` 或 `--task N` | 二选一，必须指定其一 |
| 可选 | `--from N` | 仅配合 `--all`，默认 1 |
| 可选 | `--skip N,M,...` | 默认不跳过 |
| 可选 | `--gpus 0,1` | 不传则默认 `"0"`（单卡） |
| 可选 | `--batch-size N` | 不传则由 config 决定 |
| 可选 | `--num-proc N` | 不传则由 config 决定 |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

**Step 4 与结构化日志**：`generate_counterfactual.py` 使用 `PerfMonitor`，性能块与汇总表已走 `append_log_dual`，会进入 **`code/log.out`** 以及脚本内使用的 `log_file`（默认可为 `save.log`）。若日后为 Step 4 增加与 `log/step4_*.log` 对齐的 `--log_file` 并传入 `PerfMonitor`，行为与 Step 5 一致：结构化块双写，`tee` 仍只负责 stdout/stderr。

---

### run_step3_to_step5_single.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | `--task N` | N 为 1–8 |
| 可选 | `--from 3\|4\|5` | 默认 3（从 Step 3 开始） |
| 可选 | `--eval-only` | 打开时：**Step 3** 与 **Step 5** 均只 eval（不重训）；Step 4 行为不变；默认关 |
| 可选 | `--gpus` / `--batch-size` / `--epochs` / `--num-proc` / `--ddp-nproc` | 不传则传给子脚本或由 config 决定 |
| 可选 | `--daemon` / `--bg` | 不传则前台执行 |

---

### run_step3_to_step5_all.sh

| 类型 | 参数 | 说明 |
|------|------|------|
| **必填** | 无模式选项 | 固定跑任务 1–8（可被 `--from` / `--skip` 缩小） |
| 可选 | `--from N` | 默认 1 |
| 可选 | `--skip N,M,...` | 默认不跳过 |
| 可选 | `--eval-only` | 每个任务的 **Step 3** 与 **Step 5** 均只 eval（内部传 `run_step3.sh` / `run_step5.sh` 的 `--eval-only`）；Step 4 不变 |
| 可选 | 其余同 run_step3 / run_step5 | 不传则按子脚本或 config 决定 |

---

## 三、「由 Python / torchrun / config 决定」时的具体效果

当脚本未传入某参数时，会以**空字符串**或**不传**的形式交给下层。下层逻辑如下：

### 3.1 config.py 中的默认值

| 配置项 | 默认值 | 用途 |
|--------|--------|------|
| `embed_batch_size` | **1024** | Step 1+2 嵌入计算的 batch size |
| `train_batch_size` | **1024** | Step 3/4/5 训练与推理的 batch size |
| `epochs` | **50** | Step 3 域对抗、Step 5 主训练的轮数 |
| `num_proc` | **min(有效 CPU 核数, MAX_PARALLEL_CPU)** | 仅 **`datasets.map`（Tokenize）** 等并行进程数 |
| `MAX_PARALLEL_CPU` | 环境变量，**默认 12** | 与常见 GPU 节点 `nproc≈12` 对齐；更大机器可 `export MAX_PARALLEL_CPU=16` |
| DataLoader workers | `get_dataloader_num_workers(train/valid/test)` | 与 `num_proc` **独立**，上限随 `MAX_PARALLEL_CPU` 收紧 |

其中「有效 CPU 核数」来自 `cpu_utils.effective_cpu_count()`：优先 `sched_getaffinity`（反映 cgroup/cpuset 限制），否则 `os.cpu_count()`；环境变量 `RUNNING_CPU_COUNT` 可显式覆盖（例如与 shell `nproc` 一致）。

---

### 3.2 各 Python 脚本的承接逻辑

| 脚本 | 参数 | 未传时的来源 | 实际取值 |
|------|------|--------------|----------|
| **run_preprocess_and_embed.py** | `--embed-batch-size` | `config.get_embed_batch_size()` | **1024** |
| | `--gpus` | 不传 | 单卡（`cuda:0` 或 `cuda`） |
| **compute_embeddings.py** | batch size | 环境变量 `EMBED_BATCH_SIZE` 或 `config.get_embed_batch_size()` | **1024**（由 run_preprocess 注入或 config） |
| | `--gpus` | 不传 | 单卡 |
| **generate_counterfactual.py** | `--batch-size` | `config.get_train_batch_size()` | **1024** |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |
| | `--gpus` | 默认 `"0"` | 单卡 GPU 0 |
| **AdvTrain.py** | `--batch-size` | `config.get_train_batch_size()` | **1024** |
| | `--epochs` | `config.get_epochs()` | **50** |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |
| **run-d4c.py** | `--batch-size` | `config.get_train_batch_size()` | **1024** |
| | `--epochs` | `config.get_epochs()` | **50** |
| | `--eval-only` | 由 `run_step5.sh` 传入时 | 跳过训练，加载 checkpoint 后在 valid 上输出 FINAL RESULTS |
| | `--num-proc` | `config.get_num_proc()` | **min(CPU 核数, MAX_PARALLEL_CPU)**，默认上限 **12** |

---

### 3.3 torchrun / DDP 相关

| 项目 | 未传时的来源 | 实际取值 |
|------|--------------|----------|
| **DDP 进程数** | 环境变量 `DDP_NPROC` | 脚本默认 **2**（`run_step3.sh` / `run_step5.sh` 中 `DDP_NPROC="${DDP_NPROC:-2}"`） |
| **可见 GPU** | `CUDA_VISIBLE_DEVICES` | 若未设置，使用系统默认可见 GPU |
| **全局 batch** | config 或 `--batch-size` | **1024**；须能被 `DDP_NPROC` 整除，否则脚本会报错 |

单卡时建议：`DDP_NPROC=1` 或 `--ddp-nproc 1`，并确保 `CUDA_VISIBLE_DEVICES` 只包含一块 GPU。

---

### 3.4 小结：不传任何可选参数时的实际效果

| Step | 参数 | 实际效果 |
|------|------|----------|
| Step 1+2 | 全不传 | embed batch=1024，单卡，`num_proc`=min(CPU 核数,12)（可调 `MAX_PARALLEL_CPU`） |
| Step 3/5 | 全不传 | batch=1024，epochs=50，DDP 进程=2，`num_proc`=min(CPU 核数,12) |
| Step 4 | 全不传 | batch=1024，单卡 GPU 0，`num_proc`=min(CPU 核数,12) |

显存不足时，可显式减小 `--batch-size`（如 64、128、256）或 `--embed-batch-size`；多卡时可增大 batch 或配合 `--gpus 0,1`（Step 4）和 `CUDA_VISIBLE_DEVICES`（Step 3/5）。
