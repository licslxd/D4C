# D4C 最终版路径规范备忘

本文档与仓库内 `code/paths_config.py` 及各 `sh/*.sh` 行为一致，并与 **[`sh/用法速查 copy 3.md`](sh/用法速查%20copy%203.md)**、**[`sh/脚本参数说明 copy 3.md`](sh/脚本参数说明%20copy%203.md)** 中「`checkpoints/` 与 `log/` 路径」章节对齐（备忘侧重总览与排障，细节命令仍以该两份为准）。

---

## 1. 项目根目录 `D4C_ROOT`

- **默认**：`code/` 的上一级目录（即包含 `code/`、`data/`、`sh/` 的目录，例如 `D4C-main`）。
- **覆盖**：设置环境变量 `D4C_ROOT` 为绝对路径（集群/容器挂载时常用）。
- **Python**：`paths_config.get_d4c_root()`、`get_checkpoint_task_dir` / `get_log_task_dir` 在**运行时**读取 `D4C_ROOT` 与 `D4C_CHECKPOINT_SUBDIR` / `D4C_CHECKPOINT_GROUP`，不在 `import paths_config` 时把路径冻结成常量。

所有下文路径均相对于 `D4C_ROOT`，除非另行说明。

---

## 2. Shell 脚本约定

- 入口脚本在 **`sh/`**，执行前会 `cd` 到 **`code/`** 再调用 Python。
- 解析根目录方式：`D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"`，其中 `SH_DIR` 为当前 `.sh` 所在目录。
- **Slurm**：应在项目根下提交（见 `tmux-slurm/submit_d4c.sh` 注释），保证 `log/` 与相对路径正确。
- **可选**：**`D4C_RUNTIME_PRESET`** 在 **`code/config.py`** 的 **`RUNTIME_PRESETS`** 中统一 DataLoader / **`num_proc`** 等运行时并发；与 checkpoint / log 路径无关，但与 Step 3/4/5 主流程共用同一 **`config`** 解析（详见 **`sh/run_step3_optimized.sh`** 文件头注释与 **`sh/脚本参数说明 copy 3.md`**）。
- **收敛中入口**：**`scripts/train_ddp.sh`**（实现见 **`scripts/train_lib.sh`**）为统一 DDP 启动骨架，内部仍调用现有 **`sh/run_step*_optimized.sh`**；分层与变量语义见 **`docs/D4C_RUNTIME_SPEC.md`**。

---

## 3. 数据与中间产物

| 用途 | 路径 | 说明 |
|------|------|------|
| 原始/处理后按域存放 | `data/<数据集名>/` | 如 `user_profiles.npy`、`item_profiles.npy`、`domain.npy` 等；数据集名为任务里的域标识（如 `AM_Electronics`）。 |
| 按任务合并后的数据 | `Merged_data/<task>/` | `task` 为 `1`–`8` 字符串；含 `aug_train.csv` 等，供 Step 3/4/5 使用。 |

---

## 4. 预训练与缓存

| 内容 | 路径 |
|------|------|
| 预训练模型总目录 | `pretrained_models/` |
| T5-small | `pretrained_models/t5-small/` |
| Sentence-Transformers MPNet | `pretrained_models/sentence-transformers_all-mpnet-base-v2/` |
| METEOR 缓存 | `pretrained_models/evaluate_meteor/` |
| NLTK 数据（Step 3 脚本会设） | `pretrained_models/nltk_data/`（`NLTK_DATA`） |
| BERTScore（HF） | `microsoft/deberta-xlarge-mnli`，需配合 `HF_HOME` 等离线配置 |

---

## 5. Checkpoint：`checkpoints/<task>/…`

- **根目录**：`<D4C_ROOT>/checkpoints/`。
- **单任务目录**由 `get_checkpoint_task_dir(task_idx)` 决定，`task` 为 `1`–`8`。
- **环境变量（checkpoint）**：仅决定 **`get_checkpoint_task_dir`**：
  - **`D4C_CHECKPOINT_GROUP`**：可选中间层（如 `step3`、`step5`）。
  - **`D4C_CHECKPOINT_SUBDIR`**：子目录名（实验/时间戳等）。

日志目录另见第 6.1 节：**`get_log_task_dir`** 可仅由 **`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR` / `D4C_LOG_STEP`** 决定，与上述 checkpoint 变量**解耦**；未使用 LOG_* 时才沿用 **`D4C_CHECKPOINT_*`**（且 GROUP+SUBDIR 双设时与 checkpoint **不完全对称**）。

**解析规则**（与 `paths_config.py` 一致；列顺序同 [`脚本参数说明 copy 3.md`](sh/脚本参数说明%20copy%203.md)）：

| `D4C_CHECKPOINT_GROUP` | `D4C_CHECKPOINT_SUBDIR` | 结果路径 |
|--------------------------|-------------------------|----------|
| 空 | 空 | `checkpoints/<task>/` |
| 空 | 非空 | `checkpoints/<task>/<SUBDIR>/` |
| 非空 | 空 | `checkpoints/<task>/<GROUP>/` |
| 非空 | 非空 | `checkpoints/<task>/<GROUP>/<SUBDIR>/` |

**典型文件**

- **`model.pth`**：Step 3 / Step 5 训练或 `--eval-only` 加载（路径亦可由 `run-d4c.py --save-file` 单独指定）。
- **`factuals_counterfactuals.csv`**：Step 4 生成，与 `generate_counterfactual.py` 使用的上述单任务目录一致。

**各脚本默认值（未手动 `export` 时）**

- **`run_step3_optimized.sh`**（Step 3 正式入口）：默认 **`D4C_CHECKPOINT_GROUP=step3_optimized`**、**`SUBDIR=step3_opt_<时间戳>`** → **`checkpoints/<task>/step3_optimized/step3_opt_<时间>/model.pth`**；主日志根默认 **`log/<task>/step3_optimized/`**。训练侧 env 默认值与 **`config.py`** / **`AdvTrain`** 对齐；按任务 **`D4C_TRAIN_PRESET`** 且未传 **`--batch-size`** 时可能不注入全局 batch（见脚本说明）。
- **`run_step4_optimized.sh`**：**必填 `--step3-subdir`**；**`D4C_CHECKPOINT_GROUP=step3_optimized`**、**`SUBDIR=<NAME>`**；主日志默认 **`log/<task>/step4_optimized/`**（**`D4C_LOG_STEP=step4_optimized`**，在未单独设 **`D4C_LOG_GROUP` / `D4C_LOG_SUBDIR`** 时）。
- **`run_step5_optimized.sh`**：**仅嵌套**：**`--task`** + **`--step3-subdir`**；权重 **`checkpoints/<task>/step3_optimized/<step3_opt_id>/step5/step5_opt_<时间>/model.pth`**（**`GROUP=step3_optimized`**，**`SUBDIR=<step3_opt_id>/step5/<内层>`**）；主日志默认 **`log/<task>/step5_optimized/`**。**`run_step3_to_step5_*.sh`** / **`run_step5_all.sh`** 自动选最新 **`step3_opt_*`**（及 **`--eval-only`** 时最新 **`step5_opt_*`**）。
- **仅自定义子目录**：只设 `D4C_CHECKPOINT_SUBDIR=my_exp`（不设 `GROUP`）→ `checkpoints/<task>/my_exp/…`

---

## 6. 日志：`log/` 与 `logs/` 不要混

### 6.1 `log/`（无 s）：分层主日志与扁平汇总两类

**A. 分层主日志目录**（`get_log_task_dir(task_idx)`，根为 **`log/<task>/`**）

**优先级**（与 `paths_config.get_log_task_dir` 一致）：① **`D4C_LOG_SUBDIR` / `D4C_LOG_GROUP`**（任一非空则仅用二者，语义与 checkpoint 两变量对称；**双非空**时目录为 **`log/<task>/<D4C_LOG_GROUP>/`**）→ ② 否则 **`D4C_LOG_STEP`** 非空 → **`log/<task>/<STEP>/`** → ③ 否则按 **`D4C_CHECKPOINT_*`** 下表（**注意最后一行**：checkpoint 上 GROUP、SUBDIR **均非空** 时，主日志固定在 **`log/<task>/<GROUP>/`**，**不再**按 SUBDIR 分层；eval 在同级 **`eval/`**，如 **`eval_runs.txt`** / **`eval_runs.csv`**）。

| `D4C_CHECKPOINT_GROUP` | `D4C_CHECKPOINT_SUBDIR` | `log/<task>/…`（`runs/…/train.log` 所在根目录） |
|--------------------------|-------------------------|----------------------------------------|
| 空 | 空 | `log/<task>/` |
| 空 | 非空 | `log/<task>/<SUBDIR>/` |
| 非空 | 空 | `log/<task>/<GROUP>/` |
| 非空 | 非空 | **`log/<task>/<GROUP>/`**（不按 SUBDIR 再分子目录） |

**主日志相对路径**（`run_step3_optimized.sh` / `run_step4_optimized.sh` / `run_step5_optimized.sh` 与 `paths_config` 配合）

- 默认：**`runs/<YYYYMMDD_HHMMSS>/train.log`**（秒级时间戳，避免覆盖）。
- 若 **`export D4C_LOG_USE_TIMESTAMP=0`**：固定为 **`runs/run/train.log`**（会覆盖同名文件）。

**写入方式**

- Step 3 / Step 5 将上述路径传给 **`AdvTrain.py` / `run-d4c.py` 的 `--log_file`**，由 Python **`FileHandler`** 写入；Step 4 传给 **`generate_counterfactual.py --log_file`**（**`PerfMonitor`** / **`append_log_dual`**）。**`run_step3_optimized.sh --all` 前台**：**`tee`** 默认写入 **`log/step3_optimized_all_<秒级时间戳>.log`**（**`D4C_STEP3_ALL_SHELL_LOG`** 可覆盖），与各任务 **`train.log`** 分离；其它场景仍**不要**对 `train.log` 再 **`tee`**。
- Step 1+2、串联脚本仍多用 **`tee`** 写入 **`log/` 根下扁平文件**（见下）。

**B. `log/` 根目录下的扁平文件**（**不**经过 `get_log_task_dir`）

- Step 1+2、串联脚本：如 **`step1_step2_*.log`**、**`step3_to_5_all_*.log`**、**`step3_to_5_taskN_*.log`** 等。
- **`run_step5_all.sh`**：**`log/step5_all_*.log`**（各任务输出 **`tee`** 汇总，前台与 **`--daemon`** 均有）。
- Step 3 / Step 4 在 **`--all` 且 `--daemon`** 时：终端汇总为 **`step3_optimized_daemon_*.log`** / **`step4_optimized_daemon_*.log`**；Step 3 **`--eval-only`** 时为 **`step3_optimized_eval_daemon_*.log`**。**`run_step5_optimized.sh`** 无 **`--all`**，**`--daemon`** 时无全任务汇总文件名。

**C. 镜像**

- 环境变量 **`D4C_MIRROR_LOG=1`** 时，`append_log_dual` 可将结构化日志**额外**写入 **`code/log.out`**；主排查仍以脚本传入的 **`--log_file`**（`**runs/…/train.log**` 或 **`runs/run/train.log`**）为准；与 `--log_file` 解析为**同一路径**时只写一次。默认关闭，避免单文件无限增长。

### 6.2 `logs/`（带 s）与 `D4C_LOG_DIR`（Python `train_logging` 等）

- 未显式传入有效 **`--log_file`** 时，部分模块默认可能写到 **`logs/`** 或 **`D4C_LOG_DIR`**（默认 `<D4C_ROOT>/logs`）。
- 与 **`log/`**（无 s）是不同目录；使用 `sh/run_step3_optimized.sh`、`run_step4_optimized.sh`、`run_step5_optimized.sh` 时应以脚本指定的 **`get_log_task_dir` + `runs/…/train.log`** 为主排查。

---

## 7. 环境变量速查

| 变量 | 作用 |
|------|------|
| `D4C_ROOT` | 项目根目录 |
| `D4C_CHECKPOINT_GROUP` | checkpoint 中间层（仅权重路径；**仅**在未使用 `D4C_LOG_*` 时参与 `get_log_task_dir`） |
| `D4C_CHECKPOINT_SUBDIR` | checkpoint 子目录（同上） |
| `D4C_LOG_GROUP` / `D4C_LOG_SUBDIR` | **仅日志**：任一非空则 `get_log_task_dir` 只按二者解析（双非空时目录为 `log/<task>/<LOG_GROUP>/`） |
| `D4C_LOG_STEP` | **仅日志**：非空时为 `log/<task>/<STEP>/`；`run_step4_optimized.sh` 默认 `step4_optimized`（变量未设置且未设 LOG_GROUP/SUBDIR 时）；**空字符串**表示改回随 `D4C_CHECKPOINT_*` |
| `D4C_LOG_USE_TIMESTAMP` | Step 3/4/5 主日志：非 `0` 时用 `runs/<时间戳>/train.log`；设为 `0` 时为 `runs/run/train.log` |
| `D4C_LOG_DIR` | Python 侧默认日志目录（默认 `<D4C_ROOT>/logs`） |
| `D4C_MIRROR_LOG` | 设为 `1` / `true` 等时是否额外镜像到 `code/log.out`（见 6.1-C） |
| `NLTK_DATA` | Step 3/5 脚本设为 `<D4C_ROOT>/pretrained_models/nltk_data` |
| `HF_EVALUATE_OFFLINE` | `run_step5_optimized.sh` 默认 `export HF_EVALUATE_OFFLINE=1`（若环境已有值则保留） |
| `D4C_FULL_EVAL_EVERY` | 设置时为固定间隔 N epoch 做 full valid BLEU；**未设置**固定间隔时由 `config.resolve_full_bleu_eval_training` 使用分阶段默认（epoch≤10 每 5 轮、之后每 2 轮；可调 `D4C_FULL_EVAL_EARLY_EVERY` / `D4C_FULL_EVAL_PHASE_END_EPOCH` / `D4C_FULL_EVAL_LATE_EVERY`）；若已设 **`D4C_TRAIN_PRESET`** 且预设含 **`full_eval_every_epochs`**，则优先于分阶段默认（仍低于本变量与 CLI） |
| `D4C_TRAIN_PRESET` | 命名训练预设（**`code/config.py`** 中 **`TRAINING_PRESETS`** 的键）。可为**全局一条**（顶层即字段 dict，所有任务相同）或**按任务 1–8**（顶层键为整数 1..8，值为各任务字段 dict）。影响 **`get_epochs` / `get_train_batch_size` / `get_min_lr_ratio` / `resolve_full_bleu_eval_training`** 及 Step 3 的 **`adv`**；**`D4C_PRESET_TASK_ID`**（**`run_step3_optimized.sh`** / **`run_step5_optimized.sh`** 等按任务设置）用于解析按任务预设。优先级低于对应 CLI 或 **`D4C_MIN_LR_RATIO` / `D4C_FULL_EVAL_EVERY`** 等。**`run_step3_optimized.sh`** 未设置时默认 **`step3`**，**`run_step5_optimized.sh`** 默认 **`step5`**（**`presets/training/step3.yaml`** / **`step5.yaml`**，二者超参一致；可按任务单独改 YAML 或内置表） |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | Slurm 模板中常见，离线推理 |

---

## 8. 任务编号与域对（1–8）

与 `run_step3_optimized.sh` 中由 **`get_task_params` → `TASK_DEFAULTS`** 得到的域对一致（**`run_step5_optimized.sh`** 仍为脚本内 `eta` 表，二者末列语义不同），便于对照 `Merged_data/<task>/` 与日志目录：

| Task | auxiliary → target |
|------|----------------------|
| 1 | AM_Electronics → AM_CDs |
| 2 | AM_Movies → AM_CDs |
| 3 | AM_CDs → AM_Electronics |
| 4 | AM_Movies → AM_Electronics |
| 5 | AM_CDs → AM_Movies |
| 6 | AM_Electronics → AM_Movies |
| 7 | Yelp → TripAdvisor |
| 8 | TripAdvisor → Yelp |

---

## 9. 最小目录树（逻辑视图）

```
<D4C_ROOT>/
├── code/                 # Python 入口、paths_config
├── sh/                   # bash 入口
├── data/                 # 按域
├── Merged_data/          # 按 task 1–8
├── pretrained_models/
├── checkpoints/          # 按 task，可选 group/subdir
├── log/                  # 按 task；get_log_task_dir 与 checkpoint 在「双设 GROUP+SUBDIR」时不完全对称
├── logs/                 # train_logging 默认（与 log/ 区分）
└── tmux-slurm/           # 集群模板
```

---

*与 [`sh/用法速查 copy 3.md`](sh/用法速查%20copy%203.md)、[`sh/脚本参数说明 copy 3.md`](sh/脚本参数说明%20copy%203.md) 路径章节同步。生成依据：`code/paths_config.py`、`code/train_logging.py`（eval 全局目录），`sh/run_step1_step2.sh`、`sh/smoke_test_ddp.sh`、`sh/run_step3_optimized.sh`、`sh/run_step4_optimized.sh`、`sh/run_step5_optimized.sh`、`sh/run_step5_all.sh`、`sh/run_step3_to_step5_single.sh`、`sh/run_step3_to_step5_all.sh`，`tmux-slurm/submit_d4c.sh`。*

*GPU 主链路：`AdvTrain.py` train/eval、`generate_counterfactual.py`、`run-d4c.py` 均为 **`torchrun` DDP**（含 `nproc_per_node=1`）；`compute_embeddings.py` 为单进程单卡。自检：`sh/smoke_test_ddp.sh`（`checkpoints/1/smoke_ddp/`）。*
