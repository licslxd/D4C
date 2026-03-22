# D4C 最终版路径规范备忘

本文档与仓库内 `code/paths_config.py` 及各 `sh/*.sh` 行为一致，并与 **[`sh/用法速查 copy.md`](sh/用法速查%20copy.md)**、**[`sh/脚本参数说明 copy.md`](sh/脚本参数说明%20copy.md)** 中「`checkpoints/` 与 `log/` 路径」章节**完全对齐**（备忘侧重总览与排障，细节命令仍以该两份为准）。

---

## 1. 项目根目录 `D4C_ROOT`

- **默认**：`code/` 的上一级目录（即包含 `code/`、`data/`、`sh/` 的目录，例如 `D4C-main`）。
- **覆盖**：设置环境变量 `D4C_ROOT` 为绝对路径（集群/容器挂载时常用）。

所有下文路径均相对于 `D4C_ROOT`，除非另行说明。

---

## 2. Shell 脚本约定

- 入口脚本在 **`sh/`**，执行前会 `cd` 到 **`code/`** 再调用 Python。
- 解析根目录方式：`D4C_ROOT="$(cd "$SH_DIR/.." && pwd)"`，其中 `SH_DIR` 为当前 `.sh` 所在目录。
- **Slurm**：应在项目根下提交（见 `tmux-slurm/submit_d4c.sh` 注释），保证 `log/` 与相对路径正确。

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
- **环境变量**（与 `get_log_task_dir` **完全一致**）：
  - **`D4C_CHECKPOINT_GROUP`**：可选中间层（如 `step3`、`step5`）。
  - **`D4C_CHECKPOINT_SUBDIR`**：子目录名（实验/时间戳等）。

**解析规则**（与 `paths_config.py` 一致；列顺序同 [`脚本参数说明 copy.md`](sh/脚本参数说明%20copy.md)）：

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

- **`run_step3.sh`**：设置 `D4C_CHECKPOINT_GROUP=step3`，`D4C_CHECKPOINT_SUBDIR=step3_<时间戳>` → 例如 `checkpoints/<task>/step3/step3_<时间>/model.pth`。
- **`run_step5.sh`**：若**未**预先设置 `D4C_CHECKPOINT_SUBDIR`，则自动 `SUBDIR=step5_<时间戳>`、`GROUP=step5` → 例如 `checkpoints/<task>/step5/step5_<时间>/model.pth`。**`--eval-only`** 时**不会**自动新建子目录：须先 **`export D4C_CHECKPOINT_SUBDIR=…`**（及按需 **`D4C_CHECKPOINT_GROUP`**）指向已有目录，否则脚本退出。
- **仅自定义子目录**：只设 `D4C_CHECKPOINT_SUBDIR=my_exp`（不设 `GROUP`）→ `checkpoints/<task>/my_exp/…`

---

## 6. 日志：`log/` 与 `logs/` 不要混

### 6.1 `log/`（无 s）：分层 `run.log` 与扁平汇总两类

**A. 与 checkpoint 对称的分层目录**（`get_log_task_dir(task_idx)`，规则同 **第 5 节**，根为 **`log/<task>/`**）

- Step 3 / Step 5 在**常规前台**或**单任务** **`--daemon`** 时，脚本将主日志设为 **`log/<task>/…/run.log`**（中间层随当前 `GROUP`/`SUBDIR` 变化），并传入 Python **`--log_file`**，与 **`tee`** 一致。
- 未设置 `GROUP`/`SUBDIR` 时退化为 **`log/<task>/run.log`**。

**B. `log/` 根目录下的扁平文件**（**不**经过 `get_log_task_dir`）

- Step 1+2、Step 4、串联脚本：如 **`step1_step2_*.log`**、**`step4_*.log`**、**`step3_to_5_all_*.log`**、**`step3_to_5_taskN_*.log`** 等。
- Step 3 / Step 5 在 **`--all` 且 `--daemon`** 时：汇总为 **`step3_daemon_*.log`** / **`step5_daemon_*.log`**；若同时为 **`--eval-only`**，则为 **`step3_eval_daemon_*.log`** / **`step5_eval_daemon_*.log`**。

**C. 镜像**

- 环境变量 **`D4C_MIRROR_LOG=1`** 时，`append_log_dual` 可将结构化日志**额外**写入 **`code/log.out`**；主排查仍以脚本传入的 **`--log_file`**（如 `run.log`）为准；与 `--log_file` 解析为**同一路径**时只写一次。默认关闭，避免单文件无限增长。

### 6.2 `logs/`（带 s）与 `D4C_LOG_DIR`（Python `train_logging` 等）

- 未显式传入有效 **`--log_file`** 时，部分模块默认可能写到 **`logs/`** 或 **`D4C_LOG_DIR`**（默认 `<D4C_ROOT>/logs`）。
- 与 **`log/`**（无 s）是不同目录；使用 `sh/run_step3.sh`、`run_step5.sh` 时应以脚本指定的 **`run.log`** 为主排查。

---

## 7. 环境变量速查

| 变量 | 作用 |
|------|------|
| `D4C_ROOT` | 项目根目录 |
| `D4C_CHECKPOINT_GROUP` | checkpoint / `log/<task>/` 的中间层目录名 |
| `D4C_CHECKPOINT_SUBDIR` | checkpoint / 日志子目录名 |
| `D4C_LOG_DIR` | Python 侧默认日志目录（默认 `<D4C_ROOT>/logs`） |
| `D4C_MIRROR_LOG` | 设为 `1` / `true` 等时是否额外镜像到 `code/log.out`（见 6.1-C） |
| `NLTK_DATA` | Step 3 建议设为 `<D4C_ROOT>/pretrained_models/nltk_data` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | Slurm 模板中常见，离线推理 |

---

## 8. 任务编号与域对（1–8）

与 `run_step3.sh` / `run_step5.sh` 中 `get_task_params` 一致，便于对照 `Merged_data/<task>/` 与日志目录：

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
├── log/                  # 与 checkpoints 对称的 run.log 等
├── logs/                 # train_logging 默认（与 log/ 区分）
└── tmux-slurm/           # 集群模板
```

---

*与 [`sh/用法速查 copy.md`](sh/用法速查%20copy.md)、[`sh/脚本参数说明 copy.md`](sh/脚本参数说明%20copy.md) 路径章节同步。生成依据：`code/paths_config.py`，`sh/run_step1_step2.sh`、`run_step3.sh`、`run_step4.sh`、`run_step5.sh`、`run_step3_to_step5_single.sh`、`run_step3_to_step5_all.sh`，`tmux-slurm/submit_d4c.sh`。*
