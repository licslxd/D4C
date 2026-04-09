# D4C 离线复现完整指南

本文档整合了原 `QUICK_START_OFFLINE.md`、`README.md`（数据与引用）与 `REPRODUCE_OFFLINE.md` 的全部内容，提供**纯离线环境**下完整复现步骤；不依赖联网时请先完成「有网预下载」小节。

**日常上手**：请优先阅读仓库根目录 **`README.md`**（主入口与 manifest）；**配置合并**见 **`docs/PRESETS.md`**；**脚本与路径规范**见 **`docs/D4C_Scripts_and_Runtime_Guide.md`**。本文侧重离线数据、依赖与集群补充，避免与上述文档重复堆叠。

---

## 目录

- [数据与外部资源](#数据与外部资源)
- [极简快速开始](#极简快速开始)
- [一、项目概览](#一项目概览)
- [二、需手动下载的文件与路径](#二需手动下载的文件与路径)
- [三、离线依赖安装](#三离线依赖安装)
- [四、完整运行流程](#四完整运行流程)
- [五、一键批量运行脚本](#五一键批量运行脚本)
- [六、路径与配置说明](#六路径与配置说明)
- [七、batch_size 相关说明](#七batch_size-相关说明)
- [七.1、多 GPU 使用说明](#七1多-gpu-使用说明)
- [八、常见问题](#八常见问题)
- [九、目录结构](#九目录结构)
- [十、最小依赖列表](#十最小依赖列表)
- [历史 torchrun / legacy（考古）](docs/legacy_offline_torchrun.md) · [Legacy 批量 Shell](docs/legacy_batch_shell.md)

---

## 数据与外部资源

### 数据集（Google Drive）

可下载 AM_CDs、AM_Electronics、AM_Movies、TripAdvisor、Yelp：

[Google Drive](https://drive.google.com/drive/folders/1gQz_eIvlNaIXkUM9w77i0k3zEgvWAo6Y?usp=sharing)

### 自建 / 其它数据处理

可参考 Lei Li 提供的流程处理自有数据：

- [NETE](https://github.com/lileipisces/NETE)
- [Sentires-Guide](https://github.com/lileipisces/Sentires-Guide)

### 代码引用

- [PETER](https://github.com/lileipisces/PETER)
- [AdaRex](https://github.com/YuYi0/AdaRex)

---

## 极简快速开始

### 1. 有网环境预下载（拷贝到离线机前完成）

```bash
cd D4C-main
mkdir -p pretrained_models data

# 下载 T5 与 MPNet
python -c "
from transformers import T5Tokenizer, AutoTokenizer, AutoModel
T5Tokenizer.from_pretrained('t5-small', legacy=True).save_pretrained('pretrained_models/t5-small')
AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2').save_pretrained('pretrained_models/sentence-transformers_all-mpnet-base-v2')
AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').save_pretrained('pretrained_models/sentence-transformers_all-mpnet-base-v2')
"

# 下载数据集（从 Google Drive）到 data/AM_CDs/, data/AM_Electronics/ 等
# 每个目录需含 reviews.pickle
```

### 2. 离线依赖安装

```bash
# 有网机下载 whl
pip download -r requirements_offline.txt -d offline_wheels

# 离线机安装
pip install --no-index --find-links=./offline_wheels -r requirements_offline.txt
```

### 3. 运行

**推荐（项目根 `D4C-main/`）**：Python 实验 **`python code/d4c.py …`** 为 **MAINLINE ENTRY**（见 `docs/D4C_Scripts_and_Runtime_Guide.md`）；子进程由 **`d4c_core.runners`** `torchrun` **`code/executors/step{3,4,5}_entry.py`**。单阶段 Shell 用 **`scripts/entrypoints/step*.sh`**；**官方 Bash 批量**用 **`scripts/entrypoints/train_ddp.sh`**。排障：`export D4C_DISPATCH_DETAIL=1`。**默认**在当次 run 目录写 **`manifest.json`**（`manifest_schema_version` 2.0 起为纯结构化字段；`D4C_WRITE_RUN_MANIFEST=0` 可关闭）。

**历史 / 演示（不再推荐）**：直连 `torchrun` 与 `legacy/run_all.sh` 的命令已迁至 **`docs/legacy_offline_torchrun.md`**，避免与主线并列误读。

**常用命令速查（主线）：**

```bash
# 项目根：Step3 → Step4 → Step5（参数见 python code/d4c.py -h）
python code/d4c.py step3 --task 1 --preset step3 --iter v1
python code/d4c.py step4 --task 1 --preset step3 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu
python code/d4c.py step5 --task 1 --preset step5 --iter v1 --from-run 2 --step4-run 2_1 --step5-run auto --eval-profile eval_fast_single_gpu

# DDP 链路自检
python code/d4c.py smoke-ddp
# 等价：bash scripts/entrypoints/smoke_ddp.sh
```

**附录（仅考古 / 手写排障）**：见 **`docs/legacy_offline_torchrun.md`**；日常请优先上表 `d4c.py`。

---

## 一、项目概览

- **核心任务**：跨域推荐解释生成（Domain-aware Counterfactual）
- **框架**：PyTorch + Hugging Face Transformers
- **Python 主入口（MAINLINE）**：`python code/d4c.py step3|step4|step5|eval|pipeline|smoke-ddp …`（项目根，详见 `docs/D4C_Scripts_and_Runtime_Guide.md`）
- **阶段 runner**：由 `d4c_core.runners` 经 `torchrun` 调度 **`code/executors/step{3,4,5}_entry.py`**；核心逻辑在 `code/executors/*_engine.py`
- **数据**：AM_CDs、AM_Electronics、AM_Movies、TripAdvisor、Yelp

---

## 二、需手动下载的文件与路径

### 1. 预训练模型

在**有网络的机器**上预先下载，再拷贝到离线服务器。

| 文件/模型 | 存放路径 |
|-----------|----------|
| T5-small | `D4C-main/pretrained_models/t5-small/` |
| all-mpnet-base-v2 | `D4C-main/pretrained_models/sentence-transformers_all-mpnet-base-v2/` |
| METEOR | 见下方「METEOR 与 BERTScore 离线下载」 |
| BERTScore (deberta) | 见下方「METEOR 与 BERTScore 离线下载」 |

**下载命令（有网环境执行）：**

```bash
# 以下命令假定当前目录为本仓库根目录（与 D4C_evaluator.py 同级；可先 cd 到克隆目录）
mkdir -p pretrained_models

# 方式一：huggingface-cli
pip install huggingface-hub
huggingface-cli download t5-small --local-dir pretrained_models/t5-small
huggingface-cli download sentence-transformers/all-mpnet-base-v2 --local-dir pretrained_models/sentence-transformers_all-mpnet-base-v2

# 方式二：Python 脚本
python -c "
from transformers import T5Tokenizer, AutoTokenizer, AutoModel
T5Tokenizer.from_pretrained('t5-small', legacy=True).save_pretrained('pretrained_models/t5-small')
AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2').save_pretrained('pretrained_models/sentence-transformers_all-mpnet-base-v2')
AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').save_pretrained('pretrained_models/sentence-transformers_all-mpnet-base-v2')
"
```

**METEOR 与 BERTScore 离线下载：**

评估阶段需 METEOR 和 BERTScore，二者均需在有网环境预缓存后拷贝到离线机。

```bash
# 以下命令假定当前目录为本仓库根目录
mkdir -p pretrained_models/evaluate_meteor pretrained_models/evaluate_bertscore

# 1. METEOR（evaluate 库的 metric 模块 + NLTK 数据）
# 1a. 预缓存 METEOR metric 到指定目录
python -c "
import evaluate
evaluate.load('meteor', cache_dir='pretrained_models/evaluate_meteor')
"
# 1b. 下载 METEOR 依赖的 NLTK 数据（wordnet 等）
python -c "
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
"
# 将 NLTK 数据一并拷贝：默认在 ~/nltk_data 或 /usr/share/nltk_data
# 离线机需设置 NLTK_DATA 指向该目录，或放在同等路径

# 2. BERTScore 所需模型 microsoft/deberta-xlarge-mnli（约 3GB）
# 2a. 使用 HF 缓存目录，便于离线复用
export HF_HOME="$(pwd)/pretrained_models/hf_cache"
huggingface-cli download microsoft/deberta-xlarge-mnli
# 2b. 将 pretrained_models/hf_cache 整个目录拷贝到离线机，离线运行时设置：
#     export HF_HOME="$(pwd)/pretrained_models/hf_cache"   # 先在仓库根目录 cd 再执行
#     export HF_HUB_OFFLINE=1
#     export TRANSFORMERS_OFFLINE=1
```

**离线运行时环境变量（建议在运行前设置）：**

```bash
export HF_HOME="$(pwd)/pretrained_models/hf_cache"   # 在仓库根目录执行；BERTScore 模型所在目录
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1                                  # evaluate 库走本地缓存
# 若 NLTK 数据未放在默认路径，设置：
# export NLTK_DATA=/path/to/nltk_data
```

**说明：** 若仅做训练与反事实生成，不跑评估，可跳过 METEOR/BERTScore；评估时报错会回退为 0.0，不影响主流程。

### 2. 数据集

| 数据集 | 来源 | 存放路径 |
|--------|------|----------|
| AM_CDs, AM_Electronics, AM_Movies, TripAdvisor, Yelp | [Google Drive](https://drive.google.com/drive/folders/1gQz_eIvlNaIXkUM9w77i0k3zEgvWAo6Y?usp=sharing) | `D4C-main/data/{数据集名}/reviews.pickle` |

**目录结构要求：**

```
D4C-main/data/
├── AM_CDs/
│   ├── reviews.pickle    # 原始数据（必须预先放置）
│   ├── processed.csv     # 预处理后生成
│   ├── train.csv        # 划分后生成
│   ├── valid.csv
│   ├── test.csv
│   ├── user_profiles.npy # compute_embeddings 生成
│   ├── item_profiles.npy
│   └── domain.npy       # infer_domain_semantics 生成
├── AM_Electronics/
├── AM_Movies/
├── TripAdvisor/
└── Yelp/
```

---

## 三、离线依赖安装

### 方法：本地 whl 安装

在有网的机器上先下载所有 whl 到本地目录：

```bash
pip download -r requirements_offline.txt -d ./offline_wheels
```

将 `offline_wheels/` 和 `requirements_offline.txt` 拷贝到离线服务器后：

```bash
pip install --no-index --find-links=./offline_wheels -r requirements_offline.txt
```

**PyTorch（带 CUDA）需单独按版本下载：**

```bash
# 以 torch 1.11.0+cu113 为例
pip download torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -d ./offline_wheels -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

---

## 四、完整运行流程

**所有脚本均在 `code/` 目录下执行：**

```bash
cd D4C-main/code
```

### Step 1：数据预处理

```bash
python preprocess_data.py
python split_data.py
python combine_data.py
```

### Step 2：嵌入与域语义

```bash
python compute_embeddings.py
python infer_domain_semantics.py
```

**Step 1 + Step 2 合并脚本**（一键执行上述 5 个步骤）：

```bash
# 默认嵌入 batch_size=256
python run_preprocess_and_embed.py

# 显存不足时减小嵌入批次大小
python run_preprocess_and_embed.py --embed-batch-size 64
```

### Step 3：域对抗预训练（对每个任务运行一次）

Step 3 **仅支持 DDP**（eval 在 valid 上分片推理，rank0 聚合指标）。**推荐**在项目根使用 **`python code/d4c.py step3 --task N --preset step3 …`** 或 **`bash scripts/entrypoints/step3.sh`**（内部同样调 `d4c.py`）。手工 `torchrun` 薄壳命令见 **`docs/legacy_offline_torchrun.md`**。

### Step 4：生成反事实数据

**推荐**：`python code/d4c.py step4 … --from-run <Step3 子目录名> --eval-profile <stem>`（推理 batch 仅来自该 profile 的 `eval_batch_size`）或 `bash scripts/entrypoints/step4.sh …`。手工 `torchrun` 见 **`docs/legacy_offline_torchrun.md`**。

### Step 5：主训练与评估（仅 DDP）

**推荐**：`python code/d4c.py step5 …` / `eval …`，或 `bash scripts/entrypoints/step5.sh …`；批量用 **`bash scripts/entrypoints/train_ddp.sh`**。手工 `torchrun` 见 **`docs/legacy_offline_torchrun.md`**。

---

## 五、一键批量运行脚本

**官方 Bash 批量入口（推荐）**：项目根 **`bash scripts/entrypoints/train_ddp.sh`**（GPU/DDP 校验后多次调用 **`python code/d4c.py`**）。参数与示例见 **`docs/D4C_Scripts_and_Runtime_Guide.md`** §4.6。

**单阶段 Shell（`scripts/entrypoints/`）**：`step1_step2.sh`、`step3.sh`、`step4.sh`、`step5.sh`、`smoke_ddp.sh` 等 — 细节见主指南 §1 与 §4。

**Legacy 串联脚本**（已迁出主线 **`sh/`**，考古用）：`legacy/sh/run_step3_to_step5_all.sh`、`legacy/sh/run_step3_to_step5_single.sh`、`legacy/sh/run_step5_all.sh` — 见 **`docs/legacy_batch_shell.md`**。

**历史顺序演示**：`legacy/code/run_all.sh`（见 **`docs/legacy_offline_torchrun.md`**）。

**集群 / 长任务**：自行用 Slurm/tmux/nohup 包装 **`python code/d4c.py …`** 或 **`scripts/entrypoints/train_ddp.sh`**；参数须与主指南一致。

---

## 六、路径与配置说明

- **项目根目录**：默认 `D4C-main/`，可通过环境变量 `D4C_ROOT` 覆盖
- **数据目录**：`{D4C_ROOT}/data/`
- **Merged 数据**：`{D4C_ROOT}/Merged_data/1` ~ `8`
- **训练与评测正式产物**：`runs/task{T}/vN/train/step3/<run>/`、`.../train/step5/<run>/`、`.../eval/<run>/` 等（见 `docs/D4C_Scripts_and_Runtime_Guide.md`）
- **Step5 权重文件**：`runs/.../train/step5/<run>/model/model.pth`
- **反事实 CSV**：位于对应 **Step3 run** 目录：`runs/.../train/step3/<run>/factuals_counterfactuals.csv`

---

## 七、batch_size 相关说明

项目中涉及 `batch_size` 的位置：

| 脚本/阶段 | 参数/位置 | 默认值 | 说明 |
|-----------|-----------|--------|------|
| `compute_embeddings.py` | `EMBED_BATCH_SIZE` 环境变量、`--cuda-device` | 256 | **单进程单 GPU** 计算嵌入；显存不足可减小 batch（如 64、128） |
| `infer_domain_semantics.py` | 无 | 1 | 按 chunk 逐个处理，基本不占显存 |
| step3 runner（`d4c step3` / `torchrun`） | 全局 `--batch-size` / `config.train_batch_size` | 见 config | DDP 下每 rank 的 DataLoader batch = 全局 / 进程数 |
| step5 runner（`d4c step5` / `torchrun`） | 全局 `--batch-size` / `config.train_batch_size` | 见 config | DDP 下每卡 batch = 全局 / 进程数 |
| step4 runner（`d4c step4` / `torchrun`） | 父进程传入 `--batch-size` = **`eval_profile.eval_batch_size`**（须 `--eval-profile`） | 见 `presets/eval_profiles/` | **`torchrun` DDP**：全局 eval batch，须能被 `WORLD_SIZE` 整除；**不再**使用 `train_batch_size` |

**调整方式：**

- 嵌入阶段：`EMBED_BATCH_SIZE=64 python compute_embeddings.py` 或 `run_preprocess_and_embed.py --embed-batch-size 64`；指定 GPU：`python compute_embeddings.py --cuda-device 0`
- 训练阶段：在对应脚本的 `config` 字典中修改 `"batch_size": 128` 为更小值（如 32、64）

---

## 七.1、多 GPU 使用说明

**Step 3 / Step 4 / Step 5** 的 GPU 正式链路均为 **`torchrun` + DDP**（含 **`--nproc_per_node=1` 单卡 DDP smoke**）。日常由 **`python code/d4c.py …`** 编排；step3/4/5 **eval/train 子命令**均须在分布式进程组内运行，**不要**在非 `torchrun` 环境下直接执行薄壳 `*.py`。**`compute_embeddings.py`** 为 **单进程单 device**（`--cuda-device`），不设 DataParallel。

```bash
# Step 1+2：嵌入为单卡（可调 batch 与 --cuda-device）
bash scripts/entrypoints/step1_step2.sh --embed-batch-size 1024 --cuda-device 0

# Step 3 / 4 / 5：推荐在项目根使用 d4c.py（内部再 torchrun）
python code/d4c.py step3 --task 1 --preset step3 --iter v1
python code/d4c.py step4 --task 1 --preset step3 --iter v1 --from-run 1 --eval-profile eval_fast_single_gpu
python code/d4c.py step5 --task 1 --preset step5 --iter v1 --from-run 2 --step4-run 2_1 --step5-run auto --eval-profile eval_fast_single_gpu
```

- 单卡自检：`python code/d4c.py smoke-ddp`；手写 `torchrun` 见 **`docs/legacy_offline_torchrun.md`**
- 历史脚本在 **`legacy/code/`**（索引 **`legacy/README.md`**）

---

## 八、常见问题

1. **CUDA 显存不足**：减小 `--batch_size`（默认 128）或 `EMBED_BATCH_SIZE`（嵌入阶段默认 256）
2. **METEOR/BERTScore 仍尝试联网**：按上文「METEOR 与 BERTScore 离线下载」完成预缓存，并设置 `HF_HUB_OFFLINE=1`、`HF_EVALUATE_OFFLINE=1`、`HF_HOME` 指向 BERTScore 模型目录
3. **无 GPU**：使用 `--device -1` 或修改代码中 device 为 `cpu`

---

## 九、目录结构

需提前准备的整体布局示例：

```
D4C-main/
├── pretrained_models/
│   ├── t5-small/
│   └── sentence-transformers_all-mpnet-base-v2/
├── data/
│   ├── AM_CDs/reviews.pickle
│   ├── AM_Electronics/reviews.pickle
│   ├── AM_Movies/reviews.pickle
│   ├── TripAdvisor/reviews.pickle
│   └── Yelp/reviews.pickle
├── runs/                 # 正式训练/评测产物根（task{T}/vN/...）
├── legacy/               # 考古：legacy/code/、legacy/sh/、legacy/tools/
├── scripts/entrypoints/  # 单阶段便捷编排（step3.sh 等）
├── code/
│   ├── d4c.py
│   ├── d4c_core/
│   ├── tools/            # 排障 / 比对小工具
│   ├── paths_config.py
│   └── ...
├── requirements_offline.txt
└── D4C_离线完整指南.md   # 本文件（合并版离线指南）
```

---

## 十、最小依赖列表（requirements_offline.txt）

见项目中的 `requirements_offline.txt`。
