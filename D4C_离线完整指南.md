# D4C 离线复现完整指南

本文档整合了原 `QUICK_START_OFFLINE.md`、`README.md`（数据与引用）与 `REPRODUCE_OFFLINE.md` 的全部内容，提供**纯离线环境**下完整复现步骤；不依赖联网时请先完成「有网预下载」小节。

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

### 3. 运行（在 `code/` 下执行）

```bash
cd D4C-main/code
chmod +x run_all.sh
./run_all.sh
```

或分步执行：

```bash
cd D4C-main/code
python preprocess_data.py && python split_data.py && python combine_data.py
python compute_embeddings.py && python infer_domain_semantics.py
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 AdvTrain.py train --auxiliary AM_Electronics --target AM_CDs --epochs 50
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 AdvTrain.py eval --auxiliary AM_Electronics --target AM_CDs
python generate_counterfactual.py
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 run-d4c.py --auxiliary AM_Electronics --target AM_CDs --epochs 50
```

**常用命令速查：**

```bash
# 反事实生成（单卡）
python generate_counterfactual.py

# 主训练（须 torchrun / DDP）
torchrun --standalone --nproc_per_node=1 run-d4c.py --auxiliary <AUX> --target <TGT>
```

---

## 一、项目概览

- **核心任务**：跨域推荐解释生成（Domain-aware Counterfactual）
- **框架**：PyTorch + Hugging Face Transformers
- **主入口**：`code/AdvTrain.py`（域对抗：`torchrun … train` + `eval`）→ `code/generate_counterfactual.py`（反事实数据）→ `code/run-d4c.py`（主训练：`torchrun …`，DDP）
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
cd /path/to/D4C-main
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
cd /path/to/D4C-main
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
#     export HF_HOME=/path/to/D4C-main/pretrained_models/hf_cache
#     export HF_HUB_OFFLINE=1
#     export TRANSFORMERS_OFFLINE=1
```

**离线运行时环境变量（建议在运行前设置）：**

```bash
export HF_HOME=/path/to/D4C-main/pretrained_models/hf_cache   # BERTScore 模型所在目录
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

Step 3 **仅支持 DDP**：训练与评估均为 `torchrun … AdvTrain.py train|eval …`（eval 在 valid 上分片推理，rank0 聚合指标）。建议使用 `sh/run_step3.sh` 自动依次执行；或手动执行（在 `code/` 目录）：

```bash
# 单卡等价：nproc=1
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 AdvTrain.py train \
  --auxiliary AM_Electronics --target AM_CDs --epochs 50
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 AdvTrain.py eval \
  --auxiliary AM_Electronics --target AM_CDs

# 多卡（如 2 张 GPU；全局 batch 须能被进程数整除，见 config.train_batch_size）
CUDA_VISIBLE_DEVICES=0,1 DDP_NPROC=2 torchrun --standalone --nproc_per_node=2 AdvTrain.py train \
  --auxiliary AM_Movies --target AM_CDs --epochs 50
DDP_NPROC=2 torchrun --standalone --nproc_per_node=2 AdvTrain.py eval \
  --auxiliary AM_Movies --target AM_CDs
```

其余 6 组 (auxiliary, target) 按 `config.py` 中任务表逐对运行同上命令。库用法：`from AdvTrain import *`（与 `generate_counterfactual` 等兼容）。

### Step 4：生成反事实数据

```bash
# 单卡
python generate_counterfactual.py

# 多卡
python generate_counterfactual.py --gpus 0,1
```

### Step 5：主训练与评估（仅 DDP，须 `torchrun`）

```bash
# 单卡（1 进程）
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 run-d4c.py \
  --auxiliary AM_Electronics --target AM_CDs --epochs 50

# 多卡（与可见 GPU 数一致；全局 batch 须能被进程数整除）
DDP_NPROC=2 torchrun --standalone --nproc_per_node=2 run-d4c.py \
  --auxiliary AM_Electronics --target AM_CDs --epochs 50 --batch-size 1024
# 或（在项目根目录，须指定 Step 3 子目录名）：bash sh/run_step5.sh --task 1 --step3-subdir step3_YYYYMMDD_HHMM
# 或（各任务 Step 3/4 已就绪，自动选每任务最新 step3_*）：bash sh/run_step5_all.sh
```

---

## 五、一键批量运行脚本

可直接使用仓库中的 `code/run_all.sh`（Step 3 默认 `DDP_NPROC=1`；多卡可 `DDP_NPROC=2 bash run_all.sh`）。

**便捷脚本（项目根目录 `sh/`，需先 `cd` 到 `D4C-main` 再执行，或写绝对路径）：**

| 脚本 | 用途 |
|------|------|
| `run_step1_step2.sh` | Step 1+2 合并，支持 `--embed-batch-size N`、`--gpus 0,1` |
| `run_step3.sh` | Step 3（DDP）：train 与 eval 均 `torchrun`；`DDP_NPROC` 或 `--ddp-nproc K`；`--gpus` 仅在你手动单进程跑 eval 时有意义，脚本内已忽略 |
| `run_step4.sh` | Step 4，`--all` / `--task N`，支持 `--gpus 0,1` |
| `run_step5.sh` | Step 5（DDP）**仅嵌套**：`--task N` + `--step3-subdir`；`DDP_NPROC` 或 `--ddp-nproc K`（无 `--all`） |
| `run_step5_all.sh` | Step 5 **批量任务 1–8**：仅调 `run_step5.sh`，每任务自动最新 `step3_*`；`--eval-only` 时再自动最新 `step5_*`；汇总 `log/step5_all_*.log` |
| `run_step3_to_step5_all.sh` | Step 3-5 全部任务，支持 `--from N` 续跑、`--gpus`（Step4 等）、`--ddp-nproc K`（Step3 train+eval + Step5） |
| `run_step3_to_step5_single.sh` | Step 3-5 单个任务，`--task N`，同上 |

**集群 / 长任务（`tmux-slurm/`）**：Slurm 提交、tmux 与 nohup 包装脚本及说明见 [`tmux-slurm/README.md`](tmux-slurm/README.md)（须在项目根目录执行 `sbatch` / 调用脚本）。

---

## 六、路径与配置说明

- **项目根目录**：默认 `D4C-main/`，可通过环境变量 `D4C_ROOT` 覆盖
- **数据目录**：`{D4C_ROOT}/data/`
- **Merged 数据**：`{D4C_ROOT}/Merged_data/1` ~ `8`
- **模型权重**：`checkpoints/{task_idx}/model.pth`（项目根目录下）
- **反事实数据**：`checkpoints/{task_idx}/factuals_counterfactuals.csv`

若此前权重保存在旧路径 `code/checkpoints/`，请整目录迁移到项目根 `checkpoints/`，或设置环境变量 `D4C_ROOT` 后保持相对关系一致。

---

## 七、batch_size 相关说明

项目中涉及 `batch_size` 的位置：

| 脚本/阶段 | 参数/位置 | 默认值 | 说明 |
|-----------|-----------|--------|------|
| `compute_embeddings.py` | `EMBED_BATCH_SIZE` 环境变量、`--gpus` | 256 | 计算 user/item 嵌入时的批次大小，显存不足可减小（如 64、128），多卡可增大（如 1024） |
| `infer_domain_semantics.py` | 无 | 1 | 按 chunk 逐个处理，基本不占显存 |
| `AdvTrain.py train` / `eval`（`torchrun`） | 全局 `--batch-size` / `config.train_batch_size` | 见 config | DDP 下每 rank 的 DataLoader batch = 全局 / 进程数 |
| `run-d4c.py`（`torchrun`） | 全局 `--batch-size` / `config.train_batch_size` | 见 config | DDP 下每卡 batch = 全局 / 进程数 |
| `generate_counterfactual.py` | 硬编码 | 128 | 反事实生成时的 DataLoader 批次大小 |

**调整方式：**

- 嵌入阶段：`EMBED_BATCH_SIZE=64 python compute_embeddings.py` 或 `run_preprocess_and_embed.py --embed-batch-size 64`；多卡大 batch：`run_preprocess_and_embed.py --embed-batch-size 1024 --gpus 0,1`
- 训练阶段：在对应脚本的 `config` 字典中修改 `"batch_size": 128` 为更小值（如 32、64）

---

## 七.1、多 GPU 使用说明

`compute_embeddings.py`、`generate_counterfactual.py` 支持 `--gpus`（DataParallel 等）。`AdvTrain.py eval` 在 **`python` 单进程** 下可用 `--gpus` 做 DataParallel；**`torchrun` 下为 DDP 分片评估**（与 Step 3 训练同一套进程数）。**Step 3 与 Step 5 训练**均为 **DDP**（`torchrun` + 进程数与 `CUDA_VISIBLE_DEVICES` 对齐）；`run-d4c.py` 不可再用 `python` 单进程直接启动。

```bash
# Step 1+2 嵌入阶段多卡（在项目根 D4C-main 下执行；可配合大 batch 如 1024）
bash sh/run_step1_step2.sh --embed-batch-size 1024 --gpus 0,1

# Step 3 / Step 5：在 code/ 下执行；DDP 训练 + DDP 评估（与训练相同 nproc）
cd code
DDP_NPROC=2 torchrun --standalone --nproc_per_node=2 AdvTrain.py train \
  --auxiliary AM_Electronics --target AM_CDs --epochs 50
DDP_NPROC=2 torchrun --standalone --nproc_per_node=2 AdvTrain.py eval \
  --auxiliary AM_Electronics --target AM_CDs
DDP_NPROC=2 torchrun --standalone --nproc_per_node=2 run-d4c.py \
  --auxiliary AM_Electronics --target AM_CDs --epochs 50 --batch-size 1024
python generate_counterfactual.py --gpus 0,1
# 仍位于 code/ 目录
```

- 不传 `--gpus` 时，沿用单卡运行（评估与其它步骤）
- `generate_counterfactual` 的生成阶段为自回归，多卡加速有限，主要适用于训练阶段

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
├── checkpoints/          # 模型与反事实 CSV（paths_config.CHECKPOINT_DIR）
├── sh/                   # run_step3.sh、run_step4.sh 等便捷脚本
├── tmux-slurm/           # Slurm / tmux / nohup 辅助脚本
├── code/
│   ├── run_all.sh
│   ├── paths_config.py
│   └── ...
├── requirements_offline.txt
└── D4C_离线完整指南.md   # 本文件（合并版离线指南）
```

---

## 十、最小依赖列表（requirements_offline.txt）

见项目中的 `requirements_offline.txt`。
