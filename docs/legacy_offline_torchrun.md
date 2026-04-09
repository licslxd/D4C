# 历史：直连 `torchrun` 与 `legacy/code/run_all.sh`（不再推荐）

> **本页仅供考古与手写排障。** 日常请用项目根 **`python code/d4c.py …`**、**`scripts/entrypoints/train_ddp.sh`**、**`scripts/entrypoints/step{3,4,5}.sh`**。主叙事见 **`docs/D4C_Scripts_and_Runtime_Guide.md`**。

下列命令**不**经过 `d4c.py` 对 **`D4C_STAGE_RUN_DIR` / `D4C_HF_CACHE_ROOT` 等**的注入；除 Step1–2 预处理外，**手工 torchrun 极易与 `runs/` 布局不一致**，仅作文本对照。

当前 `d4c_core.runners` 在子进程内加载的入口为 **`code/executors/step3_entry.py`**、**`step4_entry.py`**、**`step5_entry.py`**（不再是已删除的历史文件名）。

---

## `legacy/code/run_all.sh`（早期顺序演示）

在项目根执行（会先 `cd` 到 `code/`）：

```bash
cd D4C-main
bash legacy/code/run_all.sh
```

脚本内 `torchrun` 目标已为 `executors/step*_entry.py`，**仍不按新版 runs 环境变量约定**；**不保证可跑通**。

---

## 分步：在 `code/` 目录下的预处理与嵌入

```bash
cd D4C-main/code
python preprocess_data.py && python split_data.py && python combine_data.py
python compute_embeddings.py && python infer_domain_semantics.py
```

---

## Step 3：手工 `torchrun`（`code/` 目录，须自行 export `D4C_*`）

```bash
cd D4C-main/code
# 须设置 D4C_STAGE_RUN_DIR、D4C_HF_CACHE_ROOT 等，见 path_layout / runners 源码
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 executors/step3_entry.py train \
  --auxiliary AM_Electronics --target AM_CDs --epochs 50
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 executors/step3_entry.py eval \
  --auxiliary AM_Electronics --target AM_CDs
```

---

## Step 4：反事实生成

```bash
cd D4C-main/code
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 executors/step4_entry.py
```

---

## Step 5：主训练

```bash
cd D4C-main/code
DDP_NPROC=1 torchrun --standalone --nproc_per_node=1 executors/step5_entry.py train \
  --auxiliary AM_Electronics --target AM_CDs --epochs 50
```

---

## 其它历史 Python 入口

- **`legacy/code/train.py`**、**`legacy/code/naive_counterfactual_train.py`** 等：见 **`legacy/README.md`**。
