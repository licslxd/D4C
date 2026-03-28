# D4C 运行期规范（Runtime / Execution / Training）

本文档描述 **环境变量**、**命名预设（preset）**、**Shell 启动层** 与 **Python 解析层** 的分工与优先级，与 `code/config.py`、`sh/*.sh`、`scripts/train_ddp.sh` 对齐。后续迭代以源码为准，本文随结构收敛更新。

---

## 1. 分层职责

| 层级 | 职责 | 禁止/避免 |
|------|------|-----------|
| **Execution（执行 / shell）** | 解析 `DDP_NPROC` / `--ddp-nproc`，调用 `torchrun --nproc_per_node`；设置 `CUDA_VISIBLE_DEVICES`（或等价可见设备）；`cd` 到 `code/`、导出路径类 `D4C_*` | 不在 shell 中维护第二份与 `config.TASK_DEFAULTS` 重复的任务超参表 |
| **Training preset（Python）** | `D4C_TRAIN_PRESET` → `TRAINING_PRESETS`：batch、epoch、lr、adv、coef 等训练语义 | 不声明 GPU 进程数、不声明 `torchrun` 拓扑 |
| **Runtime preset（Python）** | `D4C_RUNTIME_PRESET` → `RUNTIME_PRESETS`：`num_proc`（datasets.map）、DataLoader workers、prefetch、`max_parallel_cpu` 等 CPU/IO 并发 | 不声明 GPU 进程数（无 `world_size` / `nproc_per_node` 字段） |
| **Python 训练入口** | `build_resolved_training_config` 等：base → task 表 → preset → ENV → CLI；分布式规模以 **`WORLD_SIZE` / `LOCAL_RANK`** 为准 | 不读取 `DDP_NPROC`（该变量仅 shell / torchrun 使用） |

---

## 2. `DDP_NPROC` 与 `D4C_NUM_PROC`（易混澄清）

- **`DDP_NPROC`**（及 CLI `--ddp-nproc`）：**仅执行层**。由 `sh/run_step*_optimized.sh` 或 `scripts/train_ddp.sh` 解析，传给 **`torchrun --nproc_per_node`**。Python 侧通过 **`WORLD_SIZE`** 感知进程数，**不应**依赖读取 `DDP_NPROC` 环境变量。
- **`D4C_NUM_PROC`**：**Runtime / CPU 侧** 覆盖链中的一环，用于 **`datasets.map` 的并行进程数**等（见 `config.py` 中 `resolve_num_proc_training` 一类逻辑）。与 **每张 GPU 上起几个训练进程** 无关。

命名上二者都含 “proc”，语义不同：**前者 = DDP 进程数，后者 = CPU 数据并行进程数**。

---

## 3. 推荐主路径（统一入口）

- **统一入口**：在项目根执行 `bash scripts/train_ddp.sh …`（`--help` 见 `scripts/train_lib.sh` 内 usage）。支持 **`--step 3|4|5`** 或 **`--pipeline 3,4,5`**（子集去重后按 3→4→5 执行）。`--pipeline` 含 Step 3 时，Step 4/5 使用 **本轮 Step 3 完成后最新的 `step3_opt_*`**；仅 `4`/`5` 时须预先提供存在的 **`--step3-subdir`**。启动前在日志中打印 task / CUDA / DDP_NPROC / preset / batch / subdir 等摘要。
- **兼容入口**：`sh/run_step3_optimized.sh`、`run_step4_optimized.sh`、`run_step5_optimized.sh` 及 `run_step3_to_step5_*.sh` **行为保持不变**；串联脚本对 Step 3/4/5 **透传同一套** `--ddp-nproc` / `DDP_NPROC`。

---

## 4. 环境变量与优先级（摘要）

- **可见 GPU**：`CUDA_VISIBLE_DEVICES`（执行层）；不设时由调度/默认驱动决定。
- **训练命名预设**：`D4C_TRAIN_PRESET`；解析见 `config.TRAINING_PRESETS` 与 `build_resolved_training_config`。
- **Runtime 命名预设**：`D4C_RUNTIME_PRESET`；解析见 `config.RUNTIME_PRESETS` 与 `get_dataloader_num_workers` / `resolve_num_proc_training` 等。
- **任务级辅助**：`D4C_PRESET_TASK_ID`（按任务解析 per-task 预设时由 shell 设置，与 Step 3/5 脚本一致）。

详细键名与覆盖顺序以 **`code/config.py` 注释与实现** 为准。

---

## 5. Preset 外置 YAML（yaml 优先，Python fallback）

- **目录**（相对仓库根 `D4C_ROOT` / 含 `code/` 的上一级）：
  - **`presets/training/*.yaml`**：每个文件名为一条训练预设的键名（如 `step3.yaml` → `TRAINING_PRESETS["step3"]`、`step5.yaml` → `["step5"]`）；内容为该预设的 dict（与 Python 内嵌结构相同，可为按任务 `1..8` 分条或全局一条）。
  - **`presets/runtime/*.yaml`**：每个文件名为 runtime 预设键名；内容为字段 dict（与 `RUNTIME_PRESETS` 中原有键一致）。
  - **`presets/tasks/*.yaml`**：按文件名排序依次合并；**合并后须恰好包含任务 1..8**，字段为 `auxiliary` / `target` / `lr` / `coef` / `adv`（与 `task_configs` / `TASK_DEFAULTS` 一致）。
- **加载策略**：`import config` 时 **优先** 读取上述 YAML；若对应目录不存在、无 `.yaml`/`.yml`、**未安装 PyYAML**、任一字解析失败或 tasks 合并后键集不是 1..8，则 **`warnings.warn` 说明原因后整表回退** 到 `config.py` 内 **内置** `_TASK_CONFIGS_BUILTIN` / `_TRAINING_PRESETS_BUILTIN` / `_RUNTIME_PRESETS_BUILTIN`（行为与改前一致）。训练单文件失败时 **整表**回退内置 TRAINING_PRESETS（保守、可预期）。
- **依赖**：离线依赖列表已含 **`pyyaml`**（`requirements_offline.txt`）；未安装时仅告警并走内置表，不中断 import。
- **接口不变**：`TRAINING_PRESETS`、`RUNTIME_PRESETS`、`task_configs`、`TASK_DEFAULTS`、`build_resolved_training_config` 及 shell 入口 **无需改动调用方式**。

### 5.1 何时触发 fallback（与 `import config` 行为一致）

- **`presets/<子目录>` 不存在**，或目录下 **无任何** `.yaml` / `.yml`：该类别 **不读盘**，直接使用对应 **内置表**（无告警）。
- **未安装 PyYAML**：对该类别 **`warnings.warn`** 后使用内置表。
- **任一字典解析失败**、**根类型非 dict**、**tasks 合并后键集不是恰好 1..8**、或与内置校验器不兼容（未知字段、类型错误、`num_proc < 1` 等）：**`warnings.warn` 后整表回退**（training 单文件失败则 **整表** `TRAINING_PRESETS` 回退内置）。

### 5.2 YAML 校验命令（第四阶段护栏）

在**仓库根**执行（需已安装 PyYAML，与训练环境一致）：

```bash
python3 scripts/check_presets.py
```

可选：`python3 scripts/check_presets.py --repo-root .`（在仓库根目录打开终端时，`.` 即本仓库根；也可传入任意克隆路径的绝对路径）

- 从磁盘 **独立读取** `presets/**/*.yaml`，不依赖当前进程是否已成功用 YAML 覆盖 `config`。
- 规则与 **`config._validate_training_presets` / `_validate_runtime_presets`** 及 **`_normalize_task_row_yaml`** 对齐（脚本内 `import config` 复用实现，避免双份逻辑漂移）。
- **exit 0**：当前扫描到的 yaml 均通过（某目录无文件则跳过该段）。
- **exit 非 0**：解析失败、非法预设文件名、结构/类型/任务键不完整等；错误信息前缀 **`[check_presets]`**。

**Pre-commit**：若启用 `.githooks/pre-commit`，当暂存 **`presets/**.yaml|yml`** 或 **`scripts/check_presets.py`** 时会自动运行上述命令；失败则阻止提交（可用 `git commit --no-verify` 跳过）。

### 5.3 如何新增 preset（规范）

1. **Training**：在 `presets/training/` 新增 **`<预设名>.yaml`**（文件名 = `D4C_TRAIN_PRESET` 使用的键名；仅允许字母、数字、`_`、`-`）。根映射为 **全局一条** 或 **键 `1`..`8` 的 per-task 子 dict**；字段名仅允许 `config.py` 中 `_TRAINING_PRESET_ALLOWED_KEYS` 所列，整型/浮点规则与内置校验一致。
2. **Runtime**：在 `presets/runtime/` 新增 **`<预设名>.yaml`**，根为 dict，**仅允许** `_RUNTIME_PRESET_ALLOWED_KEYS` 中字段；值为整数（`num_proc >= 1`，其余非负等规则同 `config`）。
3. **Tasks**：在 `presets/tasks/` 下增加或修改 `.yaml`；**按文件名排序合并**后须 **恰好覆盖任务 1..8**，每条含 **`auxiliary` / `target` / `lr` / `coef` / `adv`**，类型与 `TaskConfig` 一致。
4. 提交前执行 **`python3 scripts/check_presets.py`**（或依赖 pre-commit）。

---

## 6. `--gpus`（`scripts/train_ddp.sh`）

- 语义：**仅**设置 **`CUDA_VISIBLE_DEVICES`**，例如 `--gpus 0,1`。
- 不写入 training/runtime preset；**不**替代 `--ddp-nproc`（进程数仍须与可见设备数及作业意图一致，由用户保证）。

---

## 7. 与文档、钩子

- 修改 `sh/*.sh` 或 `code/paths_config.py` 时，若仓库启用 `.githooks/pre-commit`，需按钩子要求同步 `sh/` 下用法说明与路径备忘类 Markdown（以钩子列表为准）。
