# D4C

**一句话**：面向跨域推荐场景的 **可反事实解释生成**（PyTorch + Transformers）。  
**主入口（唯一推荐）**：在仓库根目录执行 **`python code/d4c.py …`**。

更完整的运行规范、Shell 参数与路径约定见 **`docs/D4C_Scripts_and_Runtime_Guide.md`**；**配置与预设**见 **`docs/PRESETS.md`**；离线数据与集群补充见根目录 **`D4C_离线完整指南.md`**。

---

## 快速开始（Quick Start）

以下命令均在 **项目根**（含 `code/`、`presets/` 的目录）执行。

```bash
# Step3 — 域对抗（默认 train + 收尾 eval）
python code/d4c.py step3 --task 1 --preset step3

# Step4 — 反事实生成（--from-run 必须与 Step3 的 run 目录名一致）
python code/d4c.py step4 --task 1 --preset step3 --from-run step3_opt_20260329_1200

# Step5 — 主模型训练（--preset 一般为 step5）
python code/d4c.py step5 --task 1 --preset step5 --from-run step3_opt_20260329_1200 --step5-run step5_opt_20260329_1300

# Eval — Step5 权重 valid 评测
python code/d4c.py eval --task 1 --preset step5 --from-run step3_opt_20260329_1200 --step5-run step5_opt_20260329_1300

# Pipeline — 串联 Step3 → Step4 → Step5（Step5 预设固定为 step5）
python code/d4c.py pipeline --task 1 --preset step3
```

帮助：`python code/d4c.py -h` 及各子命令 `python code/d4c.py step3 -h`。

---

## 目录结构（你需要知道的）

| 路径 | 作用 |
|------|------|
| **`code/d4c.py`** | 主线入口：解析预设、校验、打印摘要、`torchrun` 调度各阶段 runner |
| **`code/d4c_core/`** | 配置解析、校验、manifest、dispatch、路径解析 |
| **`code/executors/`** | Step3/4/5 业务实现（engine + torchrun 入口） |
| **`presets/`** | `tasks/`、`training/`、`runtime/`、`decode/` YAML，见 **PRESETS.md** |
| **`sh/`** | Shell 编排（批量、nohup、默认环境）；内部调用 `d4c.py` |
| **`docs/`** | 主指南、`PRESETS.md`、其它说明 |
| **`checkpoints/`** | 权重与中间产物（默认被 `.gitignore`） |
| **`log/`** | 训练/评测日志与 **manifest**（默认被 `.gitignore`） |

数据与预训练权重目录见离线指南；本 README 不展开论文背景。

---

## 预设（presets）摘要

合并顺序（后者覆盖前者）：

**任务表（tasks）→ 训练预设（training）→ 运行预设（runtime）→ 解码预设（decode）→ CLI 非空参数**

- **Task preset**（`presets/tasks/*.yaml`）：每个 `task_id` 的 **auxiliary/target 域对**及默认 **lr/coef/adv** 等任务级标量。  
- **Training preset**（`--preset` → `presets/training/<name>.yaml`）：按任务的 **batch、epochs** 及可覆盖的 lr 等。  
- **Runtime preset**（默认 `presets/runtime/default.yaml`，可用环境变量 **`D4C_RUNTIME_PRESET`** 换名）：**num_proc、ddp_world_size** 等 CPU/DDP 侧。  
- **Decode preset**（`presets/decode/default.yaml` 为底 + CLI **`--decode-preset <stem>`** 浅合并）：**decode_strategy、decode_seed、label_smoothing、repetition_penalty、temperature、top_p** 等。顶层 **只认** `--decode-preset`，勿把 `--decode-strategy` 等 step5 内部参数直接接在 `d4c.py` 后（详见 **PRESETS.md** §1.4）。  
- **CLI**：`--batch-size`、`--epochs`、`--seed`、`--ddp-world-size` 等仅在命令行显式传入时覆盖上述链。

**建议改法**：做新实验优先 **复制/调整 training YAML** 或改 **task 表**；临时试跑用 **CLI 覆盖**；机器相关并发改 **runtime**（或 `D4C_RUNTIME_PRESET`）。

详见 **`docs/PRESETS.md`**（含示例与「该改哪里」）。

---

## 输出产物

| 产物 | 位置 | 说明 |
|------|------|------|
| **Checkpoint** | `checkpoints/<task>/step3_optimized/<run>/` 等 | Step3/4/5 各阶段约定不同，以启动时打印的 `checkpoint_dir` 为准 |
| **日志** | `log/<task>/.../train.log` 或 `eval.log` | 与 `log_dir` 一致；排障首选 |
| **Manifest** | **`<log_dir>/d4c_run_manifest.json`**（文件名固定） | **默认**在每次 `torchrun` **前**写入，记录命令、预设、路径、超参等，便于复现与排障 |
| **Eval 指标** | `metrics_dir`（一般为 checkpoint 下 `eval_runs/`） | 见主指南 |
| **Step5 训练 CSV** | 默认解析为 Step3 目录下 **`factuals_counterfactuals.csv`**（Step4 产出）；可用 CLI 覆盖 | 详见 **PRESETS.md** |
| **Step5 权重** | 默认 **`checkpoint_dir/model.pth`** | Eval 可用 `--model-path` 直接指定 |

关闭 manifest JSON：`export D4C_WRITE_RUN_MANIFEST=0`（stdout 摘要仍会打印）。

---

## 排障入口（按顺序）

1. **`python code/d4c.py -h`** 与子命令 `-h`  
2. **终端 stdout**：`[Resolved Inputs/Outputs]`、`[Manifest]` 行  
3. **`d4c_run_manifest.json`**（与本次 `log_dir` 同目录）  
4. **`train.log` / `eval.log`**  
5. **`docs/D4C_Scripts_and_Runtime_Guide.md`**（路径、Shell、`DDP_NPROC` 等）  
6. **`docs/PRESETS.md`**（配置从哪来、为何与预期不一致）  

高级调试（**非新手必会**）：`D4C_DISPATCH_DETAIL=1` 可打印 `torchrun` 实际薄壳脚本名，见主指南附录。

---

## Shell / 集群（可选）

日常交互式实验 **直接用 `d4c.py`** 即可。需要 **批量任务、nohup、Slurm 包装** 时用 **`sh/run_step*_optimized.sh`** 或 **`scripts/train_ddp.sh`**（GPU/DDP 校验后仍调用 `d4c.py`）。说明见主指南 §1。

---

## 许可与引用

以仓库内随附的许可与论文引用说明为准（如有单独文件请以其为准）。
