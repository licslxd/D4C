# Legacy：历史 Shell 批量串联脚本

本页记录 **legacy/sh/** 下的批量编排脚本，**不保证**与当前预设/路径校验长期一致。日常请用：

- **Python**：`python code/d4c.py pipeline|step3|step4|step5|eval …`（仓库根）
- **官方 Bash 批量**：`bash scripts/entrypoints/train_ddp.sh …`（GPU/DDP 校验后调用 `d4c.py`）
- **单阶段**：`scripts/entrypoints/step3.sh`、`scripts/entrypoints/step4.sh`、`scripts/entrypoints/step5.sh`

## 迁出脚本（`legacy/sh/`）

| 脚本 | 原用途 |
|------|--------|
| `legacy/sh/run_step3_to_step5_all.sh` | 任务 1–8 依次 Step3→4→5 |
| `legacy/sh/run_step3_to_step5_single.sh` | 单任务 Step3→4→5，`--task N` |
| `legacy/sh/run_step5_all.sh` | 任务 1–8 仅 Step5 批量 |

以上脚本通过 **`$D4C_ROOT/scripts/entrypoints/*.sh`** 调用主线逻辑；自身路径相对 `legacy/sh/` 解析仓库根。

## 相关

- 手写 `torchrun`（非推荐）：`docs/legacy_offline_torchrun.md`
- 历史 Python / `run_all.sh`：`legacy/code/` 与 `legacy/README.md`
