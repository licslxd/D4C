# Legacy（考古区）

**不属于**当前主线（`python code/d4c.py …`、`scripts/entrypoints/train_ddp.sh`、`scripts/entrypoints/step{3,4,5}.sh`）。**不保证可运行**，仅供对照旧流程或一次性修补数据。

| 路径 | 说明 |
|------|------|
| **`legacy/code/`** | 历史 `run_all.sh`、`train.py`、`naive_counterfactual_*.py`、`do_stats.py`、`check_eval_metrics_env.py` |
| **`legacy/sh/`** | 旧批量串联脚本（现内部调用 `scripts/entrypoints/*.sh`；考古） |
| **`legacy/tools/fix_eval_csv_header.py`** | 修正旧 `eval_registry*.csv` 表头；`--legacy-only` 可扫仓库根 `log/` 下历史 `eval_runs*.csv` |

入口示例（仓库根）：

```bash
bash legacy/code/run_all.sh
python legacy/tools/fix_eval_csv_header.py --dry-run
bash legacy/sh/run_step3_to_step5_single.sh --task 2 --iter v1
```

批量脚本叙事见 **`docs/legacy_batch_shell.md`**；主线规范见 **`docs/D4C_Scripts_and_Runtime_Guide.md`**。
