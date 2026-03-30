# D4C 预设（Presets）说明

本文说明 **`python code/d4c.py`** 使用的 YAML 预设如何合并、各自管什么、以及实验时建议先改哪里。  
实现代码：`code/d4c_core/config_loader.py`（与内置 `code/config.py` 中的旧字典并行存在时，以 **d4c 主线** 为准）。

---

## 1. 四类预设分别决定什么？

### 1.1 Task preset（`presets/tasks/*.yaml`）

- **决定**：每个 **`--task 1..8`** 对应的 **`auxiliary` / `target` 域名字符串**，以及任务级默认 **`lr`、`coef`、`adv`**（若训练预设未覆盖）。  
- **合并**：目录下多个 yaml **按文件名排序后合并**；须覆盖任务 1–8。  
- **适合写进 YAML**：长期稳定的任务定义、默认优化强度。  
- **更适合 CLI**：一般不通过 CLI 改域对；改任务号用 `--task`。

### 1.2 Training preset（`--preset` → `presets/training/<name>.yaml`）

- **决定**：在某一训练预设 **切片** 下，每个 task 的 **`train_batch_size`、`epochs`**，以及可选的 **`lr` / `coef` / `adv` 覆盖**。  
- **典型文件**：`step3.yaml`（Step3/4 常用）、`step5.yaml`（Step5 / eval 常用）。  
- **适合写进 YAML**：正式实验批量、论文复现表。  
- **更适合 CLI**：临时试跑 `batch_size` / `epochs`（`--batch-size`、`--epochs`）。

### 1.3 Runtime preset（`presets/runtime/<name>.yaml`）

- **默认文件**：`presets/runtime/default.yaml`。  
- **切换方式**：环境变量 **`D4C_RUNTIME_PRESET=<stem>`**（无则 `default`；若 `presets/runtime/<stem>.yaml` 不存在则回退 `default.yaml` 逻辑以代码为准）。  
- **决定**：**`num_proc`**（datasets.map）、**`ddp_world_size`**（默认 2）等 **CPU / DDP 进程数** 相关字段。  
- **可被 CLI 覆盖**：`--num-proc`、`--ddp-world-size`。  
- **注意**：`D4C_RUNTIME_PRESET` **会改变主流程行为**（并发与 DDP 规模），应在脚本或文档中写明，避免「换机器忘了 export」。

### 1.4 Decode preset（`presets/decode/*.yaml`）

- **决定**：**`decode_strategy`、`decode_seed`、`max_explanation_length`、`label_smoothing`、`repetition_penalty`、`generate_temperature`、`generate_top_p`**（进入 `ResolvedConfig`，经 `d4c.py` 传给 Step5 `train`/`eval`）。  
- **合并**：始终以 **`presets/decode/default.yaml`** 为底，再与 **`--decode-preset <stem>`** 对应的 **`presets/decode/<stem>.yaml`** 做浅层合并（未传则仅 `default`）。  
- **适合写进 YAML**：固定 greedy / nucleus 温度等实验组（如 `greedy.yaml`、`nucleus_t08_p09.yaml`）。  
- **CLI（主线）**：**仅** **`--decode-preset <stem>`**。请勿在 `python code/d4c.py …` 后直接使用 **`--decode-strategy` / `--generate-temperature` / `--generate-top-p` / `--decode-seed` / `--repetition-penalty` / `--max-explanation-length`**（这些属于 **torchrun 内的 step5 runner / run-d4c**；误传顶层时会得到明确迁移提示）。  
- **命名辨析**：manifest / stdout 里 **`decode_preset`** 是所选预设 **文件名 stem**（如 `default`、`greedy`）；**`decode_strategy`**（在 `decode_resolved` 内）是解析后的算法（`greedy` 或 `nucleus`）。因此 **`decode_preset=default` 且 `decode_strategy=greedy` 可同时成立**（默认预设本身就是 greedy 策略）。当前仓库中 **`default` 与 `greedy` 两套 YAML 合并结果等价**，仅 manifest 中的 `decode_preset` 不同。  
- **高级**：仍可用 **`D4C_RUN_D4C_EXTRA`** 向 step5 子进程追加 **非 decode 主路径** 的 run-d4c 参数（如 `--eval-single-process-safe`）；decode 口径请优先用 `--decode-preset`。

### 1.5 CLI override

- **生效点**：在 **上述链全部合并之后**，仅对 **命令行显式传入且非 None** 的项覆盖：`--batch-size`、`--epochs`、`--num-proc`、`--seed`、`--ddp-world-size`、`--eta`（Step5 反事实权重系数）等。  
- **不改变**：decode 除 `--decode-preset` 所选文件外，其余字段仍靠 YAML；更细的 run-d4c 开关可走 `D4C_RUN_D4C_EXTRA`（非新手路径）。

---

## 2. 合并顺序（一句话）

**tasks → training(`--preset`) → runtime(`D4C_RUNTIME_PRESET` 或 default) → decode(default + `--decode-preset`) → CLI**

---

## 3. 推荐修改顺序（做实验时）

1. **先定任务与阶段**：`--task`、`step3|step4|step5|eval`、`--from-run` / `--step5-run`。  
2. **再定训练语义**：复制一份 `presets/training/my_exp.yaml`，改 batch/epochs/lr，用 `--preset my_exp`。  
3. **机器相关**：改 `presets/runtime/*.yaml` 或设 `D4C_RUNTIME_PRESET`，最后再用 CLI 微调 `--ddp-world-size` / `--num-proc`。  
4. **解码/指标口径**：改 `presets/decode/default.yaml`（团队内注意同步版本）。  
5. **临时一次性试跑**：优先 **CLI** 覆盖 batch/epochs/seed，避免污染共享 YAML。

**不建议**：在未理解路径约定前改 `code/executors/*_engine.py` 内硬编码；路径与 checkpoint 规则以 **`docs/D4C_Scripts_and_Runtime_Guide.md`** 为准。

---

## 4. 示例

### 示例 A：用 **task 4** 跑 Step3（training 用 `step3`）

```bash
python code/d4c.py step3 --task 4 --preset step3
```

- **task 4** 的 auxiliary/target 来自 **`presets/tasks/*.yaml`** 中键 `4`。  
- **batch/epochs** 来自 **`presets/training/step3.yaml`** 中 task `4` 段。  
- **num_proc / ddp_world_size** 来自 **runtime**（及可选 CLI）。

### 示例 B：Step5 训练（**training preset 用 `step5`**）

```bash
python code/d4c.py step5 --task 4 --preset step5 \
  --from-run step3_opt_20260329_1200 --step5-run step5_opt_20260329_1400
```

- **训练超参** 以 **`presets/training/step5.yaml`** 中 task 4 为主。  
- **`--eta`** 默认链上来自 adv，可在 CLI 显式覆盖。

### 示例 C：Eval / 解码相关标量

Eval 同样走 **`ResolvedConfig`**；decode 由 **`default.yaml` + `--decode-preset`** 合并后写入 manifest 的 **`decode_resolved`**，并传给 Step5 eval。

```bash
python code/d4c.py eval --task 4 --preset step5 \
  --from-run step3_opt_20260329_1200 --step5-run step5_opt_20260329_1400 \
  --decode-preset nucleus_t08_p09
```

若需其它 run-d4c 独占参数（且不宜用 YAML 表达），可查阅主指南 **高级** `D4C_RUN_D4C_EXTRA`；decode 相关请优先增加或选用 `presets/decode/*.yaml` + `--decode-preset`。

---

## 5. 与 manifest 的关系

每次主线运行（默认）在 **`log_dir/d4c_run_manifest.json`** 中记录 **`training_preset`、`runtime_preset`、`decode_preset`**、域对、路径与关键超参，用于对照「我以为的预设」与「实际解析结果」。详见 **README** 与 **`docs/D4C_Scripts_and_Runtime_Guide.md`** §1.3.1。

---

## 6. 相关文档

| 文档 | 内容 |
|------|------|
| `README.md` | 快速上手、产物路径、排障入口 |
| `docs/D4C_Scripts_and_Runtime_Guide.md` | Shell、路径、`DDP_NPROC`、高级环境变量附录 |
| `D4C_离线完整指南.md` | 离线数据、集群与补充命令 |
