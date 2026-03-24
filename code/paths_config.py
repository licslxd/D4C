# -*- coding: utf-8 -*-
"""
D4C 离线运行路径配置
通过环境变量 D4C_ROOT 指定项目根目录，默认为 code 的上级目录。
可选 D4C_CHECKPOINT_SUBDIR、D4C_CHECKPOINT_GROUP：仅 SUBDIR 时为 checkpoints/<task>/<subdir>/；GROUP+SUBDIR 均设时为 checkpoints/<task>/<group>/<subdir>/。
日志目录见 get_log_task_dir。**仅影响日志**且与 checkpoint 解耦时，可设 D4C_LOG_GROUP / D4C_LOG_SUBDIR（语义与 checkpoint 两变量对称）或单独设 D4C_LOG_STEP（等价于单层 log/<task>/<STEP>/）。
未使用上述 LOG_* 变量时，仍按 checkpoint 环境变量解析（与 checkpoint 不完全对称——当 GROUP+SUBDIR 均设时，日志固定在 log/<task>/<group>/），
runs/<时间戳>/train.log 等由 shell 传入 --log_file；eval 汇总由 train_logging 写入同级的 eval/ 子目录（eval_runs.*）；权重仍在 …/<group>/<subdir>/。
"""
import os

# 项目根目录：默认为 D4C-main
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
D4C_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if os.environ.get("D4C_ROOT"):
    D4C_ROOT = os.path.abspath(os.environ["D4C_ROOT"])

# 预训练模型本地路径（需提前下载到以下目录）
MODELS_DIR = os.path.join(D4C_ROOT, "pretrained_models")
T5_SMALL_DIR = os.path.join(MODELS_DIR, "t5-small")
MPNET_DIR = os.path.join(MODELS_DIR, "sentence-transformers_all-mpnet-base-v2")
# 兼容别名
MPNET_PATH = MPNET_DIR
METEOR_CACHE = os.path.join(MODELS_DIR, "evaluate_meteor")
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"  # 需下载到 HF_HOME

# 数据目录
DATA_DIR = os.path.join(D4C_ROOT, "data")
MERGED_DATA_DIR = os.path.join(D4C_ROOT, "Merged_data")
# 模型权重与反事实 CSV（见 get_checkpoint_task_dir）
# D4C_CHECKPOINT_SUBDIR：轮次/实验名；D4C_CHECKPOINT_GROUP：可选中间层；仅 SUBDIR 时 checkpoints/<task>/<subdir>/，GROUP+SUBDIR 时 …/<group>/<subdir>/
_ckpt_sub = os.environ.get("D4C_CHECKPOINT_SUBDIR", "").strip()
_ckpt_group = os.environ.get("D4C_CHECKPOINT_GROUP", "").strip()
CHECKPOINT_DIR = os.path.join(D4C_ROOT, "checkpoints")


def get_checkpoint_task_dir(task_idx):
    """单任务目录：checkpoints/<task>/ 或 checkpoints/<task>/<subdir>/ 或 checkpoints/<task>/<group>/<subdir>/"""
    t = str(task_idx)
    base = os.path.join(D4C_ROOT, "checkpoints", t)
    if _ckpt_sub:
        if _ckpt_group:
            return os.path.join(base, _ckpt_group, _ckpt_sub)
        return os.path.join(base, _ckpt_sub)
    if _ckpt_group:
        return os.path.join(base, _ckpt_group)
    return base


def get_log_task_dir(task_idx):
    """单任务日志根目录：其下可有 runs/<时间戳>/train.log（由 run_step3/4/5.sh 约定）；eval 汇总在子目录 eval/。

    优先级（从高到低）：
    1. **仅日志** — 若 ``D4C_LOG_SUBDIR`` 或 ``D4C_LOG_GROUP`` 任一非空：布局与 checkpoint 两变量对称
       （二者均非空时目录为 ``log/<task>/<D4C_LOG_GROUP>/``，不按 LOG_SUBDIR 再分层）。
    2. **仅日志** — 否则若 ``D4C_LOG_STEP`` 非空：``log/<task>/<D4C_LOG_STEP>/``（与 checkpoint 无关）。
    3. **沿用 checkpoint 环境变量**（与 get_checkpoint_task_dir 使用同一组 D4C_CHECKPOINT_*）：
       - 无 SUBDIR：log/<task>/ 或（仅有 GROUP 时）log/<task>/<group>/
       - 仅有 SUBDIR：log/<task>/<subdir>/
       - GROUP 与 SUBDIR 均设：checkpoint 为 …/<group>/<subdir>/，日志统一在 log/<task>/<group>/（不按 SUBDIR 再分子目录）
    """
    t = str(task_idx)
    base = os.path.join(D4C_ROOT, "log", t)
    log_sub = os.environ.get("D4C_LOG_SUBDIR", "").strip()
    log_group = os.environ.get("D4C_LOG_GROUP", "").strip()
    log_step = os.environ.get("D4C_LOG_STEP", "").strip()
    if log_sub or log_group:
        if log_sub:
            if log_group:
                return os.path.join(base, log_group)
            return os.path.join(base, log_sub)
        return os.path.join(base, log_group)
    if log_step:
        return os.path.join(base, log_step)
    if _ckpt_sub:
        if _ckpt_group:
            return os.path.join(base, _ckpt_group)
        return os.path.join(base, _ckpt_sub)
    if _ckpt_group:
        return os.path.join(base, _ckpt_group)
    return base


# 可选镜像日志：设置环境变量 D4C_MIRROR_LOG=1 时，额外追加到 code/log.out（默认关闭，避免单文件无限增长）
CODE_DIR = _SCRIPT_DIR
DEFAULT_MIRROR_LOG = os.path.join(CODE_DIR, "log.out")


def _mirror_enabled():
    v = os.environ.get("D4C_MIRROR_LOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def append_log_dual(primary_log_file, text, mirror=None):
    """写入 primary_log_file（若给定）。mirror=True 或环境变量 D4C_MIRROR_LOG=1 时再写入 code/log.out。"""
    if mirror is None:
        mirror = _mirror_enabled()
    paths = []
    if primary_log_file:
        paths.append(os.path.abspath(os.path.expanduser(primary_log_file)))
    if mirror:
        paths.append(os.path.abspath(DEFAULT_MIRROR_LOG))
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


def get_data_path(dataset):
    """获取数据集路径"""
    return os.path.join(DATA_DIR, dataset)


def get_merged_path(task_idx):
    """获取 Merged_data 中任务目录"""
    return os.path.join(MERGED_DATA_DIR, str(task_idx))


def get_t5_tokenizer_path():
    """T5 tokenizer 本地路径"""
    return T5_SMALL_DIR


def get_mpnet_path():
    """MPNet 模型本地路径"""
    return MPNET_DIR
