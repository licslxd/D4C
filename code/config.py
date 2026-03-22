import os

from cpu_utils import effective_cpu_count

# 全局配置（Step 1+2 嵌入）
# 可通过 run_step1_step2.sh --embed-batch-size N 或 run_preprocess_and_embed.py --embed-batch-size N 覆盖
embed_batch_size = 1024  # compute_embeddings 嵌入批次大小，显存不足时可减小（64/128），多卡可增大（512/1024）

# 全局配置（CPU 并行）
# 与作业实际可用核数对齐：优先 sched_getaffinity（ cgroup/cpuset 下常为 nproc ），非 Linux 则 os.cpu_count()
# 也可 export RUNNING_CPU_COUNT=12 显式指定（与 shell `nproc` 一致）
# 可通过 run_step3/4/5.sh --num-proc N 或各 Python 脚本 --num-proc N 覆盖
_num_cpu = effective_cpu_count()
# 默认可并联上限：与常见单节点 GPU 作业「最多约 12 核」对齐；更大机器请 export MAX_PARALLEL_CPU=16 等
_MAX_PARALLEL_CPU = max(1, int(os.environ.get("MAX_PARALLEL_CPU", "12")))
# datasets.map（Tokenize）等并行进程数；与 PyTorch DataLoader 的 num_workers 独立，见 get_dataloader_num_workers()
num_proc = min(_num_cpu, _MAX_PARALLEL_CPU)

# 全局配置（Step 3/4/5 训练与推理）
# 可通过 run_step3/4/5.sh --batch-size N 或各 Python 脚本 --batch-size N 覆盖
train_batch_size = 1024  # Step 3 域对抗预训练、Step 4 反事实生成、Step 5 主训练的批次大小，显存不足时可减小
# 可通过 run_step3/5.sh --epochs N 或 torchrun AdvTrain.py train / torchrun run-d4c.py --epochs N 覆盖
epochs = 50  # Step 3 域对抗预训练、Step 5 主训练的轮数

task_configs = {
    1: {
        "auxiliary": "AM_Electronics",
        "target": "AM_CDs",
        "lr": 5e-4,
        "coef": 1,
        "adv": 0.01
    },
    2: {
        "auxiliary": "AM_Movies",
        "target": "AM_CDs",
        "lr": 1e-3,
        "coef": 0.1,
        "adv": 0.01
    },
   3: {
        "auxiliary": "AM_CDs",
        "target": "AM_Electronics",
        "lr": 5e-4,
        "coef": 0.5,
        "adv": 0.1
    },
   4: {
        "auxiliary": "AM_Movies",
        "target": "AM_Electronics",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01
    },
   5: {
        "auxiliary": "AM_CDs",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01
    },
   6: {
        "auxiliary": "AM_Electronics",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01
    },
    7: {
        "auxiliary": "Yelp",
        "target": "TripAdvisor",
        "lr": 1e-4,
        "coef": 0.5,
        "adv": 0.01
    },
    8: {
        "auxiliary": "TripAdvisor",
        "target": "Yelp",
        "lr": 5e-4,
        "coef": 1,
        "adv": 0.01
    }
}

def get_task_config(task_idx):
    return task_configs.get(task_idx, None)


def get_embed_batch_size():
    """返回 embedding 计算的 batch_size，供 run_preprocess_and_embed / compute_embeddings 使用"""
    return embed_batch_size


def get_train_batch_size():
    """返回训练/推理的 batch_size，供 Step 3/4/5 使用"""
    return train_batch_size


def get_epochs():
    """返回训练轮数，供 Step 3/5 使用"""
    return epochs


def get_num_proc():
    """返回 datasets.map（Tokenize）等使用的并行进程数；与 DataLoader 的 num_workers 无关，见 get_dataloader_num_workers()"""
    return num_proc


def get_max_parallel_cpu():
    """与 MAX_PARALLEL_CPU 一致（默认 12）；DDP 每 rank 的 DataLoader worker 上限等可复用。"""
    return _MAX_PARALLEL_CPU


def get_dataloader_num_workers(split="train"):
    """
    PyTorch DataLoader 的 num_workers，与 datasets.map 的 num_proc 独立。
    split: 'train' | 'valid' | 'test' — 训练/验证/推理测试可适当区分上限。
    单路 worker 数不超过 _MAX_PARALLEL_CPU（默认 12），避免在 12 核节点上过度抢占。
    """
    n = _num_cpu or 8
    cap_t = min(_MAX_PARALLEL_CPU, 16)  # 训练侧单 DataLoader 上限
    cap_v = min(max(4, _MAX_PARALLEL_CPU // 2), 8)  # 验证/测试略保守
    if split == "train":
        return min(max(2, n // 2), cap_t)
    if split in ("valid", "test"):
        return min(max(1, n // 4), cap_v)
    return min(max(1, n // 4), cap_v)
