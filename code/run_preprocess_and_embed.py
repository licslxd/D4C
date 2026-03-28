#!/usr/bin/env python3
"""
Step 1 + Step 2 合并脚本：数据预处理 + 嵌入与域语义

用法：
    python run_preprocess_and_embed.py [--embed-batch-size N] [--cuda-device N]
    python run_preprocess_and_embed.py --skip-step1 --embed-datasets Yelp [--cuda-device N]  # 从失败处续跑

参数：
    --embed-batch-size N    compute_embeddings 中 embedding 的批次大小（默认 256）
                            显存不足时可减小（如 64、128）
    --cuda-device N       嵌入计算使用的 GPU 编号（默认 0）；无 CUDA 时用 CPU
    --skip-step1            跳过 Step 1 数据预处理（用于续跑）
    --embed-datasets DS     只运行 compute_embeddings 指定数据集，逗号分隔，如 Yelp
"""

import argparse
import subprocess
import sys
import os
from config import get_embed_batch_size

def run(cmd, desc):
    print(f"\n>>> {desc}")
    print(f"    $ {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"错误: {desc} 执行失败，退出码 {ret.returncode}")
        sys.exit(ret.returncode)

def _reject_legacy_gpus_argv(argv):
    for arg in argv[1:]:
        if arg == "--gpus" or arg.startswith("--gpus="):
            sys.stderr.write(
                "run_preprocess_and_embed.py: error: --gpus has been removed.\n"
                "Use --cuda-device N for single-GPU embedding, or CUDA_VISIBLE_DEVICES.\n"
                "DDP training is torchrun + AdvTrain.py / run-d4c.py (--nproc_per_node=1 is still DDP).\n"
            )
            raise SystemExit(2)


def main():
    _reject_legacy_gpus_argv(sys.argv)
    parser = argparse.ArgumentParser(description="Step 1 + Step 2: 数据预处理与嵌入")
    parser.add_argument("--embed-batch-size", type=int, default=None,
                        help="compute_embeddings 嵌入批次大小（不传则用 config.get_embed_batch_size()）")
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        metavar="N",
        help="compute_embeddings 使用的 CUDA 设备编号（默认 0）",
    )
    parser.add_argument("--skip-step1", action="store_true",
                        help="跳过 Step 1 数据预处理，只执行 Step 2")
    parser.add_argument("--embed-datasets", type=str, default=None,
                        help="只对指定数据集计算嵌入，如 Yelp；不传则处理全部")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not args.skip_step1:
        print("=" * 60)
        print("Step 1: 数据预处理")
        print("=" * 60)
        run("python preprocess_data.py", "预处理原始数据")
        run("python split_data.py", "划分训练/验证/测试集")
        run("python combine_data.py", "合并数据")

    print("\n" + "=" * 60)
    print("Step 2: 嵌入与域语义")
    print("=" * 60)
    batch_size = args.embed_batch_size if args.embed_batch_size is not None else get_embed_batch_size()
    embed_cmd = f"EMBED_BATCH_SIZE={batch_size} python compute_embeddings.py"
    if args.cuda_device is not None:
        embed_cmd += f" --cuda-device {args.cuda_device}"
    if args.embed_datasets:
        embed_cmd += f" --datasets {args.embed_datasets}"
    _dev_desc = f"cuda:{args.cuda_device}" if args.cuda_device is not None else "默认 cuda:0 或 CPU"
    run(
        embed_cmd,
        f"计算 user/item 嵌入 (batch_size={batch_size}, device={_dev_desc}, datasets={args.embed_datasets or '全部'})",
    )
    run("python infer_domain_semantics.py", "推断域语义")

    print("\n" + "=" * 60)
    print("Step 1 + Step 2 完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
