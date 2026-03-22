#!/usr/bin/env python3
"""
Step 1 + Step 2 合并脚本：数据预处理 + 嵌入与域语义

用法：
    python run_preprocess_and_embed.py [--embed-batch-size N] [--gpus 0,1]
    python run_preprocess_and_embed.py --skip-step1 --embed-datasets Yelp [--gpus 0,1]  # 从失败处续跑

参数：
    --embed-batch-size N    compute_embeddings 中 embedding 的批次大小（默认 256）
                            显存不足时可减小（如 64、128），多卡时可增大（如 1024）
    --gpus 0,1             多卡时指定 GPU ID，逗号分隔
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

def main():
    parser = argparse.ArgumentParser(description="Step 1 + Step 2: 数据预处理与嵌入")
    parser.add_argument("--embed-batch-size", type=int, default=None,
                        help="compute_embeddings 嵌入批次大小（不传则用 config.get_embed_batch_size()）")
    parser.add_argument("--gpus", type=str, default=None,
                        help="逗号分隔的 GPU ID，如 '0,1' 使用多卡，不传则单卡")
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
    if args.gpus:
        embed_cmd += f" --gpus {args.gpus}"
    if args.embed_datasets:
        embed_cmd += f" --datasets {args.embed_datasets}"
    run(embed_cmd,
        f"计算 user/item 嵌入 (batch_size={batch_size}, gpus={args.gpus or '单卡'}, datasets={args.embed_datasets or '全部'})")
    run("python infer_domain_semantics.py", "推断域语义")

    print("\n" + "=" * 60)
    print("Step 1 + Step 2 完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
