import os
import sys
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm
from paths_config import DATA_DIR, MPNET_DIR
from config import get_embed_batch_size

def extract_embeddings_in_batches(encoded_inputs, model, device, batch_size=8):
    # Move model to device
    model.to(device)
    model.eval()

    # Prepare for batch processing
    total_batches = (encoded_inputs.input_ids.size(0) + batch_size - 1) // batch_size
    total_embeddings = []
    for i in tqdm(range(0, encoded_inputs.input_ids.size(0), batch_size), total=total_batches, desc="Processing batches"):
        # Process inputs in batches
        batch_input_ids = encoded_inputs.input_ids[i:i+batch_size].to(device)
        batch_attention_mask = encoded_inputs.attention_mask[i:i+batch_size].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        total_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    return np.concatenate(total_embeddings, axis=0)


if __name__ == "__main__":
    for _arg in sys.argv[1:]:
        if _arg == "--gpus" or _arg.startswith("--gpus="):
            sys.stderr.write(
                "compute_embeddings.py: error: --gpus has been removed.\n"
                "Use --cuda-device N (single process / single GPU) or CUDA_VISIBLE_DEVICES.\n"
            )
            raise SystemExit(2)

    parser = argparse.ArgumentParser(
        description="计算 user/item 嵌入（单进程单 device；不设多卡并行）",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        metavar="N",
        help="CUDA 设备编号（默认 0）；无 CUDA 时自动使用 CPU",
    )
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名，如 'Yelp' 或 'AM_Movies,TripAdvisor,Yelp'，只处理指定数据集；不传则处理全部")
    args = parser.parse_args()

    # 优先用环境变量 EMBED_BATCH_SIZE，否则从 config 读取
    batch_size = int(os.environ.get("EMBED_BATCH_SIZE") or 0) or get_embed_batch_size()
    if torch.cuda.is_available():
        dev = 0 if args.cuda_device is None else int(args.cuda_device)
        device = torch.device(f"cuda:{dev}")
    else:
        device = torch.device("cpu")

    _mpnet = MPNET_DIR if os.path.exists(MPNET_DIR) else "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(_mpnet)
    model = AutoModel.from_pretrained(_mpnet).to(device)
    all_datasets = ["AM_CDs", "AM_Electronics", "AM_Movies", "TripAdvisor", "Yelp"]
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else all_datasets
    for dataset in datasets:
        df = pd.read_csv(os.path.join(DATA_DIR, dataset, "train.csv"))
        nusers = df['user_idx'].max() + 1
        nitems = df['item_idx'].max() + 1

        # user embeddings (review 列可能含 NaN/float，需转为 str 再 join)
        grouped_reviews = df.groupby('user_idx')['review'].apply(
            lambda reviews: ' '.join(str(r) for r in reviews if pd.notna(r))
        )
        encoded_input = tokenizer(
            list(grouped_reviews), padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        embeddings = extract_embeddings_in_batches(encoded_input, model, device, batch_size=batch_size)
        user_embeddings = np.random.rand(nusers, embeddings.shape[1])
        user_embeddings[grouped_reviews.index] = embeddings
        os.makedirs(os.path.join(DATA_DIR, dataset), exist_ok=True)
        np.save(os.path.join(DATA_DIR, dataset, "user_profiles.npy"), user_embeddings)

        # item embeddings (同上，处理 review 列中的 NaN/float)
        grouped_reviews = df.groupby('item_idx')['review'].apply(
            lambda reviews: ' '.join(str(r) for r in reviews if pd.notna(r))
        )
        encoded_input = tokenizer(
            list(grouped_reviews), padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        embeddings = extract_embeddings_in_batches(encoded_input, model, device, batch_size=batch_size)
        item_embeddings = np.random.rand(nitems, embeddings.shape[1])
        item_embeddings[grouped_reviews.index] = embeddings
        np.save(os.path.join(DATA_DIR, dataset, "item_profiles.npy"), item_embeddings)
        
        
