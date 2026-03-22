import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths_config import DATA_DIR, MPNET_DIR


def extract_embeddings_in_batches(encoded_inputs, model, device, batch_size=8):
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
    import random
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _mpnet = MPNET_DIR if os.path.exists(MPNET_DIR) else "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(_mpnet)
    model = AutoModel.from_pretrained(_mpnet)

    datasets = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]
    domain2sentences = {}
    for i, dataset in enumerate(datasets):
        df = pd.read_csv(os.path.join(DATA_DIR, dataset, "train.csv"))
        sentences = []
        for j in range(len(df)):
            sentences.append(df.iloc[j]["explanation"])
        domain2sentences[dataset] = sentences


    domain2embeddings = {}
    for i, dataset in enumerate(datasets):
        sentences = domain2sentences[dataset]
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=100, return_tensors='pt').to(device)
        embeddings = extract_embeddings_in_batches(encoded_input, model, device, batch_size=512)
        domain2embeddings[dataset] = embeddings


    pairs = [
        ("AM_Electronics", "AM_CDs"),
        ("AM_Movies", "AM_CDs"),
        ("AM_CDs", "AM_Electronics"),
        ("AM_Movies", "AM_Electronics"),
        ("AM_CDs", "AM_Movies"),
        ("AM_Electronics", "AM_Movies"),
        ("Yelp", "TripAdvisor"),
        ("TripAdvisor", "Yelp")
    ]

    counterfactual_explanations = []
    for auxiliary_domain, target_domain in pairs:
        M = torch.tensor(domain2embeddings[auxiliary_domain], dtype=torch.float32).to(device)
        N = torch.tensor(domain2embeddings[target_domain], dtype=torch.float32).to(device)
        
        # Normalize M and N for cosine similarity calculation
        M_norm = M / M.norm(dim=1, keepdim=True)
        N_norm = N / N.norm(dim=1, keepdim=True)

        # Array to hold the most similar vectors
        similar_N = torch.empty_like(N_norm)
        most_similar_indexs = []
        # Compute cosine similarity for each vector in N against all in M, one by one
        for i, n_vec in tqdm(enumerate(N_norm),total=len(N_norm)):
            similarity_scores = torch.matmul(M_norm, n_vec.unsqueeze(1)).squeeze()
            most_similar_index = torch.argmax(similarity_scores).item()
            most_similar_indexs.append(most_similar_index)
        sentences = domain2sentences[auxiliary_domain]
        counterfactual_explanations.append([sentences[i] for i in most_similar_indexs])
    print(f"Computed {len(counterfactual_explanations)} counterfactual explanation sets")

    for index, pair in enumerate(pairs):
        target_dataset = pair[1]
        out_dir = os.path.join(DATA_DIR, "Augmentation1", str(index+1))
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(os.path.join(DATA_DIR, target_dataset, "train.csv"))
        assert len(df) == len(counterfactual_explanations[index]), "not equal"
        df["counterfactual"] = counterfactual_explanations[index]
        df.to_csv(os.path.join(out_dir, "train.csv"), index=False)

    for index, pair in enumerate(pairs):
        target_dataset = pair[1]
        df = pd.read_csv(os.path.join(DATA_DIR, "Augmentation1", str(index+1), "train.csv"))
        df_long = pd.melt(df, id_vars=['user', 'item', 'rating', 'review', 'user_idx', 'item_idx'], value_vars=['explanation', 'counterfactual'], var_name='type', value_name='text')
        #rename
        df_long.rename(columns={'text': 'explanation'}, inplace=True)
        #drop
        df_long = df_long.drop(columns=['type'])
        df_long.to_csv(os.path.join(DATA_DIR, "Augmentation1", str(index+1), "aug_train.csv"), index=False)