import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch.nn.functional as F

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)

    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1) 

    return torch.mm(a_norm, b_norm.transpose(0, 1))  # (n, m)

model_path = "./retrieve_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Encode the documents
def encode_texts(texts, batch_size=8):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=8192
            ).to(device)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


# Retreive Function
def retrieve(documents, query, top_k=3):
    doc_embeddings = encode_texts(documents)
    query_embedding = encode_texts([query])
    similarities = cos_sim(query_embedding, doc_embeddings)
    top_results = torch.topk(similarities, k=top_k, dim=1)
    
    results = []
    for score, idx in zip(top_results.values[0], top_results.indices[0]):
        results.append({
            "score": score.item(),
            "text": documents[idx.item()]
        })
    return results

import torch

def route(mem_instance, query, search_query_list):
    Q = encode_texts([query])      # shape: (1, d)
    X = encode_texts(mem_instance) # shape: (n, d)

    q_score = []
    for q in search_query_list:
        q_emb = encode_texts([q])
        q_score.append(cos_sim(X, q_emb.reshape(1, -1)).flatten())
    sim_q = cos_sim(X, Q.reshape(1, -1)).flatten()  # (n,)
    q_score.append(sim_q)
    stacked_scores = torch.stack(q_score)
    sim_q, _ = torch.max(stacked_scores, dim=0)
    sim_xx = cos_sim(X, X)  # (n, n)

    mask_diag = 1 - torch.eye(sim_xx.size(0), device=sim_xx.device)
    sim_xx = sim_xx * mask_diag

    # Set Thresholds
    tau = 0.3

    # Query-Conditioned Bridge Potential
    bridge_potential = torch.zeros(len(X), device=X.device)

    # ReLU-style soft threshold
    weights = torch.clamp(sim_q - tau, min=0.0)

    for i in range(len(X)):
        mask = torch.ones(len(X), dtype=torch.bool, device=X.device)
        mask[i] = False  # exclude self

        weighted_sims = sim_xx[i] * weights
        total_weight = weights[mask].sum()

        if total_weight > 0:
            bridge_potential[i] = weighted_sims[mask].sum() / total_weight
        else:
            bridge_potential[i] = 0.0

    # Combination
    alpha, beta = 0.6, 0.4
    C = alpha * sim_q + beta * bridge_potential

    current = Q
    visited = set()
    path = []

    for _ in range(min(15, len(X))):
        candidates = []
        for i in range(len(X)):
            if i not in visited:
                cont_score = cos_sim(current, X[i].unsqueeze(0)).item()
                candidates.append((i, C[i].item(), cont_score))

        if not candidates:
            break

        scores = torch.tensor([
            c_score * torch.exp(torch.tensor(-1.0 * (1 - cont_score))).item()
            for _, c_score, cont_score in candidates
        ], device=X.device)

        best_idx_in_candidates = torch.argmax(scores).item()
        best_idx = candidates[best_idx_in_candidates][0]

        path.append(best_idx)
        visited.add(best_idx)
        current = X[best_idx].unsqueeze(0)

    prompt = f"[Query]：{query}\n\n[Information Flow for Reasoning]\n"
    for step, idx in enumerate(path, 1):
        memory_piece = mem_instance[idx]
        # prompt += f"Step {step}: {memory_piece}\n"
        prompt += f"{memory_piece}\n"

    return prompt
    
