import os
import torch
from transformers import AutoTokenizer, AutoModel
# from sentence_transformers.util import cos_sim
import numpy as np
import torch.nn.functional as F

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算两个张量之间的余弦相似度，等效于 sentence_transformers.util.cos_sim
    
    Args:
        a: Tensor of shape (n, d)
        b: Tensor of shape (m, d)
    
    Returns:
        Tensor of shape (n, m) with cos similarity between each pair
    """
    if a.ndim == 1:
        a = a.unsqueeze(0)  # (d,) -> (1, d)
    if b.ndim == 1:
        b = b.unsqueeze(0)  # (d,) -> (1, d)

    # 归一化向量（L2 归一化）
    a_norm = F.normalize(a, p=2, dim=1)  # (n, d)
    b_norm = F.normalize(b, p=2, dim=1)  # (m, d)

    # 计算点积，得到余弦相似度
    return torch.mm(a_norm, b_norm.transpose(0, 1))  # (n, m)

# 1. 从本地路径加载 BGE-M3 模型和分词器
model_path = "./retrieve_model"  # 替换为你的本地模型路径

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

# 将模型移到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 文档集合（用于构建检索库）
documents = [
    "人工智能是计算机科学的一个分支，致力于创造能够执行人类智能任务的系统。",
    "大语言模型通过在大量文本上训练，学习语言的统计规律。",
    "北京是中国的首都，拥有丰富的历史和文化。",
    "杭州以西湖和互联网科技产业闻名。",
    "RAG 模型结合了检索和生成，提升生成内容的准确性。"
]

# 3. 对文档进行编码（批量编码）
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
            outputs = model(**inputs)
            # 取 [CLS] 向量并归一化（BGE 推荐）
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

# # 编码所有文档
# doc_embeddings = encode_texts(documents)
# print(f"文档编码完成，形状: {doc_embeddings.shape}")

# 4. 检索函数：给定查询，返回最相似的文档
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

    # Step 2: 计算 Query-Conditioned 贡献度
    q_score = []
    for q in search_query_list:
        q_emb = encode_texts([q])
        q_score.append(cos_sim(X, q_emb.reshape(1, -1)).flatten())
    sim_q = cos_sim(X, Q.reshape(1, -1)).flatten()  # (n,)
    q_score.append(sim_q)
    stacked_scores = torch.stack(q_score)  # shape: (3, 7)
    sim_q, _ = torch.max(stacked_scores, dim=0)  # shape: (7,)
    sim_xx = cos_sim(X, X)  # (n, n)

    # 自己和自己相似度设为0（替代 np.fill_diagonal）
    mask_diag = 1 - torch.eye(sim_xx.size(0), device=sim_xx.device)
    sim_xx = sim_xx * mask_diag

    # 设置 query 相关性阈值
    tau = 0.3  # 可调整，或用 torch.quantile(sim_q, 0.3)

    # 计算 Query-Conditioned Bridge Potential
    bridge_potential = torch.zeros(len(X), device=X.device)

    # ReLU-style soft threshold
    weights = torch.clamp(sim_q - tau, min=0.0)  # 等价于 np.maximum(sim_q - tau, 0)

    for i in range(len(X)):
        mask = torch.ones(len(X), dtype=torch.bool, device=X.device)
        mask[i] = False  # exclude self

        weighted_sims = sim_xx[i] * weights  # (n,)
        total_weight = weights[mask].sum()

        if total_weight > 0:
            bridge_potential[i] = weighted_sims[mask].sum() / total_weight
        else:
            bridge_potential[i] = 0.0

    # 组合最终贡献度
    alpha, beta = 0.6, 0.4
    C = alpha * sim_q + beta * bridge_potential  # (n,)

    # 贪心路径构建
    current = Q  # shape: (1, d)
    visited = set()
    path = []

    for _ in range(min(15, len(X))):  # 最多15
        # 候选：未访问的索引
        candidates = []
        for i in range(len(X)):
            if i not in visited:
                # 计算当前点与 X[i] 的连续性（语义相似度）
                cont_score = cos_sim(current, X[i].unsqueeze(0)).item()  # 标量
                candidates.append((i, C[i].item(), cont_score))

        if not candidates:
            break

        # 计算选择得分：贡献度 * exp(- (1 - 连续性))
        scores = torch.tensor([
            c_score * torch.exp(torch.tensor(-1.0 * (1 - cont_score))).item()
            for _, c_score, cont_score in candidates
        ], device=X.device)

        best_idx_in_candidates = torch.argmax(scores).item()
        best_idx = candidates[best_idx_in_candidates][0]

        path.append(best_idx)
        visited.add(best_idx)
        current = X[best_idx].unsqueeze(0)  # 保持 shape (1, d)

    # 构建最终 prompt
    prompt = f"[Query]：{query}\n\n[Information Flow for Reasoning]\n"
    for step, idx in enumerate(path, 1):
        memory_piece = mem_instance[idx]
        # prompt += f"Step {step}: {memory_piece}\n"
        prompt += f"{memory_piece}\n"

    return prompt
    