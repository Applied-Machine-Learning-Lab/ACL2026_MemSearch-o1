import transformers
import torch
import random
from datasets import load_dataset
import requests
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from bge import retrieve


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def search(query, retriever):
    result = retriever.get_relevant_documents(query)
    result = "".join([f"Doc {i+1}: {r.page_content}\n" for i, r in enumerate(result)])

    return result

# def search(parts, query):
#     result = retrieve(parts, query, top_k=3)
#     result = "".join([f"Doc {i+1}: {r}\n" for i, r in enumerate(result)])

#     return result