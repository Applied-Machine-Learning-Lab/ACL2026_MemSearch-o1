import hashlib
from datetime import datetime
from openai import OpenAI
import sys
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

API_SECRET_KEY = "sk-zk22528c689c68abb04cbfaf2ec6373a09868bbdef5a073b"
BASE_URL = "https://api.zhizengzeng.com/v1"

class MemoryPool:
    """可更新维护的记忆池 - 所有变量均为字符串格式"""
    
    def __init__(self, model_name):
        self.memory_pool: str = ""  # 存储所有记忆内容的字符串
        self.query_entities = ""
        self.key_entities: str = ""  # 存储关键实体，用逗号分隔
        self.history_documents: str = ""  # 历史文档存储，用特殊分隔符分隔
        self.model_name = model_name
        
    def add_document(self, document: str):
        """添加历史文档用于后续检索"""
        if self.history_documents:
            self.history_documents += "|||" + document
        else:
            self.history_documents = document
    
    def process_new_text(self, new_text: str, query: str) -> str:
        """处理新文本片段的核心方法"""
        
        entities = self._extract_key_entities_from_query(query)
    
        # 3. 提取相关实体和内容
        response1 = self._query_guided_entity_contents(new_text, entities, query)
        # response = self._extract_entity_contents(new_text, entities, query)
        
        # 4. 添加到记忆池
        # response2 = self._query_guided_contents(new_text, query)
        response2 = ''
        self.memory_pool = response1 + response2
        
        return self.memory_pool, response2
    
    def _extract_key_entities_from_query(self, query: str) -> str:
        """初始提取query中最关键实体"""
        prompt = f"""
        请从以下问题中提取最重要的关键实体、动词和时间：
        问题: {query}

        仅返回用逗号分隔的关键实体、动词和时间，例如：“实体1, 实体2, 动词1, 动词2, 时间1, 时间2...”。
        最多返回最重要的15项（实体、动词、时间合计不超过15个）。
        """
        
        response = self.memory_manager(self.model_name, prompt)
        return response.strip()

    def _query_guided_entity_contents(self, new_text, entities, query):
        """根据query指导提取实体内容"""
        if not entities:
            return ""
        
        prompt = f"""
        请根据与查询上下文的相关性，从新的搜索结果中为以下实体提取内容：
        实体: {entities}
        新文本: {new_text}
        查询: {query}

        请为每个实体提取最有助于回答该查询的内容，并按如下格式输出：“实体名: 关于该实体的内容”
        不要使用你自己的语言描述实体内容。
        每个实体-内容对单独占一行。
        聚焦上下文信息、实体间关系及关键事实。
        如果某个实体在当前查询下无有用信息，请跳过该实体。
        """
        response = self.memory_manager(self.model_name, prompt)
        return response

    def _query_guided_contents(self, new_text, query):
        
        prompt = f"""
        请根据与查询上下文的相关性，提取与查询内容相关的内容：
        新文本: {new_text}
        查询: {query}

        请提取有助于回答该查询的内容，
        如果有多部分信息和查询相关，请分点叙述相关信息。
        聚焦上下文信息，根据查询的需求，目的，语义提取有用的信息。
        提取的信息不需要总结，保持原文的叙述和信息完整性。
        """
        response = self.memory_manager(self.model_name, prompt)
        return response

    def memory_manager(self, model_name, prompt: str) -> str:
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
        response = resp.choices[0].message.content
    
        return response.strip()
        
def generator(model_name, prompt: str) -> str:
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])
    response = resp.choices[0].message.content
    
    return response.strip()
        
def generate_response(model_name, memory, query, instruction):
    # data_name = 'memory'
    prompt = f"""
请根据记忆线索回答问题。{instruction}
记忆线索: {memory}
问题: {query}
请**逐步思考**！
请基于提供的记忆信息，以及其中实体、事件和时间之间的关系，生成准确的答案。
"""
    response = generator(model_name, prompt)
    return response  # 建议补上 return，原代码可能遗漏

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


def insert_texts_with_retry(rag, texts, retries=3, delay=5):
    for _ in range(retries):
        try:
            rag.insert(texts)
            return
        except Exception as e:
            print(
                f"Error occurred during insertion: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
    raise RuntimeError("Failed to insert texts after multiple retries.")

def main():
    # 初始化记忆池
    memory_pool = MemoryPool()
    
    # 处理新文本
    new_text = "Deep learning is a subset of machine learning that uses multi-layered neural networks. It has achieved breakthrough results in image recognition and natural language processing."
    new_text2 = "There are many papers in the domain of deep learning. One representative famous one is Deep Residual Network. This network is especially effective for pattern recognition in computer vision. The computer vision is also a famous domain in artificial intelligence. The artificial intelligence is part of machine leanrning."
    new_text3 = "I like bananas, and John like peaches. We both like fruits and papers."
    query = "Which famous paper related to artificial intelligence is mentioned?"
    
    memory = memory_pool.process_new_text(new_text, query)
    memory = memory_pool.process_new_text(new_text2, query)
    memory = memory_pool.process_new_text(new_text3, query)
    response = generate_response(memory, query)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()