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
        # response1 = self._query_guided_entity_contents(new_text, entities, query)
        response1 = self._extract_entity_contents(new_text, entities, query)
        
        # 4. 添加到记忆池
        # response2 = self._query_guided_contents(new_text, query)
        # print('=======*******')
        # print(response2)
        response2 = ''
        self.memory_pool = response1

        
        return self.memory_pool, response2
    
    def _extract_key_entities_from_query(self, query):
        doc = nlp(query)
        noun = [chunk.text for chunk in doc.noun_chunks]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        time = []
        for entity in doc.ents:
            if entity.label_ == 'DATE':
                time.append(entity.text)
        entities = noun + verbs + time
        entities = '\n'.join([f"{r}," for i, r in enumerate(entities)])
        print("****************")
        print(entities)
        return entities
    # def _extract_key_entities_from_query(self, query: str) -> str:
    #     """初始提取query中最关键实体"""
    #     prompt = f"""
    #     Please extract the most important key entities, the verbs, and the time from the following query:
    #     Query: {query}
        
    #     Return only the key entities separated by commas, for example: "entity1, entity2, entity3, verb1, verb2, time1, time2...".
    #     Only return the 15 most important entities, verbs and time.
    #     """
        
    #     response = self.memory_manager(self.model_name, prompt)
    #     return response.strip()

    def _extract_entity_contents(self, text: str, entities: str, query: str) -> str:
        """提取指定实体的内容（格式：entity_name: content）"""
        if not entities:
            return ""
            
        prompt = f"""
        Please extract content that is useful for inferencing query corresponding to the following entities, verbs or time from the text:
        Entities, verbs or time: {entities}
        Text: {text}
        Query: {query}
        
        For each entity, extract raw relevant content and format like this example: "entity_name: content about the entity" or "verbs: content about something or someone takes the action on something or someone or "time: the event that happens".
        Do not use your own words to describe entity contents.
        You must put each entity, verb, time-content pair on a separate line. 
        Contents of entity, verb, time **must include diverse information related to the query**.
        If no useful information for the query, skip this entity.
        """
        
        response = self.memory_manager(self.model_name, prompt)
        return response.strip()

    def _query_guided_contents(self, new_text, query):
        
        prompt = f"""
        Extract only the content from the new text that is relevant to the query, based on semantic and contextual relevance:

        New text: {new_text}
        Query: {query}

        Identify and extract specific portions of the text that directly help answer the query.
        If multiple parts are relevant, list them as separate points.
        Do NOT summarize, paraphrase, or interpret the content — preserve the original wording and information exactly as it appears.
        Focus strictly on the context provided; extract only what is necessary to address the intent, purpose, and meaning of the query.
        """
        response = self.memory_manager(self.model_name, prompt)
        return response
        
    def _query_guided_entity_contents(self, new_text, entities, query):
        # doc = nlp(new_text)
        # sentences = [sent.text for sent in doc.sents]
        # response = []
        # entities = noun + verbs + time
        # print("====")
        # print(entities)
        # temp_list = []
        # for item in entities:
        #     for sent in sentences:
        #         if item in sent:
        #             temp_list.append(sent)
        # final_set = set(temp_list)
        # for sent in final_set:
        #     response.append(sent + '\n')
        """根据query指导提取实体内容"""
        if not entities:
            return ""
            
        prompt = f"""
        Please extract content for the following keywords from the new search results, guided by relevance to typical query contexts:
        Keywords: {entities}
        Context: {new_text}
        Query: {query}
        
        For each key word, extract diverse sentences from the context and format as: "key word: content about the keyword" and "content related to the query"
        Do not use your own words to decribe the entity contents.
        Put each keyword-content pair on a separate line.
        Focus on contextual information, relationships, and key facts.
        If no useful information for the query, skip this entity.
        """
        response = self.memory_manager(self.model_name, prompt)
        return response

    def memory_manager(self, model_name, prompt: str) -> str:
        if '1.5b' not in model_name:
            client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
            resp = client.chat.completions.create(
            model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ])
            response = resp.choices[0].message.content
    
            return response.strip()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = "./local"

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
    Answer the question based on the memory clues. {instruction}
    Memory clues: {memory}
    question: {query}
    Please **think step by step**!
    Generate an accurate answer based on the provided memory information and the relationships among the entites, events and time.
    """
    response = generator(model_name, prompt)
        
    return response

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