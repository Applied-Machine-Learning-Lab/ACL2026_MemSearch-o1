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

API_SECRET_KEY = "Your API Key"
BASE_URL = "Your Base URL"

class MemoryPool:
    def __init__(self, model_name):
        self.memory_pool: str = ""  
        self.query_entities = ""
        self.key_entities: str = ""  
        self.history_documents: str = "" 
        self.model_name = model_name
        
    def add_document(self, document: str):
        if self.history_documents:
            self.history_documents += "|||" + document
        else:
            self.history_documents = document
    
    def process_new_text(self, new_text: str, query: str) -> str:
        
        entities = self._extract_key_entities_from_query(query)
    
        response1 = self._extract_entity_contents(new_text, entities, query)
        response2 = ''
        self.memory_pool = response1

        
        return self.memory_pool, response2
    
    def _extract_key_entities_from_query(self, query):
        doc = nlp(query)
        noun = [chunk.text for chunk in doc.noun_chunks]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        adverbs = [token.text for token in doc if token.pos_ == "ADV"]
        adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
        time = []
        for entity in doc.ents:
            if entity.label_ == 'DATE':
                time.append(entity.text)
        # Optional
        # out = noun + verbs + time + adverbs + adjectives
        out = noun + verbs + time
        out = '\n'.join([f"{r}," for i, r in enumerate(entities)])
        return out


    def _extract_entity_contents(self, text: str, entities: str, query: str) -> str:
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
    # Add degree modifiers if approriate
    prompt = f"""
    Answer the question based on the memory clues. {instruction}
    Memory clues: {memory}
    question: {query}
    Please **think step by step**!
    Generate an accurate answer based on the provided memory information and the relationships among the entites, events and time.
    """
    response = generator(model_name, prompt)
        
    return response