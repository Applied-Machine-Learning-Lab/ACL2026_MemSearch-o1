# run_search_o1.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse
from datasets import load_dataset
from datasets import load_from_disk
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from evaluate import (
    run_evaluation, 
    extract_answer
)
from openai import OpenAI

from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
from mem_api import generate_docs_to_reasonchain_batch, run_generation

from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_docs_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)

API_SECRET_KEY = "sk-zk22528c689c68abb04cbfaf2ec6373a09868bbdef5a073b"
BASE_URL = "https://api.zhizengzeng.com/v1"

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Name of the dataset to use."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=3,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=3,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_TURN = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    
    # Adjust parameters based on dataset
    if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wikimqa', 'medmcqa', 'pubhealth']:
        MAX_SEARCH_LIMIT = 3
        if dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wikimqa']:
            MAX_SEARCH_LIMIT = 3
            MAX_TURN = 3
        top_k = 10
        max_doc_len = 3000
    

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.0

    # Data paths based on dataset
    data_path = f'./longbench/data/{dataset_name}'

    print('-----------------------')
    print(f'Using {dataset_name} set.')
    print('-----------------------')
    
    # ---------------------- Model Loading ----------------------
    model_name = 'qwen2.5-3b-instruct'
    tokenizer = AutoTokenizer.from_pretrained('./deepseek-chat')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define output directory based on model and dataset
    out_path = f"longbench/mem_bge_route_mquery/{model_name}/{dataset_name}"
    os.makedirs(out_path, exist_ok=True)

    # ---------------------- Data Loading ----------------------
    data = load_dataset('json', data_files=f"longbench/data/{dataset_name}.jsonl", split='train')
    data.save_to_disk(f"longbench/data/{dataset_name}")
    data = [x for x in data]
    
    # ---------------------- Batch Generation Function----------------------
    # Already Imported

    # ---------------------- Preparation of Input Prompts ----------------------
    input_list = []
    for item in data:
        question = item['input']

        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wikimqa']:
            if dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            elif dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wikimqa']:
                instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
        user_prompt = get_task_instruction_openqa(question)
        instruction = ''

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)


    total_time = 3788.2
    temp_list = []
    pred_list = []
    mode1 = False

    if mode1:

        with open(f'./longbench/mem_bge_route_mquery/qwen2.5-3b-instruct/2wikimqa/train_new.json', 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        if '1.5b' in model_name:
            for item in data_list:
                temp_list.append(f"\\boxed{{{item}}}")
        else:
            for item in data_list:
                if "Output" in item:
                    temp_list.append(item["Output"])

        with open('./longbench/mem_bge_route_mquery/qwen2.5-3b-instruct/2wikimqa/output_best.json', 'w', encoding='utf-8') as f_out:
            json.dump(temp_list, f_out, ensure_ascii=False, indent=2)
    
        # run_evaluation(data, input_list, pred_list, dataset_name, out_path, total_time, split='train')
        print("Process completed.")
    else:
        with open(f'./longbench/mem_bge_route_mquery/qwen2.5-3b-instruct/2wikimqa/output_best.json', 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        for item in data_list:
            pred_list.append(item)
    
        run_evaluation(data, input_list, pred_list, dataset_name, out_path, total_time, split='train')
        print("Process completed.")

if __name__ == "__main__":
    main()
