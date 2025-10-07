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
from utils import retrieve, route

from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
from mem_api import generate_docs_to_reasonchain_batch, run_generation

from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction,
    get_singleqa_search_o1_instruction_zh, 
    get_multiqa_search_o1_instruction_zh,
    get_docs_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_openqa_zh,
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)

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
        choices=['hotpotqa', '2wikimqa', 'musique', 'narrativeqa', 'qasper', 'multifieldqa_en','dureader', 'multifieldqa_zh'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        default='deepseek-chat',
        help="Name of the model to use."
    )

    parser.add_argument(
        '--worker',
        type=str,
        required=True,
        default='line1',
        help="Code worker."
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
    model_name = args.model_name
    worker = args.worker

    API_SECRET_KEY = "Your API Key"
    BASE_URL = "Your Base URL"
    
    # Adjust parameters based on dataset
    if dataset_name in ['hotpotqa', 'musique', '2wikimqa', 'narrativeqa', 'qasper', 'multifieldqa_en', 'dureader', 'multifieldqa_zh']:
        MAX_SEARCH_LIMIT = 5
        if dataset_name in ['hotpotqa', 'musique', '2wikimqa', 'narrativeqa', 'qasper', 'multifieldqa_en', 'dureader', 'multifieldqa_zh']:
            MAX_SEARCH_LIMIT = 5
            MAX_TURN = 5

    if dataset_name in ['dureader', 'multifieldqa_zh']:
        from memory_adv_zh import MemoryPool
    else:
        from memory_adv import MemoryPool

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.0

    # Data paths based on dataset
    data_path = f'./your_dataset_path/{dataset_name}'

    print('-----------------------')
    print(f'Using {dataset_name} set.')
    print('-----------------------')
    
    # ---------------------- Model Loading ----------------------
    tokenizer = AutoTokenizer.from_pretrained('your tokenizer')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define output directory based on model and dataset
    out_path = f"your output path/{model_name}/{dataset_name}"
    os.makedirs(out_path, exist_ok=True)

    # ---------------------- Data Loading ----------------------
    data = load_dataset('json', data_files=f"longbench/data/{dataset_name}.jsonl", split='train')
    data.save_to_disk(f"longbench/data/{dataset_name}")
    data = [x for x in data]

    # ---------------------- Preparation of Input Prompts ----------------------
    input_list = []
    for item in data:
        question = item['input']

        if dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wikimqa', 'musique', 'narrativeqa', 'qasper', 'multifieldqa_en', 'dureader', 'multifieldqa_zh' ]:
            if dataset_name in ['nq', 'triviaqa', 'narrativeqa', 'qasper', 'multifieldqa_en']:
                instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                user_prompt = get_task_instruction_openqa(question)
            elif dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wikimqa']:
                instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                user_prompt = get_task_instruction_openqa(question)
            elif dataset_name in ['dureader']:
                instruction = get_multiqa_search_o1_instruction_zh(MAX_SEARCH_LIMIT)
                user_prompt = get_task_instruction_openqa_zh(question)
            elif dataset_name in ['multifieldqa_zh']:
                instruction = get_singleqa_search_o1_instruction_zh(MAX_SEARCH_LIMIT)
                user_prompt = get_task_instruction_openqa_zh(question)

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    # Initialize active sequences
    active_sequences = [{
        'item': item,
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
    } for item, prompt in zip(data, input_list)]

    # Function to extract text between two tags
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def replace_recent_steps(origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # Remove the step if it exists
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # Replace or add the step
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        return new_reasoning_steps
    
    # ---------------------- Final Gneration ------------------------------
    def generator(model_name, prompt: str) -> str:
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

        
    def generate_response(model_name, memory, query):
        if dataset_name in ['dureader', 'multifieldqa_zh']:
            prompt = ("请根据提供的上下文回答问题。只需给出答案，不要输出其他任何信息。\n"
                      "请将最终答案以 \\boxed{你的答案} 的格式提供。\n\n"
                      f"上下文: {memory}\n"
                      f"问题: {query}\n"
                      "请基于提供的上下文信息，生成答案。\n")
        else:
            prompt = ("Answer the question based on the given context. Just give the answer and do not output other information.\n"
            "You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
            "If the answer is in the context, maintain the illustrations in the context to give the answer.\n\n"
            f"Context: {memory}\n"
            f"question: {query}\n"
            "Generate an accurate answer based on the provided information.\n")
    
        try:
            response = generator(model_name, prompt)
        except Exception as e:
            print(f"Error: {e}")
            response = ''
        
        return response

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []
    output_list = []
    count_list = []
    pred_list = []

    start_time = time.time()
    search_cache = {}
    # Main loop until all sequences are finished or maximum turns reached
    
    sequences_list = [seq for seq in active_sequences if not seq['finished']]
    for sequences_needing_generation in tqdm(sequences_list, desc = "Executing Search o1..."):
        temp = [seq for seq in active_sequences if not seq['finished']]
        turn = 0
        compressed_memory = ''
        memory_adv = ''
        search_query_list = []
        while True:
            # Identify sequences that need generation
            
            turn += 1
            memory_pool = MemoryPool(model_name)
            if sequences_needing_generation:
                print(f"We have {len(temp)} sequences needing generation...")
                print(f" ================= Turn {turn} =================")
                outputs = run_generation(model_name, sequences_needing_generation, max_tokens=512)
                print(outputs)
                print("Generation completed, processing outputs...")

                # Initialize batch variables

                # Process each sequence and collect URLs
                seq = sequences_needing_generation
                text = outputs
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text
                
                context = seq['item']['context']
                tokenized_context = tokenizer(context, truncation=False, return_tensors="pt").input_ids[0]

                parts = []
                for i in range(0, len(tokenized_context), 256):
                    parts.append(tokenized_context[i:i+256])
                parts = [tokenizer.decode(part) for part in parts]
                # retriever = BM25Retriever.from_documents([Document(page_content=part) for part in parts], k=3)
                
                question = seq['item']['input']
                question = question.strip()
                if question[-1] != '?':
                    question += '?'

                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                # If a search query is present and needs to be executed
                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                    search_query_list.append(search_query)
                    if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                        # Execute search, use cache if available
                        if search_query in search_cache:
                            cache = search_cache[search_query]
                            print(f"Using cached search results for query: \"{search_query}\"")
                        else:
                            try:
                                result = retrieve(parts, search_query, top_k)
                                if top_k > 1:
                                    result = "".join([f"{r['text']}\n" for i, r in enumerate(result)])
                                else:
                                    result = result[0]['text']
                                memory, memory2 = memory_pool.process_new_text(result, search_query)
                                compressed_memory += memory
                                memory_adv += memory2
                                search_cache[search_query] = result
                                print(f"Executed and cached search for query: \"{search_query}\"")
                                # print(f"Search Result: {result}")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                search_cache[search_query] = ''
                                result = ''

                        # Extract relevant information from Bing search results
                        relevant_info = result
                        seq['relevant_info'] = result

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)

                    elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Repeated search for query: \"{search_query}\"")

                else:
                    # If no search query needs to be executed, mark the sequence as finished
                    seq['finished'] = True
                    print("Sequence marked as complete.")
            
            if seq['finished']:
                break
            if turn >= MAX_TURN:
                print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                break
            
            batch_output_records.append(compressed_memory)
                
            if isinstance(result, str):
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}{compressed_memory}{END_SEARCH_RESULT}\n\n"
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)
            else:
                append_text = replace_recent_steps(seq['output'], analysis)
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)

        mem_instance = compressed_memory.split('\n')
        print(compressed_memory)
        if len(mem_instance) > 30:
            source = route(mem_instance, sequences_needing_generation['item']['input'], search_query_list) + memory_adv
        else:
            source = compressed_memory + memory_adv
        if len(source) < 10:
            # This case seldom occurs.
            parts = []
            for i in range(0, len(tokenized_context), 256):
                parts.append(tokenized_context[i:i+256])
            parts = [tokenizer.decode(part) for part in parts]
            result = retrieve(parts, sequences_needing_generation['item']['input'], top_k=3)
            result = "".join([f"{r['text']}\n" for i, r in enumerate(result)])
            source = result
        else:
            source = compressed_memory
        pred = generate_response(model_name, source, sequences_needing_generation['item']['input'])
                        
        count_list.append(seq['search_count'])
        output_list.append(seq['output'])
        pred_list.append(pred)
        

    total_time = time.time() - start_time

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # Define output JSON file path
    t = time.localtime()
    c = []
    c.append(sum(count_list)/len(count_list))
    count_file = os.path.join(out_path, 'count.json')
    batch_output_file = os.path.join(out_path, 'info.json')
    pred_file = os.path.join(out_path, f'{dataset_name}.json')
    final_pred_file = os.path.join(out_path, f'final_pred_{dataset_name}.json')

    with open(count_file, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2)

    # Save batch_output_records to JSON file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False)

    print(f"Batch outputs saved to {batch_output_file}")

    # Prepare output list for evaluation
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)
        # for output in output_list:
        #     f.write(json.dumps(output, ensure_ascii=False) + "\n")
            
    with open(final_pred_file, "w", encoding="utf-8") as f:
        json.dump(pred_list, f, ensure_ascii=False, indent=2)
        # for pred in pred_list:
        #     f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    run_evaluation(data, input_list, pred_list, dataset_name, out_path, total_time, split='train')
    print("Process completed.")
    print(f"Average search count: {sum(count_list)/len(count_list):.4f}")

if __name__ == "__main__":
    main()
