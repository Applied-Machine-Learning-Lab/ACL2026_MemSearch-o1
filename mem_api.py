from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from evaluate import (
    run_evaluation, 
    extract_answer
)

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


API_SECRET_KEY = ""
BASE_URL = ""
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"

client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)


def run_generation(
    model_name,
    sequences: Dict[str, str],
    max_tokens: int = 256,
) -> str:

    prompt = sequences["prompt"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        if 'gpt' in model_name:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                frequency_penalty=0.05,
                stop=[END_SEARCH_QUERY],
                extra_body={
                    "repetition_penalty": 1.05
                }
            )
            generated_text = completion.choices[0].message.content.strip()

            # 保留并截断到 END_SEARCH_QUERY
            # if END_SEARCH_QUERY in generated_text:
            #     idx = generated_text.index(END_SEARCH_QUERY) + len(END_SEARCH_QUERY)
            #     generated_text = generated_text[:idx]
            if BEGIN_SEARCH_QUERY in generated_text:
                generated_text += END_SEARCH_QUERY
        elif '1.5b' in model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = "./local"

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            generated_text = response
            # 保留并截断到 END_SEARCH_QUERY
            if END_SEARCH_QUERY in response:
                idx = generated_text.index(END_SEARCH_QUERY) + len(END_SEARCH_QUERY)
                generated_text = response[:idx]
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                frequency_penalty=0.05,
                stop=[END_SEARCH_QUERY],
                extra_body={
                    "top_k": 20,
                    "repetition_penalty": 1.05
                }
            )
            generated_text = completion.choices[0].message.content.strip()

            # 保留并截断到 END_SEARCH_QUERY
            # if END_SEARCH_QUERY in generated_text:
            #     idx = generated_text.index(END_SEARCH_QUERY) + len(END_SEARCH_QUERY)
            #     generated_text = generated_text[:idx]
            if BEGIN_SEARCH_QUERY in generated_text:
                generated_text += END_SEARCH_QUERY

    except Exception as e:
        print(f"Error generating for prompt: {messages}")
        print(f"Error: {e}")
        generated_text = ''

    return generated_text

def generate_docs_to_reasonchain_batch(
    original_questions: List[str],
    prev_reasonings: List[str],
    search_queries: List[str],
    documents: List[str],
    dataset_name: str,
    max_tokens: int = 32768,
    coherent: bool = False,
) -> List[str]:
    """
    使用 API 批量生成从网页内容到推理链的转换，等价于本地 vLLM 版本
    """
    # Step 1: 构造 user prompt
    user_prompts = get_docs_to_reasonchain_instruction(prev_reasonings[0], search_queries[0], documents[0])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./deepseek-chat")
    prompts = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompts}],
            tokenize=False,
            add_generation_prompt=True)

    use_raw_prompt = True

    raw_outputs = []
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompts}
    ]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1.05,
        }
    )
    generated_text = completion.choices[0].message.content.strip()

    # Step 4: 提取信息
    extracted_infos = extract_answer(generated_text, mode='gen')

    # Step 5: 记录日志
    
    records = {
            'prompt': prompts,
            'raw_output': generated_text,
            'extracted_info': extracted_infos
    }

    return extracted_infos, records