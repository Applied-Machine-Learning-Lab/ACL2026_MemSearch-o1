def mem_search_o1_prompt():
     p = f"""Answer the given question. \
        Only give me the answer and do not output any other words. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as your want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\n"""
        

def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_docs_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant pasaages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant passages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_singleqa_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
    "你是一个具备搜索能力的推理助手，可以通过执行搜索来帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
    "- 要执行搜索：请写入 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
    "随后，系统将搜索并分析相关段落，并以如下格式为你提供有用的信息：<|begin_search_result|> ...搜索结果... <|end_search_result|>。\n\n"
    f"如有必要，你可以多次重复搜索过程，但搜索次数上限为 {MAX_SEARCH_LIMIT} 次。\n\n"
    "当你收集到所有必要信息后，请继续进行推理。\n\n"
    "示例：\n"
    "问题：“谁获得了第一届诺贝尔物理学奖？”\n"
    "助手思考步骤：\n"
    "- 我需要查出谁被授予了第一届诺贝尔物理学奖。\n\n"
    "助手：\n"
    "<|begin_search_query|>第一届诺贝尔物理学奖得主<|end_search_query|>\n\n"
    "（系统返回从相关段落中处理后的信息）\n\n"
    "助手根据新信息继续推理……\n\n"
    "请记住：\n"
    "- 使用 <|begin_search_query|> 发起搜索，并以 <|end_search_query|> 结束。\n"
    "- 搜索完成后，请继续你的推理。\n\n"
    )

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform searches to help "
        "you answer the user's question accurately. **When answering, just give the answer and do not output other information.**\n"
        "**You should provide your final answer in the format \\boxed{YOUR_ANSWER}.**\n"
        "You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant passages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"If you think the searched information is not enough, you can continue searching. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "**Once you have all the information you need, stop the search and continue your reasoning.**\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant passages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant passages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_multiqa_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备搜索能力的推理助手，可以通过执行搜索来帮助你准确回答用户的问题。**回答时只需给出答案，不要输出其他信息。**\n"
        "**请将最终答案以 \\boxed{YOUR_ANSWER} 的格式提供。**\n"
        "你拥有以下特殊工具：\n\n"
        "- 要执行搜索：请写入 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "随后，系统将搜索并分析相关段落，并以如下格式为你提供有用的信息：<|begin_search_result|> ...搜索结果... <|end_search_result|>。\n\n"
        f"如果你认为已搜索到的信息不够充分，可以继续搜索，但搜索次数上限为 {MAX_SEARCH_LIMIT} 次。\n\n"
        "**当你收集到所有必要信息后，请停止搜索并继续推理。**\n\n"
        "示例：\n"
        "问题：“Alice David 是哪一家公司开发的电子游戏中 Lara Croft 的配音演员？”\n"
        "助手思考步骤：\n"
        "- 我需要查出 Alice David 在哪个电子游戏中为 Lara Croft 配音。\n"
        "- 然后，我需要确定开发该游戏的公司是哪家。\n\n"
        "助手：\n"
        "<|begin_search_query|>Alice David Lara Croft 配音<|end_search_query|>\n\n"
        "（系统返回从相关段落中处理后的信息）\n\n"
        "助手思考：搜索结果表明，Alice David 在某款特定电子游戏中为 Lara Croft 配音。现在我需要查出是哪家公司开发了该游戏。\n\n"
        "助手：\n"
        "<|begin_search_query|>Alice David Lara Croft 配音 游戏开发商<|end_search_query|>\n\n"
        "（系统返回从相关段落中处理后的信息）\n\n"
        "助手根据新信息继续推理……\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起搜索，并以 <|end_search_query|> 结束。\n"
        "- 搜索完成后，请继续你的推理。\n\n"
    )

    
def get_singleqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention several URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/first_nobel_physics.html<|end_url|>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_multiqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the search tool with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of retrieved passages (Docs).\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>voice actor of Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results provide names of voice actors for Lara Croft. I need to confirm if Alice David is one of them.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a search and end with <|end_search_query|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_gpqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/ppIII_neutrino.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_math_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s math-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/integration_e_x_squared.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_code_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s programming-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do I implement a binary search algorithm in Python?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up the implementation details of binary search in Python.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>binary search algorithm implementation in Python<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/python_binary_search.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )



def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_openqa_zh(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '请回答以下问题。请直接给出最终答案，不要输出其他内容。\n'
            '请将最终答案以 \\boxed{答案} 的格式输出。\n\n'
            f'问题：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '请逐步思考并解答以下问题。\n\n'
            '请将最终答案以 \\boxed{答案} 的格式输出。\n\n'
            f'问题：\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please choose the correct answer to the following question. You should think step by step to solve it.\n\n'
            'Format your response as follows: "The correct answer is (insert answer here)".\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt

