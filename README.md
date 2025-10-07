# MemSearch-o1
Recent advances in large language models (LLMs) have scaled the potential for reasoning and agentic search, wherein models autonomously plan, retrieve, and reason over external knowledge to answer complex queries. However, the iterative think–search loop accumulates long system memories, leading to memory dilution problem. In addition, existing memory management methods struggle to capture fine-grained semantic relations between queries and documents and often lose substantial information. Therefore, we propose MemSearch-o1, an agentic search framework built on reasoning-aligned memory growth and retracing. MemSearch-o1 dynamically grows fine-grained memory fragments from memory seed tokens from the queries, then retraces and deeply refines the memory via a contribution function, and finally reorganizes a globally connected memory path. This shifts memory management from stream-like concatenation to structured, token-level growth with path-based reasoning. Experiments on eight benchmark datasets show that MemSearch-o1 substantially mitigates memory dilution, and more effectively activates the reasoning potential of diverse LLMs, establishing a solid foundation for memory-aware agentic intelligence.
This is the main implementation code of MemSearch-o1.
First, you should set your API Key and Base URL in `memory_o1_adv.py':
```
API_SECRET_KEY = "Your API Key"
BASE_URL = "Your Base URL"
```
For quick start, you can diretly run the following code:
```
python scripts/mem_o1_adv.py --dataset_name hotpotqa
```
You can also change the `top_k` and other configs for more tests.
