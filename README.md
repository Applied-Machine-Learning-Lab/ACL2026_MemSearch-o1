# MemSearch-o1
In this paper, we propose MemSearch-o1, an agentic search framework built on reasoning-aligned memory growth and retracing. MemSearch-o1 dynamically grows fine-grained memory fragments from memory seed tokens from the queries, then retraces and deeply refines the memory via a contribution function, and finally reorganizes a globally connected memory path. This shifts memory management from stream-like concatenation to structured, token-level growth with path-based reasoning. Experiments on eight benchmark datasets show that MemSearch-o1 substantially mitigates memory dilution, and more effectively activates the reasoning potential of diverse LLMs, establishing a solid foundation for memory-aware agentic intelligence.

This is the main implementation code of MemSearch-o1.

## Installation

1. Install Python dependencies (CUDA-enabled `torch` is recommended; CPU also works but the embedding retriever will be slow on long contexts):
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Set your API Key and Base URL in `mem_api.py`:
```
API_SECRET_KEY = "Your API Key"
BASE_URL = "Your Base URL"
```

3. Prepare a local embedding model for `utils/__init__.py`. By default it reads from the relative path `./retrieve_model`; you can override it via the `RETRIEVE_MODEL_PATH` environment variable. (e.g. `BAAI/bge-small-en-v1.5`).

4. Download the LongBench JSONL files for the dataset(s) you want to run, and place them under `longbench/data/<dataset_name>.jsonl`. The dataset can be downloaded from https://huggingface.co/datasets/THUDM/LongBench

## Quick Start

For quick start, you can directly run the following code:
```
python mem_o1_adv.py --dataset_name hotpotqa
```
You can also change `top_k`, `max_search_limit`, `max_turn`, and other configs for more tests.
