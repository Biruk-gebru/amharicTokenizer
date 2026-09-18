"""Record the exact commit sha of every HF tokenizer used.

HF repos are mutable: `Qwen/Qwen3-8B` today is not necessarily the same tokenizer in
six months, and the committed notebook outputs would silently stop matching. This pins
each repo to the sha that produced the results in the repo.
"""
import json
from pathlib import Path
from huggingface_hub import HfApi

REPOS = [
 "moonshotai/Kimi-K3","openai/gpt-oss-20b","Qwen/Qwen3-8B","deepseek-ai/DeepSeek-V3",
 "allenai/OLMo-2-1124-7B","mistralai/Mistral-7B-Instruct-v0.3","mistralai/Mistral-Nemo-Base-2407",
 "HuggingFaceTB/SmolLM2-1.7B","EleutherAI/gpt-neox-20b","gpt2","meta-llama/Llama-3.1-8B",
 "google/gemma-2-9b","bigscience/bloom-560m","google-bert/bert-base-multilingual-cased",
 "xlm-roberta-base","facebook/nllb-200-distilled-600M","CohereForAI/aya-101",
 "CohereLabs/aya-expanse-8b","Davlan/afro-xlmr-large","rasyosef/roberta-base-amharic",
 "abdukuzi45/qwen3.5-4b-amharic-sft",
]
api = HfApi(); out = {}
for r in REPOS:
    try:
        out[r] = api.model_info(r).sha
        print(f"  {r:42} {out[r][:12]}")
    except Exception as e:
        print(f"  {r:42} SKIP {type(e).__name__}")
Path("data/revisions.json").write_text(json.dumps(out, indent=2, sort_keys=True))
print(f"\npinned {len(out)} repos -> data/revisions.json")
