import os

LLM_INPUT = "../data/corpus.jsonl"
LLM_OUT: str | None = "./corpus.llm.jsonl"

# ID del modello su OpenRouter
#   qwen/qwen3-235b-a22b-2507 - Qwen3, non-thinking, 262K ctx
#   meta-llama/llama-4-maverick - Llama 4 MoE, 1M ctx, multimodale
LLM_MODEL = "qwen/qwen3-235b-a22b-2507"

OPENROUTER_API_KEY: str | None = (
    ""
)

LLM_MAX_TOKENS = 2048
LLM_LIMIT = 0  # 0 = tutto il file
LLM_WORKERS = 8
