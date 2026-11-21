"""
Hugging Face repository mappings for GGUF models.
Single source of truth for model -> HF repo mapping.
"""
from typing import Optional


HF_REPO_MAP = {
    "embeddinggemma-2-Q8_0": "ggml-org/embeddinggemma-2-GGUF",
    "Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B-GGUF",
    "gte-qwen2-1.5b-instruct": "mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF",
    "gte-qwen2-7b-instruct": "fishbone64/gte-Qwen2-7B-instruct-Q8_0-GGUF",
    "llama-3.2-3b-instruct": "bartowski/Llama-3.2-3B-Instruct-GGUF",
}


def get_hf_repo(model_name: str) -> Optional[str]:
    """Map model name to HF repo."""
    return HF_REPO_MAP.get(model_name)


def get_hf_repo_for_download(model_name: str) -> Optional[str]:
    """Get HF repo with include pattern for Q8_0 downloads."""
    repo = HF_REPO_MAP.get(model_name)
    if repo:
        return f"{repo} *Q8_0*"
    return None