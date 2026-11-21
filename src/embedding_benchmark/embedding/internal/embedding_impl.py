"""
Embedding implementation for the embedding benchmark suite.
"""
import os
import httpx
from typing import List, Callable, Dict, Any

from ..interface import EmbeddingInterface


# Physical batch size is 512, so with batch_size=8, max 64 tokens per text
MAX_TOKENS_PER_REQUEST = 64
CHARS_PER_TOKEN = 4


def truncate_text(text: str, max_tokens: int = MAX_TOKENS_PER_REQUEST) -> str:
    """Truncate text to fit within token limit."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


class EmbeddingImpl(EmbeddingInterface):
    """Implementation of embedding generation."""
    
    def get_embedding_fn(self, model_cfg: Dict[str, Any]) -> Callable[[List[str]], List[List[float]]]:
        """Return the appropriate embedding function for the model type."""
        if model_cfg["type"] == "fastembed":
            return lambda texts: self.embed_fastembed(texts, model_cfg["model_name"])
        elif model_cfg["type"] == "openai":
            return lambda texts: self.embed_openai(texts, model_cfg["model_name"])
        elif model_cfg["type"] == "llama_cpp":
            return lambda texts: self.embed_llama_cpp(texts, model_cfg["server_port"])
        else:
            raise ValueError(f"Unknown model type: {model_cfg['type']}")
    
    def embed_fastembed(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Generate embeddings using fastembed."""
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=model_name)
        return list(model.embed(texts))
    
    def embed_openai(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"input": batch, "model": model_name},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            embeddings.extend([d["embedding"] for d in sorted(data, key=lambda x: x["index"])])
        return embeddings
    
    def embed_llama_cpp(self, texts: List[str], port: int) -> List[List[float]]:
        """Generate embeddings using llama.cpp server."""
        truncated_texts = [truncate_text(text) for text in texts]
        print(f"DEBUG: Truncated {len(texts)} texts, first text len: {len(texts[0]) if texts else 0} -> {len(truncate_text(texts[0])) if texts else 0}")
        
        resp = httpx.post(
            f"http://localhost:{port}/v1/embeddings",
            json={"input": truncated_texts},
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]