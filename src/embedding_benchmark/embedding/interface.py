"""
Embedding interface for the embedding benchmark suite.
"""
from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Any


class EmbeddingInterface(ABC):
    """Interface for embedding generation."""
    
    @abstractmethod
    def get_embedding_fn(self, model_cfg: Dict[str, Any]) -> Callable[[List[str]], List[List[float]]]:
        """Return the appropriate embedding function for the model type."""
        pass
    
    @abstractmethod
    def embed_fastembed(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Generate embeddings using fastembed."""
        pass
    
    @abstractmethod
    def embed_openai(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        pass
    
    @abstractmethod
    def embed_llama_cpp(self, texts: List[str], port: int) -> List[List[float]]:
        """Generate embeddings using llama.cpp server."""
        pass