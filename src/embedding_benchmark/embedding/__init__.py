"""
Embedding module for the embedding benchmark suite.
"""
from .interface import EmbeddingInterface

_embedding: EmbeddingInterface = None


def get_embedding() -> EmbeddingInterface:
    """Get the embedding implementation singleton."""
    global _embedding
    if _embedding is None:
        from .internal.embedding_impl import EmbeddingImpl
        _embedding = EmbeddingImpl()
    return _embedding


__all__ = ["EmbeddingInterface", "get_embedding"]