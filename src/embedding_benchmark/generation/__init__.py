"""
Generation module for the embedding benchmark suite.
"""
from .interface import GenerationInterface

_generation: GenerationInterface = None


def get_generation() -> GenerationInterface:
    """Get the generation implementation singleton."""
    global _generation
    if _generation is None:
        from .internal.generation_impl import GenerationImpl
        _generation = GenerationImpl()
    return _generation


__all__ = ["GenerationInterface", "get_generation"]