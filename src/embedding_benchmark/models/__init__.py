"""
Models module for the embedding benchmark suite.
"""
from .registry import ModelConfig, ModelRegistry, get_registry

__all__ = ["ModelConfig", "ModelRegistry", "get_registry"]