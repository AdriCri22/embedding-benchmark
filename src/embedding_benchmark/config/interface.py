"""
Configuration interface for the embedding benchmark suite.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List


class ConfigInterface(ABC):
    """Interface for configuration management."""
    
    @property
    @abstractmethod
    def root(self) -> Path:
        """Project root directory."""
        pass
    
    @property
    @abstractmethod
    def models_dir(self) -> Path:
        """Models directory."""
        pass
    
    @property
    @abstractmethod
    def embedding_models(self) -> List[Dict[str, Any]]:
        """List of embedding model configurations."""
        pass
    
    @property
    @abstractmethod
    def generation_model(self) -> Dict[str, Any]:
        """Generation model configuration."""
        pass
    
    @property
    @abstractmethod
    def judge_model(self) -> Dict[str, Any]:
        """Judge model configuration."""
        pass
    
    @property
    @abstractmethod
    def lang_config(self) -> Dict[str, Any]:
        """Language configuration."""
        pass
    
    @abstractmethod
    def get_test_data_dir(self, lang: str) -> Path:
        """Get test data directory for a language."""
        pass
    
    @abstractmethod
    def get_queries(self, lang: str) -> List[Dict[str, Any]]:
        """Get queries for a language."""
        pass
    
    @abstractmethod
    def get_format_queries(self, lang: str) -> List[Dict[str, Any]]:
        """Get format-aware queries for a language."""
        pass