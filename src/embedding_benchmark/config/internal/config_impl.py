"""
Configuration implementation for the embedding benchmark suite.
"""
from pathlib import Path
from typing import Dict, Any, List
import yaml

from ..interface import ConfigInterface
from embedding_benchmark.models.registry import get_registry
from embedding_benchmark.utils import get_project_root, get_models_dir


class ConfigImpl(ConfigInterface):
    """Implementation of configuration management."""

    def __init__(self):
        self._root = get_project_root()
        self._models_dir = get_models_dir()
        self._embedding_models = self._load_embedding_models()
        self._generation_model = self._load_generation_model()
        self._judge_model = self._load_judge_model()
        self._lang_config = self._load_lang_config()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    @property
    def embedding_models(self) -> List[Dict[str, Any]]:
        return self._embedding_models

    @property
    def generation_model(self) -> Dict[str, Any]:
        return self._generation_model

    @property
    def judge_model(self) -> Dict[str, Any]:
        return self._judge_model

    @property
    def lang_config(self) -> Dict[str, Any]:
        return self._lang_config

    def get_test_data_dir(self, lang: str) -> Path:
        return self._root / self._lang_config[lang]["test_data_dir"]

    def get_queries(self, lang: str) -> List[Dict[str, Any]]:
        return self._lang_config[lang]["queries"]

    def get_format_queries(self, lang: str) -> List[Dict[str, Any]]:
        return self._lang_config[lang]["format_queries"]

    def _load_embedding_models(self) -> List[Dict[str, Any]]:
        """Load embedding models from ModelRegistry (models.yaml)."""
        registry = get_registry()
        models = registry.get_all(enabled_only=True)
        result = []
        for m in models:
            if "generation" in m.tags or "judge" in m.tags:
                continue
            result.append(m.to_dict())
        return result

    def _load_generation_model(self) -> Dict[str, Any]:
        registry = get_registry()
        model = registry.get_generation_model()
        if model:
            return model.to_llama_dict()
        return {
            "name": "llama-3.2-3b-instruct",
            "gguf_path": str(self._models_dir / "llama-3.2-3b-instruct-q8_0.gguf"),
            "server_port": 8083,
            "ctx": 4096
        }

    def _load_judge_model(self) -> Dict[str, Any]:
        registry = get_registry()
        model = registry.get_judge_model()
        if model:
            return model.to_llama_dict()
        return {
            "name": "llama-3.2-3b-instruct",
            "gguf_path": str(self._models_dir / "llama-3.2-3b-instruct-q8_0.gguf"),
            "server_port": 8084,
            "ctx": 4096
        }

    def _load_lang_config(self) -> Dict[str, Any]:
        config_dir = self._root / "config" / "queries"
        langs = {}
        for lang in ["en", "es", "ca"]:
            data = self._load_yaml(config_dir / f"{lang}.yaml")
            langs[lang] = {
                "test_data_dir": data.get("test_data_dir", f"test_data/{lang}"),
                "queries": data.get("queries", []),
                "format_queries": data.get("format_queries", [])
            }
        return langs

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}