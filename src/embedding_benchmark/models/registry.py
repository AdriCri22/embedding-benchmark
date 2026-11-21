"""
Model registry for loading and managing model configurations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
import yaml

from embedding_benchmark.utils import get_project_root


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    type: Literal["fastembed", "llama_cpp", "openai"]
    model_name: Optional[str] = None
    gguf_path: Optional[str] = None
    server_port: Optional[int] = None
    dim: int = 0
    ctx: int = 8192
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for benchmark consumption."""
        cfg = {
            "name": self.name,
            "type": self.type,
            "dim": self.dim,
            "tags": self.tags,
        }
        if self.type == "fastembed":
            cfg["model_name"] = self.model_name
        elif self.type == "llama_cpp":
            cfg["gguf_path"] = self.gguf_path
            cfg["server_port"] = self.server_port
            cfg["ctx"] = self.ctx
        elif self.type == "openai":
            cfg["model_name"] = self.model_name
        return cfg

    def to_llama_dict(self) -> Dict[str, Any]:
        """Convert to dict for llama.cpp server (generation/judge models)."""
        return {
            "name": self.name,
            "gguf_path": self.gguf_path,
            "server_port": self.server_port,
            "ctx": self.ctx
        }


class ModelRegistry:
    """Registry for managing model configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        self._models: Dict[str, ModelConfig] = {}
        self._presets: Dict[str, List[str]] = {}
        self._generation_model_name: str = ""
        self._judge_model_name: str = ""
        self._config_path = config_path or get_project_root() / "config" / "models.yaml"
        self._load()

    def _load(self):
        """Load models from YAML config."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self._config_path}")

        with open(self._config_path) as f:
            data = yaml.safe_load(f)

        for m in data.get("models", []):
            config = ModelConfig(
                name=m["name"],
                type=m["type"],
                model_name=m.get("model_name"),
                gguf_path=m.get("gguf_path"),
                server_port=m.get("server_port"),
                dim=m.get("dim", 0),
                ctx=m.get("ctx", 8192),
                tags=m.get("tags", []),
                enabled=m.get("enabled", True),
            )
            self._models[config.name] = config

        self._presets = data.get("presets", {})
        self._generation_model_name = data.get("generation_model", "llama-3.2-3b-instruct")
        self._judge_model_name = data.get("judge_model", "llama-3.2-3b-instruct")

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model by name."""
        return self._models.get(name)

    def get_all(self, enabled_only: bool = True) -> List[ModelConfig]:
        """Get all models."""
        models = list(self._models.values())
        if enabled_only:
            models = [m for m in models if m.enabled]
        return models

    def get_by_tags(self, tags: List[str], match_all: bool = False) -> List[ModelConfig]:
        """Get models matching tags."""
        models = self.get_all()
        if match_all:
            return [m for m in models if all(t in m.tags for t in tags)]
        return [m for m in models if any(t in m.tags for t in tags)]

    def get_preset(self, name: str) -> List[ModelConfig]:
        """Get models from a preset."""
        model_names = self._presets.get(name, [])
        return [self._models[n] for n in model_names if n in self._models]

    def list_presets(self) -> List[str]:
        """List available presets."""
        return list(self._presets.keys())

    def get_generation_model(self) -> Optional[ModelConfig]:
        """Get generation model config."""
        return self._models.get(self._generation_model_name)

    def get_judge_model(self) -> Optional[ModelConfig]:
        """Get judge model config."""
        return self._models.get(self._judge_model_name)

    def get_generation_model_name(self) -> str:
        """Get generation model name."""
        return self._generation_model_name

    def get_judge_model_name(self) -> str:
        """Get judge model name."""
        return self._judge_model_name


_registry: Optional[ModelRegistry] = None


def get_registry(config_path: Optional[Path] = None) -> ModelRegistry:
    """Get global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(config_path)
    return _registry