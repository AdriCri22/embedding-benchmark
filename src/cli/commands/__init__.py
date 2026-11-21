"""
CLI commands package.
"""
from .setup import setup
from .benchmark import benchmark
from .models import list_models, list_presets, download_models
from .test_setup import test_setup

__all__ = ["setup", "benchmark", "list_models", "list_presets", "download_models", "test_setup"]