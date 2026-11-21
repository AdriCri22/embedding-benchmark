"""
Benchmark module for the embedding benchmark suite.
"""
from .interface import BenchmarkInterface

_benchmark: BenchmarkInterface = None


def get_benchmark() -> BenchmarkInterface:
    """Get the benchmark implementation singleton."""
    global _benchmark
    if _benchmark is None:
        from .internal.benchmark_impl import BenchmarkImpl
        _benchmark = BenchmarkImpl()
    return _benchmark


def to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_serializable(v) for v in obj)
    return obj


__all__ = ["BenchmarkInterface", "get_benchmark", "to_serializable"]