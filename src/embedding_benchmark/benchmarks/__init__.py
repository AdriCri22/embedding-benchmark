"""
Benchmark implementations package.
"""
from typing import List
from .runner import run_benchmark

# Backward compatibility wrappers
def run_full_benchmark(lang: str = "en") -> List:
    return run_benchmark(lang=lang, mode="full")

def run_quick_benchmark(lang: str = "en") -> List:
    return run_benchmark(lang=lang, mode="quick")

def run_formats_benchmark(lang: str = "en") -> List:
    return run_benchmark(lang=lang, mode="formats")

def run_mini_benchmark(lang: str = "en") -> List:
    return run_benchmark(lang=lang, mode="mini")

__all__ = ["run_benchmark", "run_full_benchmark", "run_quick_benchmark", "run_formats_benchmark", "run_mini_benchmark"]