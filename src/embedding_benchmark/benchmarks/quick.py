"""
Quick Embedding Benchmark - Backward compatibility wrapper.
Usage: python -m embedding_benchmark.benchmarks.quick [--lang en|es|ca]
"""
import argparse
from embedding_benchmark.benchmarks import run_quick_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Embedding Benchmark")
    parser.add_argument("--lang", choices=["en", "es", "ca"], default="en", help="Language (default: en)")
    args = parser.parse_args()
    run_quick_benchmark(args.lang)