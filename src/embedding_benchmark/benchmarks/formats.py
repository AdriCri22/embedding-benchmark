"""
Format-Aware Embedding Benchmark - Backward compatibility wrapper.
Usage: python -m embedding_benchmark.benchmarks.formats [--lang en|es|ca]
"""
import argparse
from embedding_benchmark.benchmarks import run_formats_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format-Aware Embedding Benchmark")
    parser.add_argument("--lang", choices=["en", "es", "ca"], default="en", help="Language for test data (default: en)")
    args = parser.parse_args()
    run_formats_benchmark(args.lang)