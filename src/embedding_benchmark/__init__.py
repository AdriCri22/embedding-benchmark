"""
Embedding Benchmark Suite - Modular Monolith Architecture

This package provides a comprehensive benchmark suite for evaluating
embedding models in RAG pipelines.

Architecture:
- Each module has an ABC interface and an internal implementation
- Modules communicate only through interfaces (Dependency Inversion)
- Factory functions provide singleton instances

Modules:
- config: Configuration management
- metrics: Metrics calculation
- data_loader: Data loading
- database: Database operations (SQLite + sqlite-vec + FTS5)
- embedding: Embedding generation (fastembed, OpenAI, llama.cpp)
- generation: LLM generation and judging
- benchmark: Benchmark execution
- benchmarks: Benchmark runners (quick, formats, full, mini)
"""

from .config import ConfigInterface, get_config
from .metrics import MetricsInterface, get_metrics
from .data_loader import DataLoaderInterface, get_data_loader
from .database import DatabaseInterface, get_database
from .embedding import EmbeddingInterface, get_embedding
from .generation import GenerationInterface, get_generation
from .benchmark import BenchmarkInterface, get_benchmark

from .benchmarks import (
    run_full_benchmark,
    run_quick_benchmark,
    run_formats_benchmark,
    run_mini_benchmark,
)

config = get_config()
ROOT = config.root
MODELS_DIR = config.models_dir
EMBEDDING_MODELS = config.embedding_models
GENERATION_MODEL = config.generation_model
JUDGE_MODEL = config.judge_model
LANG_CONFIG = config.lang_config

metrics = get_metrics()
get_ram_mb = metrics.get_ram_mb
escape_fts5 = metrics.escape_fts5
embeddings_to_bytes = metrics.embeddings_to_bytes
cosine_sim = metrics.cosine_sim
dcg_at_k = metrics.dcg_at_k
ndcg_at_k = metrics.ndcg_at_k
evaluate_recall = metrics.evaluate_recall
evaluate_mrr = metrics.evaluate_mrr
evaluate_ndcg = metrics.evaluate_ndcg
safe_str = metrics.safe_str

data_loader = get_data_loader()
load_chunks = data_loader.load_chunks

database = get_database()
build_index = database.build_index
vector_search = database.vector_search
hybrid_search = database.hybrid_search

embedding = get_embedding()
get_embedding_fn = embedding.get_embedding_fn
embed_fastembed = embedding.embed_fastembed
embed_openai = embedding.embed_openai
embed_llama_cpp = embedding.embed_llama_cpp

generation = get_generation()
generate_answer = generation.generate_answer
judge_answer = generation.judge_answer

benchmark = get_benchmark()
to_serializable = benchmark.to_serializable
start_llama_server = benchmark.start_llama_server
stop_llama_server = benchmark.stop_llama_server
run_single_thread_benchmark = benchmark.run_single_thread_benchmark
run_concurrent_benchmark = benchmark.run_concurrent_benchmark

__all__ = [
    "ConfigInterface", "MetricsInterface", "DataLoaderInterface",
    "DatabaseInterface", "EmbeddingInterface", "GenerationInterface", "BenchmarkInterface",
    "get_config", "get_metrics", "get_data_loader", "get_database",
    "get_embedding", "get_generation", "get_benchmark",
    "run_full_benchmark", "run_quick_benchmark", "run_formats_benchmark", "run_mini_benchmark",
    "ROOT", "MODELS_DIR", "EMBEDDING_MODELS", "GENERATION_MODEL", "JUDGE_MODEL", "LANG_CONFIG",
    "get_ram_mb", "escape_fts5", "embeddings_to_bytes", "cosine_sim",
    "dcg_at_k", "ndcg_at_k", "evaluate_recall", "evaluate_mrr", "evaluate_ndcg", "safe_str",
    "load_chunks",
    "build_index", "vector_search", "hybrid_search",
    "get_embedding_fn", "embed_fastembed", "embed_openai", "embed_llama_cpp",
    "generate_answer", "judge_answer",
    "to_serializable", "start_llama_server", "stop_llama_server",
    "run_single_thread_benchmark", "run_concurrent_benchmark",
]