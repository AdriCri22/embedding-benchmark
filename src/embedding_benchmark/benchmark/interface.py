"""
Benchmark interface for the embedding benchmark suite.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BenchmarkInterface(ABC):
    """Interface for benchmark execution."""
    
    @abstractmethod
    def build_index(
        self, 
        model_cfg: Dict[str, Any], 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build SQLite index with vector and FTS5 search."""
        pass
    
    @abstractmethod
    def run_single_thread_benchmark(
        self, 
        model_cfg: Dict[str, Any], 
        db_path: str, 
        chunks: List[Dict[str, Any]], 
        queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run single-threaded benchmark for a model."""
        pass
    
    @abstractmethod
    def run_concurrent_benchmark(
        self, 
        model_cfg: Dict[str, Any], 
        db_path: str, 
        num_workers: int, 
        queries_per_worker: int,
        queries: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Run concurrent benchmark with multiple workers."""
        pass
    
    @abstractmethod
    def run_benchmark(
        self, 
        model_cfg: Dict[str, Any], 
        db_path: str, 
        chunks: List[Dict[str, Any]], 
        queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run format-aware benchmark."""
        pass
    
    @abstractmethod
    def start_llama_server(self, model_cfg: Dict[str, Any]):
        """Start llama.cpp server for embedding or generation."""
        pass
    
    @abstractmethod
    def stop_llama_server(self, proc):
        """Stop llama.cpp server."""
        pass
    
    @abstractmethod
    def to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        pass

    @abstractmethod
    def estimate_api_cost(self, model_name: str, num_queries: int) -> float:
        """Estimate API cost for OpenAI models."""
        pass