"""
Metrics interface for the embedding benchmark suite.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import sqlite3


class MetricsInterface(ABC):
    """Interface for metrics calculation."""
    
    @abstractmethod
    def get_ram_mb(self) -> float:
        """Get current process RAM usage in MB."""
        pass
    
    @abstractmethod
    def escape_fts5(self, query: str) -> str:
        """Escape query for FTS5 full-text search."""
        pass
    
    @abstractmethod
    def embeddings_to_bytes(self, embeddings: List[float]) -> bytes:
        """Convert embeddings list to bytes for SQLite storage."""
        pass
    
    @abstractmethod
    def cosine_sim(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        pass
    
    @abstractmethod
    def dcg_at_k(self, relevances: List[float], k: int) -> float:
        """Discounted Cumulative Gain at k."""
        pass
    
    @abstractmethod
    def ndcg_at_k(self, relevances: List[float], k: int) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        pass
    
    @abstractmethod
    def evaluate_recall(
        self, 
        found_ids: List[int], 
        expected_file: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate recall@k: whether expected file is in top-k results."""
        pass
    
    @abstractmethod
    def evaluate_mrr(
        self, 
        found_ids: List[int], 
        expected_file: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate Mean Reciprocal Rank at k."""
        pass
    
    @abstractmethod
    def evaluate_ndcg(
        self, 
        found_ids: List[int], 
        expected_file: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate nDCG at k with graded relevance."""
        pass
    
    @abstractmethod
    def evaluate_format_recall(
        self, 
        found_ids: List[int], 
        expected_format: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate recall@k for a specific format."""
        pass
    
    @abstractmethod
    def safe_str(self, value: Any) -> str:
        """Convert any value to string safely."""
        pass