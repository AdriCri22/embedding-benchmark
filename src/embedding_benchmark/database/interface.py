"""
Database interface for the embedding benchmark suite.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import sqlite3


class DatabaseInterface(ABC):
    """Interface for database operations."""
    
    @abstractmethod
    def build_index(
        self, 
        model_cfg: Dict[str, Any], 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build SQLite index with vector and FTS5 search."""
        pass
    
    @abstractmethod
    def vector_search(
        self, 
        conn: sqlite3.Connection, 
        q_emb: List[float], 
        k: int
    ) -> List[Tuple]:
        """Search using vector similarity (cosine distance)."""
        pass
    
    @abstractmethod
    def hybrid_search(
        self, 
        conn: sqlite3.Connection, 
        q_emb: List[float], 
        query_text: str, 
        k: int, 
        rrf_k: int = 60
    ) -> List[Tuple]:
        """Hybrid search combining vector similarity and FTS5 full-text search using RRF."""
        pass