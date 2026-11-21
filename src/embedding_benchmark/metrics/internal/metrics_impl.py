"""
Metrics implementation for the embedding benchmark suite.
"""
import math
import struct
import os
import resource
import json
from typing import List, Dict, Any
import sqlite3

try:
    import psutil
except ImportError:
    psutil = None

from ..interface import MetricsInterface


class MetricsImpl(MetricsInterface):
    """Implementation of metrics calculation."""
    
    def get_ram_mb(self) -> float:
        """Get current process RAM usage in MB."""
        try:
            if psutil:
                return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except:
            pass
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except:
            return 0.0
    
    def escape_fts5(self, query: str) -> str:
        """Escape query for FTS5 full-text search."""
        escaped = query.replace('"', '""')
        return f'"{escaped}"'
    
    def embeddings_to_bytes(self, embeddings: List[float]) -> bytes:
        """Convert embeddings list to bytes for SQLite storage."""
        return struct.pack(f"<{len(embeddings)}f", *embeddings)
    
    def cosine_sim(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    
    def dcg_at_k(self, relevances: List[float], k: int) -> float:
        """Discounted Cumulative Gain at k."""
        relevances = relevances[:k]
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    
    def ndcg_at_k(self, relevances: List[float], k: int) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        dcg = self.dcg_at_k(relevances, k)
        ideal = self.dcg_at_k(sorted(relevances, reverse=True), k)
        return dcg / ideal if ideal > 0 else 0.0
    
    def evaluate_recall(
        self, 
        found_ids: List[int], 
        expected_file: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate recall@k: whether expected file is in top-k results."""
        if not found_ids:
            return 0.0
        placeholders = ",".join("?" * min(k, len(found_ids)))
        rows = conn.execute(
            f"SELECT path FROM chunks WHERE rowid IN ({placeholders})", 
            found_ids[:k]
        ).fetchall()
        return 1.0 if any(expected_file in r[0] for r in rows) else 0.0
    
    def evaluate_mrr(
        self, 
        found_ids: List[int], 
        expected_file: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate Mean Reciprocal Rank at k."""
        if not found_ids:
            return 0.0
        placeholders = ",".join("?" * min(k, len(found_ids)))
        rows = conn.execute(
            f"SELECT path FROM chunks WHERE rowid IN ({placeholders})", 
            found_ids[:k]
        ).fetchall()
        for rank, (path,) in enumerate(rows, 1):
            if expected_file in path:
                return 1.0 / rank
        return 0.0
    
    def evaluate_ndcg(
        self, 
        found_ids: List[int], 
        expected_file: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate nDCG at k with graded relevance."""
        if not found_ids:
            return 0.0
        placeholders = ",".join("?" * min(k, len(found_ids)))
        rows = conn.execute(
            f"SELECT path FROM chunks WHERE rowid IN ({placeholders})", 
            found_ids[:k]
        ).fetchall()
        relevances = []
        for (path,) in rows:
            if expected_file in path:
                relevances.append(3.0)
            elif any(part in path for part in expected_file.split("/")[:-1]):
                relevances.append(2.0)
            elif expected_file.split("/")[0] in path:
                relevances.append(1.0)
            else:
                relevances.append(0.0)
        return self.ndcg_at_k(relevances, k)
    
    def evaluate_format_recall(
        self, 
        found_ids: List[int], 
        expected_format: str, 
        k: int, 
        conn: sqlite3.Connection
    ) -> float:
        """Evaluate recall@k for a specific format."""
        if not found_ids:
            return 0.0
        placeholders = ",".join("?" * min(k, len(found_ids)))
        rows = conn.execute(
            f"SELECT format FROM chunks WHERE rowid IN ({placeholders})", 
            found_ids[:k]
        ).fetchall()
        return 1.0 if any(expected_format == r[0] for r in rows) else 0.0
    
    def safe_str(self, value: Any) -> str:
        """Convert any value to string safely."""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return " ".join(self.safe_str(v) for v in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)