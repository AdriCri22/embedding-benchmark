"""
Benchmark implementation for the embedding benchmark suite.
"""
import time
import json
import subprocess
import threading
import queue
import statistics
import os
import signal
from pathlib import Path
from typing import List, Dict, Any, Tuple

from ..interface import BenchmarkInterface
from embedding_benchmark.embedding import get_embedding
from embedding_benchmark.database import get_database
from embedding_benchmark.metrics import get_metrics
from embedding_benchmark.utils import create_sqlite_connection, percentile, sqlite_connection, safe_mean


# OpenAI API pricing per 1M tokens (approximate)
OPENAI_PRICING_PER_1M_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

# Estimated tokens per query for cost estimation
TOKENS_PER_QUERY = 500


class BenchmarkImpl(BenchmarkInterface):
    """Implementation of benchmark execution."""
    
    def __init__(self):
        self._embedding = get_embedding()
        self._database = get_database()
        self._metrics = get_metrics()
    
    def build_index(
        self, 
        model_cfg: Dict[str, Any], 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build SQLite index with vector and FTS5 search."""
        return self._database.build_index(model_cfg, chunks)
    
    def _process_queries(
        self,
        model_cfg: Dict[str, Any],
        db_path: str,
        queries: List[Dict[str, Any]],
        embed_fn,
        evaluate_fn,
    ) -> Dict[str, Any]:
        """Process queries with cold start tracking and latency measurement."""
        latencies = []
        cold_start_done = False
        cold_start_ms = 0
        results = []

        with sqlite_connection(db_path) as conn:
            for q in queries:
                query = q["query"]
                expected_file = q.get("expected_file")
                expected_answer = q.get("expected_answer", "")
                expected_format = q.get("expected_format")
                expected_path_contains = q.get("expected_path_contains")
                category = q.get("category")

                t0 = time.time()
                q_emb = embed_fn([query])[0]
                if not cold_start_done:
                    cold_start_ms = (time.time() - t0) * 1000
                    cold_start_done = True

                hybrid_rows = self._database.hybrid_search(conn, q_emb, query, 10)
                hybrid_ids = [r[0] for r in hybrid_rows]

                latency_ms = (time.time() - t0) * 1000
                latencies.append(latency_ms)

                result = evaluate_fn(hybrid_ids, q, conn)
                if result is not None:
                    results.append(result)

        return {
            "latencies": latencies,
            "cold_start_ms": cold_start_ms,
            "results": results,
        }
    
    def run_single_thread_benchmark(
        self, 
        model_cfg: Dict[str, Any], 
        db_path: str, 
        chunks: List[Dict[str, Any]], 
        queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run single-threaded benchmark for a model."""
        embed_fn = self._embedding.get_embedding_fn(model_cfg)
        
        def evaluate(hybrid_ids, q, conn):
            expected_file = q["expected_file"]
            return {
                "recall": self._metrics.evaluate_recall(hybrid_ids, expected_file, 5, conn),
                "mrr": self._metrics.evaluate_mrr(hybrid_ids, expected_file, 10, conn),
                "ndcg": self._metrics.evaluate_ndcg(hybrid_ids, expected_file, 5, conn),
            }
        
        result = self._process_queries(model_cfg, db_path, queries, embed_fn, evaluate)
        
        recalls = [r["recall"] for r in result["results"]]
        mrrs = [r["mrr"] for r in result["results"]]
        ndcgs = [r["ndcg"] for r in result["results"]]
        
        return {
            "latencies": result["latencies"],
            "cold_start_ms": result["cold_start_ms"],
            "recalls_hybrid": recalls,
            "mrr_hybrid": mrrs,
            "ndcg5_hybrid": ndcgs,
        }
    
    def run_concurrent_benchmark(
        self, 
        model_cfg: Dict[str, Any], 
        db_path: str, 
        num_workers: int, 
        queries_per_worker: int,
        queries: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Run concurrent benchmark with multiple workers."""
        all_queries = [q["query"] for q in queries]
        queries_list = (all_queries * ((queries_per_worker // len(all_queries)) + 1))[:queries_per_worker]
        worker_queries = [queries_list[i::num_workers] for i in range(num_workers)]
        results_queue = queue.Queue()
        threads = []
        start = time.time()
        for wq in worker_queries:
            t = threading.Thread(target=self._concurrent_worker, args=(model_cfg, db_path, wq, results_queue))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start
        all_latencies = []
        while not results_queue.empty():
            all_latencies.extend(results_queue.get())
        total_queries = len(all_latencies)
        qps = total_queries / total_time if total_time > 0 else 0
        return {
            "qps": qps,
            "avg_latency_ms": safe_mean(all_latencies),
            "p95_latency_ms": percentile(all_latencies, 0.95),
            "total_queries": total_queries
        }
    
    def _concurrent_worker(
        self, 
        model_cfg: Dict[str, Any], 
        db_path: str, 
        queries: List[str], 
        results_queue: queue.Queue
    ):
        """Worker function for concurrent benchmark."""
        embed_fn = self._embedding.get_embedding_fn(model_cfg)
        latencies = []
        
        with sqlite_connection(db_path) as conn:
            for query in queries:
                t0 = time.time()
                q_emb = embed_fn([query])[0]
                _ = self._database.vector_search(conn, q_emb, 5)
                latencies.append((time.time() - t0) * 1000)
        results_queue.put(latencies)
    
    def run_benchmark(
        self, 
        model_cfg: Dict[str, Any], 
        db_path: str, 
        chunks: List[Dict[str, Any]], 
        queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run format-aware benchmark."""
        embed_fn = self._embedding.get_embedding_fn(model_cfg)
        
        format_recalls = {}
        category_recalls = {}
        
        def evaluate(hybrid_ids, q, conn):
            expected_format = q["expected_format"]
            expected_path_contains = q["expected_path_contains"]
            category = q["category"]
            
            if expected_format not in format_recalls:
                format_recalls[expected_format] = []
            format_recalls[expected_format].append(
                self._metrics.evaluate_format_recall(hybrid_ids, expected_format, 5, conn)
            )
            
            if category not in category_recalls:
                category_recalls[category] = []
            category_recalls[category].append(
                self._metrics.evaluate_recall(hybrid_ids, expected_path_contains, 5, conn)
            )
            return None
        
        result = self._process_queries(model_cfg, db_path, queries, embed_fn, evaluate)
        
        return {
            "latencies": result["latencies"],
            "cold_start_ms": result["cold_start_ms"],
            "format_recalls": format_recalls,
            "category_recalls": category_recalls,
        }
    
    def start_llama_server(self, model_cfg: Dict[str, Any]):
        """Start llama.cpp server for embedding or generation."""
        import httpx
        gguf_path = model_cfg["gguf_path"]
        port = model_cfg["server_port"]
        ctx = model_cfg.get("ctx", 8192)
        
        try:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                return None
        except:
            pass

        cmd = [
            "llama-server",
            "-m", gguf_path,
            "--port", str(port),
            "--ctx-size", str(ctx),
            *(["--embedding"] if model_cfg.get("type") == "llama_cpp" else []),
            "--host", "127.0.0.1",
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        
        for _ in range(30):
            try:
                resp = httpx.get(f"http://localhost:{port}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            proc.terminate()
            raise RuntimeError(f"llama-server failed to start on port {port}")
        
        return proc
    
    def stop_llama_server(self, proc):
        """Stop llama.cpp server."""
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                proc.terminate()
    
    def to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self.to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.to_serializable(v) for v in obj)
        return obj

    def estimate_api_cost(self, model_name: str, num_queries: int) -> float:
        """Estimate API cost for OpenAI models."""
        pricing = OPENAI_PRICING_PER_1M_TOKENS
        total_tokens = num_queries * TOKENS_PER_QUERY
        cost_per_1m = pricing.get(model_name, 0.02)
        return (total_tokens / 1_000_000) * cost_per_1m