"""
Database implementation for the embedding benchmark suite.
"""
import sqlite3
import sqlite_vec
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

from ..interface import DatabaseInterface
from embedding_benchmark.embedding import get_embedding
from embedding_benchmark.metrics import get_metrics


# Batch sizes per model type to avoid token limit issues
BATCH_SIZE_LLAMA_CPP = 8
BATCH_SIZE_DEFAULT = 32


class DatabaseImpl(DatabaseInterface):
    """Implementation of database operations."""
    
    def __init__(self):
        self._embedding = get_embedding()
        self._metrics = get_metrics()
    
    def build_index(
        self, 
        model_cfg: Dict[str, Any], 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build SQLite index with vector and FTS5 search."""
        safe_name = model_cfg['name'].replace('/', '_').replace('.', '_')
        db_path = Path(f".orchestrator/index/bench_{safe_name}.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        conn.executescript(f"""
            CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, path TEXT, format TEXT, heading TEXT, line INT, body TEXT);
            CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{model_cfg['dim']}]);
            CREATE VIRTUAL TABLE fts_chunks USING fts5(path, format, heading, body, tokenize='unicode61');
        """)

        batch_size = BATCH_SIZE_LLAMA_CPP if model_cfg.get('type') == 'llama_cpp' else BATCH_SIZE_DEFAULT
        embed_fn = self._embedding.get_embedding_fn(model_cfg)

        start = time.time()
        ram_before = self._metrics.get_ram_mb()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embed_fn(texts)

            with conn:
                for chunk, emb in zip(batch, embeddings):
                    cur = conn.execute(
                        "INSERT INTO chunks (path, format, heading, line, body) VALUES (?, ?, ?, ?, ?)",
                        (chunk["path"], chunk["format"], chunk["heading"], chunk["line"], chunk["text"])
                    )
                    rowid = cur.lastrowid
                    emb_bytes = self._metrics.embeddings_to_bytes(emb)
                    conn.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, emb_bytes))
                    conn.execute(
                        "INSERT INTO fts_chunks(rowid, path, format, heading, body) VALUES (?, ?, ?, ?, ?)",
                        (rowid, chunk["path"], chunk["format"], chunk["heading"], chunk["text"])
                    )

        index_time = time.time() - start
        ram_after = self._metrics.get_ram_mb()
        index_size = db_path.stat().st_size / (1024*1024)
        conn.close()

        return {
            "index_time_sec": index_time,
            "index_ram_mb": ram_after - ram_before,
            "chunks_per_sec": len(chunks) / index_time,
            "index_size_mb": index_size,
            "db_path": str(db_path)
        }
    
    def vector_search(
        self, 
        conn: sqlite3.Connection, 
        q_emb: List[float], 
        k: int
    ) -> List[Tuple]:
        """Search using vector similarity (cosine distance)."""
        q_emb_bytes = self._metrics.embeddings_to_bytes(q_emb)
        return conn.execute(
            "SELECT c.rowid, c.path, c.format, c.heading, c.line, c.body, "
            "vec_distance_cosine(v.embedding, ?) as dist "
            "FROM vec_chunks v JOIN chunks c ON v.rowid = c.rowid "
            "ORDER BY dist LIMIT ?",
            (self._metrics.embeddings_to_bytes(q_emb), k)
        ).fetchall()
    
    def hybrid_search(
        self, 
        conn: sqlite3.Connection, 
        q_emb: List[float], 
        query_text: str, 
        k: int, 
        rrf_k: int = 60
    ) -> List[Tuple]:
        """Hybrid search combining vector similarity and FTS5 full-text search using RRF."""
        safe_query = self._metrics.escape_fts5(query_text)
        q_emb_bytes = self._metrics.embeddings_to_bytes(q_emb)

        vec_rows = conn.execute(
            "SELECT c.rowid, c.path, c.format, c.heading, c.line, c.body, "
            "vec_distance_cosine(v.embedding, ?) as dist "
            "FROM vec_chunks v JOIN chunks c ON v.rowid = c.rowid "
            "ORDER BY dist LIMIT ?",
            (q_emb_bytes, k * 2)
        ).fetchall()

        fts_rows = conn.execute(
            "SELECT c.rowid, c.path, c.format, c.heading, c.line, c.body, "
            "bm25(fts_chunks) as score "
            "FROM fts_chunks JOIN chunks c ON fts_chunks.rowid = c.rowid "
            "WHERE fts_chunks MATCH ? ORDER BY score LIMIT ?",
            (safe_query, k * 2)
        ).fetchall()

        scores = {}
        for rank, row in enumerate(vec_rows):
            scores[row[0]] = scores.get(row[0], 0) + 1.0 / (rrf_k + rank + 1)
        for rank, row in enumerate(fts_rows):
            scores[row[0]] = scores.get(row[0], 0) + 1.0 / (rrf_k + rank + 1)

        top_ids = sorted(scores.keys(), key=lambda x: -scores[x])[:k]
        id_to_row = {r[0]: r for r in vec_rows}
        id_to_row.update({r[0]: r for r in fts_rows})

        return [id_to_row[tid] for tid in top_ids if tid in id_to_row]