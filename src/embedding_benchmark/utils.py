"""
Shared utilities for the embedding benchmark suite.
"""
import sqlite3
import sqlite_vec
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Set, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_config_dir() -> Path:
    """Get the config directory."""
    return get_project_root() / "config"


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_project_root() / "models"


def get_output_dir() -> Path:
    """Get the default output directory."""
    return get_project_root() / ".orchestrator"


DEFAULT_SKIP_DIRS: Set[str] = {
    "node_modules", ".git", ".orchestrator", ".next", "dist", "coverage", "models"
}


def get_skip_dirs() -> Set[str]:
    """Get the set of directories to skip when loading test data."""
    return DEFAULT_SKIP_DIRS.copy()


def check_gguf_exists(model_cfg: Dict[str, Any]) -> bool:
    """Check if GGUF file exists for a llama_cpp model."""
    if model_cfg.get("type") != "llama_cpp":
        return True
    gguf_path = model_cfg.get("gguf_path")
    if not gguf_path:
        return False
    return Path(gguf_path).exists()


def get_missing_ggufs(models: List[Dict[str, Any]]) -> List[str]:
    """Get list of missing GGUF model names."""
    missing = []
    for m in models:
        if m.get("type") == "llama_cpp" and not check_gguf_exists(m):
            missing.append(m["name"])
    return missing


def has_openai_key() -> bool:
    """Check if OPENAI_API_KEY is set."""
    import os
    return bool(os.environ.get("OPENAI_API_KEY"))


def safe_mean(values: List[float]) -> float:
    """Calculate mean safely, returning 0.0 for empty lists."""
    import statistics
    return statistics.mean(values) if values else 0.0


def calc_qps(latencies: List[float]) -> float:
    """Calculate QPS from latencies in milliseconds."""
    if not latencies:
        return 0.0
    import statistics
    return 1000 / statistics.mean(latencies)


def print_skip(model_name: str, reason: str) -> None:
    """Print standardized skip message."""
    print(f"\n⏭️  SALTANDO {model_name} ({reason})")


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print formatted table."""
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*row))


def create_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """Create SQLite connection with sqlite-vec extension loaded."""
    import sqlite3
    import sqlite_vec
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn


@contextmanager
def sqlite_connection(db_path: str):
    """Context manager for SQLite connection with auto-close."""
    conn = create_sqlite_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def percentile(latencies: List[float], p: float) -> float:
    """Calculate percentile of latencies."""
    if not latencies:
        return 0.0
    return sorted(latencies)[int(len(latencies) * p)]