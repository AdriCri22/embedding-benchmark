"""
Metrics module for the embedding benchmark suite.
"""
from .interface import MetricsInterface

_metrics: MetricsInterface = None


def get_metrics() -> MetricsInterface:
    """Get the metrics implementation singleton."""
    global _metrics
    if _metrics is None:
        from .internal.metrics_impl import MetricsImpl
        _metrics = MetricsImpl()
    return _metrics


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
evaluate_format_recall = metrics.evaluate_format_recall
safe_str = metrics.safe_str


__all__ = [
    "MetricsInterface", "get_metrics",
    "get_ram_mb", "escape_fts5", "embeddings_to_bytes", "cosine_sim",
    "dcg_at_k", "ndcg_at_k", "evaluate_recall", "evaluate_mrr", "evaluate_ndcg",
    "evaluate_format_recall", "safe_str",
]