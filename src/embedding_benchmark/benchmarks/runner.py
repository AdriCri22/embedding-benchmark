"""
Unified Benchmark Runner - Single parameterized function for all benchmark modes.
"""
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Callable

from embedding_benchmark.config import get_config
from embedding_benchmark.benchmark import get_benchmark
from embedding_benchmark.data_loader import get_data_loader
from embedding_benchmark.benchmark import to_serializable
from embedding_benchmark.metrics import get_ram_mb
from embedding_benchmark.models.registry import get_registry
from embedding_benchmark.utils import (
    get_output_dir, check_gguf_exists, get_missing_ggufs, percentile,
    has_openai_key, safe_mean, calc_qps, print_skip, print_table
)


@dataclass
class BenchmarkResult:
    """Unified benchmark result dataclass."""
    model: str
    lang: str
    index_time_sec: float = 0.0
    index_ram_mb: float = 0.0
    chunks_per_sec: float = 0.0
    index_size_mb: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    infer_ram_mb: float = 0.0
    recall_at_5_hybrid: float = 0.0
    mrr_at_10_hybrid: float = 0.0
    ndcg_at_5_hybrid: float = 0.0
    concurrent_qps_4: float = 0.0
    concurrent_qps_8: float = 0.0
    concurrent_p95_4: float = 0.0
    concurrent_p95_8: float = 0.0
    cold_start_ms: float = 0.0
    api_cost_per_1k_queries_usd: float = 0.0
    recall_md: float = 0.0
    recall_json: float = 0.0
    recall_jsonl: float = 0.0
    recall_csv: float = 0.0
    recall_company_info: float = 0.0
    recall_api_spec: float = 0.0
    recall_hr_policy: float = 0.0
    recall_legal: float = 0.0
    recall_runbook: float = 0.0
    recall_incident: float = 0.0
    recall_finance: float = 0.0
    db_path: str = ""
    errors: List[str] = field(default_factory=list)


def run_benchmark(
    lang: str = "en",
    mode: str = "full",
    model_names: Optional[List[str]] = None,
    output_dir: str = str(get_output_dir()),
) -> List[BenchmarkResult]:
    """
    Unified benchmark runner.

    Args:
        lang: Language (en, es, ca)
        mode: Benchmark mode (full, quick, formats, mini)
        model_names: Optional list of specific model names to test
        output_dir: Output directory for results

    Returns:
        List of BenchmarkResult objects
    """
    config = get_config()
    benchmark = get_benchmark()
    data_loader = get_data_loader()
    registry = get_registry()

    mode_config = {
        "full": {
            "preset": "full",
            "use_format_queries": False,
            "run_concurrent": True,
            "concurrent_workers": [4, 8],
            "output_suffix": "",
            "description": "Full benchmark: all models, ~1 hour",
        },
        "quick": {
            "preset": "quick",
            "use_format_queries": False,
            "run_concurrent": True,
            "concurrent_workers": [4],
            "output_suffix": "_quick",
            "description": "Quick benchmark: 3 models, ~5 min",
        },
        "formats": {
            "preset": "formats",
            "use_format_queries": True,
            "run_concurrent": False,
            "concurrent_workers": [],
            "output_suffix": "_format",
            "description": "Format-aware benchmark: 8 models, ~10 min",
        },
        "mini": {
            "preset": None,
            "use_format_queries": False,
            "run_concurrent": False,
            "concurrent_workers": [],
            "output_suffix": "_mini",
            "description": "Mini test: 1 model, 1 query, ~30 sec",
        },
    }

    if mode not in mode_config:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {list(mode_config.keys())}")

    cfg = mode_config[mode]

    if mode == "mini":
        embedding_models = [
            {"name": "gte-base", "type": "fastembed", "model_name": "thenlper/gte-base", "dim": 768},
        ]
    else:
        if model_names:
            embedding_models = []
            for name in model_names:
                model = registry.get(name)
                if model and model.enabled:
                    embedding_models.append(model.to_dict())
                else:
                    print(f"   ⚠️  Model not found or disabled: {name}")
        else:
            preset_models = registry.get_preset(cfg["preset"])
            embedding_models = [m.to_dict() for m in preset_models if m.enabled]

    if cfg["use_format_queries"]:
        QUERIES = config.get_format_queries(lang)
        chunks_loader = data_loader.load_all_documents
    else:
        QUERIES = config.get_queries(lang)
        chunks_loader = data_loader.load_chunks

    TEST_DATA = config.get_test_data_dir(lang)
    GENERATION_MODEL = config.generation_model
    JUDGE_MODEL = config.judge_model

    print("=" * 80)
    print(f"{mode.upper()} EMBEDDING BENCHMARK - Language: {lang.upper()}")
    print("=" * 80)
    print(f"\n📁 Datos: {TEST_DATA} | ❓ Queries: {len(QUERIES)}")
    print(f"🤖 Modelos: {len(embedding_models)}")
    print(f"📝 {cfg['description']}")

    print("\n📚 Cargando chunks...")
    chunks = chunks_loader(TEST_DATA)
    print(f"   {len(chunks)} chunks de {len(set(c['path'] for c in chunks))} archivos")

    missing = get_missing_ggufs(embedding_models)
    for name in missing:
        print(f"   ⚠️  GGUF no encontrado: {name}")

    for m in [GENERATION_MODEL, JUDGE_MODEL]:
        if not check_gguf_exists(m):
            print(f"   ⚠️  GGUF gen/juez no encontrado: {m.get('gguf_path')}")

    openai_models = [m for m in embedding_models if m["type"] == "openai"]
    if openai_models and not has_openai_key():
        print("   ⚠️  OPENAI_API_KEY no configurado - modelos OpenAI se saltarán")

    results = []

    for model_cfg in embedding_models:
        if model_cfg["type"] == "openai" and not has_openai_key():
            print_skip(model_cfg["name"], "requiere OPENAI_API_KEY")
            continue

        if model_cfg["type"] == "llama_cpp" and not check_gguf_exists(model_cfg):
            print_skip(model_cfg["name"], "GGUF no encontrado")
            continue

        print(f"\n{'='*80}")
        print(f"🧪 PROBANDO: {model_cfg['name']}")
        print(f"{'='*80}")

        result = BenchmarkResult(model=model_cfg["name"], lang=lang)
        llama_proc = None

        try:
            if model_cfg["type"] == "llama_cpp":
                print("   🚀 Iniciando llama-server...")
                llama_proc = benchmark.start_llama_server(model_cfg)
                if llama_proc is None:
                    print("   ✅ Server ya estaba corriendo")
                else:
                    print(f"   ✅ Server iniciado en puerto {model_cfg['server_port']}")

            print("   📦 Indexando...")
            index_metrics = benchmark.build_index(model_cfg, chunks)
            result.index_time_sec = index_metrics["index_time_sec"]
            result.index_ram_mb = index_metrics["index_ram_mb"]
            result.chunks_per_sec = index_metrics["chunks_per_sec"]
            result.index_size_mb = index_metrics["index_size_mb"]
            result.db_path = index_metrics["db_path"]
            print(f"      {result.index_time_sec:.1f}s | RAM: {result.index_ram_mb:.0f} MB | {result.chunks_per_sec:.0f} ch/s | BD: {result.index_size_mb:.1f} MB")

            if cfg["use_format_queries"]:
                print("   🔍 Evaluando retrieval por formato...")
                st_metrics = benchmark.run_benchmark(model_cfg, index_metrics["db_path"], chunks, QUERIES)
                _process_format_metrics(result, st_metrics)
            else:
                print("   🔍 Evaluando retrieval (vector + híbrido)...")
                st_metrics = benchmark.run_single_thread_benchmark(model_cfg, index_metrics["db_path"], chunks, QUERIES)
                _process_standard_metrics(result, st_metrics)

            if cfg["run_concurrent"]:
                for workers in cfg["concurrent_workers"]:
                    print(f"   ⚡ Concurrente ({workers} workers)...")
                    queries_per_worker = 10 if workers == 4 else 20
                    conc = benchmark.run_concurrent_benchmark(model_cfg, index_metrics["db_path"], workers, queries_per_worker, QUERIES)
                    if workers == 4:
                        result.concurrent_qps_4 = conc["qps"]
                        result.concurrent_p95_4 = conc["p95_latency_ms"]
                    elif workers == 8:
                        result.concurrent_qps_8 = conc["qps"]
                        result.concurrent_p95_8 = conc["p95_latency_ms"]
                    print(f"      QPS: {conc['qps']:.1f} | P95: {conc['p95_latency_ms']:.0f}ms")

            if model_cfg["type"] == "openai":
                result.api_cost_per_1k_queries_usd = benchmark.estimate_api_cost(model_cfg["model_name"], 1000)
                print(f"      Costo API estimado: ${result.api_cost_per_1k_queries_usd:.4f} por 1K queries")

            results.append(result)

        except Exception as e:
            result.errors.append(str(e))
            print(f"   ❌ ERROR: {e}")
            results.append(result)
        finally:
            if llama_proc:
                benchmark.stop_llama_server(llama_proc)
                print("   🛑 llama-server detenido")

    _print_results_table(results, mode, cfg)
    _save_results(results, lang, mode, cfg, output_dir)
    _print_summary(results, mode)

    return results


def _process_standard_metrics(result: BenchmarkResult, st_metrics: Dict[str, Any]) -> None:
    latencies = st_metrics["latencies"]
    result.avg_latency_ms = safe_mean(latencies)
    result.p50_latency_ms = percentile(latencies, 0.50)
    result.p95_latency_ms = percentile(latencies, 0.95)
    result.p99_latency_ms = percentile(latencies, 0.99)
    result.throughput_qps = calc_qps(latencies)
    result.cold_start_ms = st_metrics["cold_start_ms"]
    result.infer_ram_mb = get_ram_mb()

    result.recall_at_5_hybrid = safe_mean(st_metrics["recalls_hybrid"])
    result.mrr_at_10_hybrid = safe_mean(st_metrics["mrr_hybrid"])
    result.ndcg_at_5_hybrid = safe_mean(st_metrics["ndcg5_hybrid"])

    print(f"      Latencia: avg={result.avg_latency_ms:.0f}ms p50={result.p50_latency_ms:.0f}ms p95={result.p95_latency_ms:.0f}ms p99={result.p99_latency_ms:.0f}ms")
    print(f"      Recall@5H: {result.recall_at_5_hybrid:.1%} | MRR@10H: {result.mrr_at_10_hybrid:.3f} | nDCG@5H: {result.ndcg_at_5_hybrid:.3f}")


def _process_format_metrics(result: BenchmarkResult, st_metrics: Dict[str, Any]) -> None:
    latencies = st_metrics["latencies"]
    result.avg_latency_ms = safe_mean(latencies)
    result.p50_latency_ms = percentile(latencies, 0.50)
    result.p95_latency_ms = percentile(latencies, 0.95)
    result.throughput_qps = calc_qps(latencies)
    result.cold_start_ms = st_metrics["cold_start_ms"]
    result.infer_ram_mb = get_ram_mb()

    fmt = st_metrics["format_recalls"]
    cat = st_metrics["category_recalls"]

    result.recall_md = safe_mean(fmt.get("md", []))
    result.recall_json = safe_mean(fmt.get("json", []))
    result.recall_jsonl = safe_mean(fmt.get("jsonl", []))
    result.recall_csv = safe_mean(fmt.get("csv", []))

    result.recall_company_info = safe_mean(cat.get("company_info", []))
    result.recall_api_spec = safe_mean(cat.get("api_spec", []))
    result.recall_hr_policy = safe_mean(cat.get("hr_policy", []))
    result.recall_legal = safe_mean(cat.get("legal", []))
    result.recall_runbook = safe_mean(cat.get("runbook", []))
    result.recall_incident = safe_mean(cat.get("incident", []))
    result.recall_finance = safe_mean(cat.get("finance", []))

    print(f"      Latencia: avg={result.avg_latency_ms:.0f}ms p50={result.p50_latency_ms:.0f}ms p95={result.p95_latency_ms:.0f}ms")
    print(f"      Recall MD: {result.recall_md:.1%} | JSON: {result.recall_json:.1%} | JSONL: {result.recall_jsonl:.1%} | CSV: {result.recall_csv:.1%}")
    print(f"      Company: {result.recall_company_info:.1%} | API: {result.recall_api_spec:.1%} | HR: {result.recall_hr_policy:.1%} | Legal: {result.recall_legal:.1%} | Runbook: {result.recall_runbook:.1%} | Incident: {result.recall_incident:.1%} | Finance: {result.recall_finance:.1%}")


def _print_results_table(results: List[BenchmarkResult], mode: str, cfg: Dict[str, Any]) -> None:
    if mode == "formats":
        headers = ["Modelo", "Lat(ms)", "QPS", "R@5 MD", "R@5 JSON", "R@5 JSONL", "R@5 CSV", "Company", "API", "HR", "Legal", "Runbook", "Incident", "Finance", "RAMq", "BD"]
        rows = [[r.model[:25], f"{r.avg_latency_ms:.0f}", f"{r.throughput_qps:.1f}", f"{r.recall_md:.0%}", f"{r.recall_json:.0%}", f"{r.recall_jsonl:.0%}", f"{r.recall_csv:.0%}", f"{r.recall_company_info:.0%}", f"{r.recall_api_spec:.0%}", f"{r.recall_hr_policy:.0%}", f"{r.recall_legal:.0%}", f"{r.recall_runbook:.0%}", f"{r.recall_incident:.0%}", f"{r.recall_finance:.0%}", f"{r.infer_ram_mb:.0f}", f"{r.index_size_mb:.1f}"] for r in results]
    elif mode == "mini":
        headers = ["Modelo", "Index(s)", "Lat(ms)", "QPS", "R@5H", "MRR@10H", "nDCG@5H", "RAMq", "BD"]
        rows = [[r.model[:28], f"{r.index_time_sec:.1f}", f"{r.avg_latency_ms:.0f}", f"{r.throughput_qps:.1f}", f"{r.recall_at_5_hybrid:.0%}", f"{r.mrr_at_10_hybrid:.3f}", f"{r.ndcg_at_5_hybrid:.3f}", f"{r.infer_ram_mb:.0f}", f"{r.index_size_mb:.1f}"] for r in results]
    elif mode == "quick":
        headers = ["Modelo", "Index(s)", "Lat(ms)", "QPS", "R@5H", "MRR@10H", "nDCG@5H", "CQPS4", "CP95_4", "RAMq", "BD"]
        rows = [[r.model[:28], f"{r.index_time_sec:.1f}", f"{r.avg_latency_ms:.0f}", f"{r.throughput_qps:.1f}", f"{r.recall_at_5_hybrid:.0%}", f"{r.mrr_at_10_hybrid:.3f}", f"{r.ndcg_at_5_hybrid:.3f}", f"{r.concurrent_qps_4:.1f}", f"{r.concurrent_p95_4:.0f}", f"{r.infer_ram_mb:.0f}", f"{r.index_size_mb:.1f}"] for r in results]
    else:
        headers = ["Modelo", "Index(s)", "Lat(ms)", "QPS", "R@5H", "CQPS4", "CP95_4", "CQPS8", "CP95_8", "RAMq", "BD", "Costo/1K"]
        rows = []
        for r in results:
            cost_str = f"${r.api_cost_per_1k_queries_usd:.4f}" if r.api_cost_per_1k_queries_usd > 0 else "N/A"
            rows.append([r.model[:30], f"{r.index_time_sec:.1f}", f"{r.avg_latency_ms:.0f}", f"{r.throughput_qps:.1f}", f"{r.recall_at_5_hybrid:.0%}", f"{r.concurrent_qps_4:.1f}", f"{r.concurrent_p95_4:.0f}", f"{r.concurrent_qps_8:.1f}", f"{r.concurrent_p95_8:.0f}", f"{r.infer_ram_mb:.0f}", f"{r.index_size_mb:.1f}", cost_str])

    print_table(headers, rows)


def _save_results(results: List[BenchmarkResult], lang: str, mode: str, cfg: Dict[str, Any], output_dir: str) -> None:
    out_file = Path(output_dir) / f"benchmark_results{cfg['output_suffix']}_{lang}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps([to_serializable(asdict(r)) for r in results], indent=2, ensure_ascii=False))
    print(f"\n💾 Resultados en {out_file}")


def _print_summary(results: List[BenchmarkResult], mode: str) -> None:
    print("\n🏆 RESUMEN:")
    valid = [r for r in results if r.avg_latency_ms > 0]
    if valid:
        best_speed = min(valid, key=lambda r: r.avg_latency_ms)
        print(f"   ⚡ Más rápido (latencia): {best_speed.model} ({best_speed.avg_latency_ms:.0f}ms)")

        if mode == "formats":
            best_overall = max(valid, key=lambda r: (r.recall_md + r.recall_json + r.recall_jsonl + r.recall_csv) / 4)
            best_json = max(valid, key=lambda r: r.recall_json)
            best_md = max(valid, key=lambda r: r.recall_md)
            best_jsonl = max(valid, key=lambda r: r.recall_jsonl)
            best_csv = max(valid, key=lambda r: r.recall_csv)
            print(f"   Mejor overall (avg formatos): {best_overall.model}")
            print(f"   Mejor JSON: {best_json.model} ({best_json.recall_json:.1%})")
            print(f"   Mejor Markdown: {best_md.model} ({best_md.recall_md:.1%})")
            print(f"   Mejor JSONL: {best_jsonl.model} ({best_jsonl.recall_jsonl:.1%})")
            print(f"   Mejor CSV: {best_csv.model} ({best_csv.recall_csv:.1%})")
            print(f"   Mejor velocidad: {best_speed.model} ({best_speed.avg_latency_ms:.0f}ms)")
        else:
            best_quality = max(valid, key=lambda r: r.ndcg_at_5_hybrid if r.ndcg_at_5_hybrid > 0 else r.recall_at_5_hybrid)
            best_concurrent = max(valid, key=lambda r: r.concurrent_qps_4)
            print(f"   🎯 Mejor calidad: {best_quality.model}")
            if mode != "mini":
                print(f"   🚀 Mejor concurrente (4 workers): {best_concurrent.model} ({best_concurrent.concurrent_qps_4:.1f} QPS)")

    for r in results:
        if r.errors:
            print(f"   ❌ {r.model}: {r.errors}")