"""
Benchmark command - run embedding benchmarks.
"""
import typer
from typing import Optional

from embedding_benchmark.types import BenchmarkLang, BenchmarkMode
from embedding_benchmark.benchmarks import run_benchmark
from embedding_benchmark.utils import get_output_dir


def benchmark(
    lang: BenchmarkLang = typer.Option(
        BenchmarkLang.EN,
        "--lang", "-l",
        help="Language for test data",
        case_sensitive=False,
    ),
    mode: BenchmarkMode = typer.Option(
        BenchmarkMode.FULL,
        "--mode", "-m",
        help="Benchmark mode",
        case_sensitive=False,
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models",
        help="Comma-separated list of models to test",
    ),
    output: str = typer.Option(
        str(get_output_dir()),
        "--output", "-o",
        help="Output directory for results",
    ),
):
    """
    Run embedding benchmark.

    MODE options:
      - full: Full benchmark (all models from preset, ~1 hour)
      - quick: Quick benchmark (3 models, ~5 min)
      - formats: Format-aware benchmark (8 models, ~10 min)
      - mini: Mini test (1 model, 1 query, ~30 sec) - for quick verification
    """
    mode_descriptions = {
        BenchmarkMode.FULL: "Full benchmark: all models from preset, ~1 hour",
        BenchmarkMode.QUICK: "Quick benchmark: 3 models, ~5 min",
        BenchmarkMode.FORMATS: "Format-aware: 8 models, ~10 min",
        BenchmarkMode.MINI: "Mini test: 1 model, 1 query, ~30 sec",
    }

    typer.echo(f"🚀 Running benchmark for language: {lang.value}")
    typer.echo(f"📊 Mode: {mode.value} - {mode_descriptions[mode]}")

    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(",")]
        typer.echo(f"🎯 Testing specific models: {', '.join(model_list)}")

    typer.echo(f"📁 Output directory: {output}")
    typer.echo(f"🌐 Language: {lang.value}")

    run_benchmark(
        lang=lang.value,
        mode=mode.value,
        model_names=model_list,
        output_dir=output,
    )

    typer.echo("✅ Benchmark completed!")