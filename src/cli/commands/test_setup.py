"""
Test setup command - verify environment is correctly configured.
"""
import typer

from embedding_benchmark.types import BenchmarkLang
from embedding_benchmark.benchmarks.mini import run_mini_benchmark


def test_setup(
    lang: BenchmarkLang = typer.Option(
        BenchmarkLang.EN,
        "--lang", "-l",
        help="Language for test data",
        case_sensitive=False,
    ),
):
    """
    Run a quick setup verification test (mini benchmark).

    Runs a minimal benchmark with 1 model and 1 query to verify
    the environment is correctly set up. Takes ~30 seconds.
    """
    typer.echo(f"🧪 Running setup verification test for language: {lang.value}")
    typer.echo("⚡ Mini test: 1 model, 1 query, ~30 seconds")

    try:
        run_mini_benchmark(lang.value)
        typer.echo("✅ Setup verification completed!")
    except Exception as e:
        typer.echo(f"❌ Setup verification failed: {e}")
        raise typer.Exit(1)