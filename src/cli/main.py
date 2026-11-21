#!/usr/bin/env python3
"""CLI for Embedding Benchmark Suite"""
import typer

from .commands import (
    setup,
    benchmark,
    list_models,
    list_presets,
    download_models,
    test_setup,
)

app = typer.Typer(
    name="embedding-benchmark",
    help="Embedding Benchmark Suite CLI",
    add_completion=False,
    no_args_is_help=True,
)

app.command()(setup)
app.command()(benchmark)
app.command()(list_models)
app.command()(list_presets)
app.command()(download_models)
app.command()(test_setup)


if __name__ == "__main__":
    app()