#!/usr/bin/env python3
"""
Embedding Benchmark Suite - Main Entry Point
"""

import sys
from pathlib import Path


def _add_src_to_path() -> None:
    """Add the src directory to Python path for package imports."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> None:
    _add_src_to_path()
    from cli import app
    app()


if __name__ == "__main__":
    main()