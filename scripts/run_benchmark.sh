#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="quick"
LANG="en"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        -m)
            MODE="$2"
            shift 2
            ;;
        --lang)
            LANG="$2"
            shift 2
            ;;
        -l)
            LANG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--mode quick|formats|full|mini] [--lang en|es|ca]"
            echo "Options:"
            echo "  --mode, -m    Benchmark mode (quick, formats, full, mini) [default: quick]"
            echo "  --lang, -l    Language for test data (en, es, ca) [default: en]"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--mode quick|formats|full|mini] [--lang en|es|ca]"
            exit 1
            ;;
    esac
done

if [[ ! "$LANG" =~ ^(en|es|ca)$ ]]; then
    echo "Error: Invalid language '$LANG'. Must be one of: en, es, ca"
    exit 1
fi

if [[ ! "$MODE" =~ ^(quick|formats|full|mini)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be one of: quick, formats, full, mini"
    exit 1
fi

if [[ -d ".venv" ]]; then
    source .venv/bin/activate
    echo "✅ Using existing venv"
else
    echo "⚠️  No venv found. Run ./run_all.sh first."
    exit 1
fi

# Set PYTHONPATH to include src directory
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Check required models based on mode
MISSING=0
case "$MODE" in
    quick|formats|full)
        for model in "embeddinggemma-2-Q8_0.gguf" "llama-3.2-3b-instruct-q8_0.gguf"; do
            if [[ ! -f "models/$model" ]]; then
                echo "❌ Missing: models/$model"
                MISSING=1
            fi
        done
        ;;
    mini)
        # Mini only needs fastembed models (no GGUF required)
        ;;
esac

if [[ $MISSING -eq 1 ]]; then
    echo "Run ./run_all.sh first to download models"
    exit 1
fi

echo "🚀 Running $MODE benchmark for language: $LANG"
case "$MODE" in
    quick)
        echo "   Expected time: ~5-10 minutes"
        ;;
    formats)
        echo "   Tests: Markdown, JSON, JSONL, CSV retrieval"
        echo "   Expected time: ~10 minutes"
        ;;
    full)
        echo "   Expected time: ~1 hour"
        ;;
    mini)
        echo "   Expected time: ~30 seconds"
        ;;
esac
echo

python -m embedding_benchmark.benchmarks.$MODE --lang "$LANG"

echo
echo "✅ $MODE benchmark done! Results in .orchestrator/benchmark${MODE}_results_${LANG}.json"