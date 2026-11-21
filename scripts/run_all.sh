#!/usr/bin/env bash

set -euo pipefail

LANG="en"
BENCHMARK_TYPE="full"
while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)
            LANG="$2"
            shift 2
            ;;
        --quick)
            BENCHMARK_TYPE="quick"
            shift
            ;;
        --formats)
            BENCHMARK_TYPE="formats"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --lang en|es|ca    Language for test data (default: en)"
            echo "  --quick            Run quick benchmark only (3 models, ~5 min)"
            echo "  --formats          Run format-aware benchmark only (8 models, ~10 min)"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! "$LANG" =~ ^(en|es|ca)$ ]]; then
    echo "Error: Invalid language '$LANG'. Must be one of: en, es, ca"
    exit 1
fi

echo "=================================================================="
echo "  Embedding Benchmark - Automated Setup & Run"
echo "  Language: $LANG | Benchmark: $BENCHMARK_TYPE"
echo "=================================================================="

detect_os() {
    case "$(uname -s)" in
        Darwin*)    OS="macos" ;;
        Linux*)     OS="linux" ;;
        CYGWIN*|MINGW*|MSYS*) OS="windows" ;;
        *)          OS="unknown" ;;
    esac
    echo "Detected OS: $OS"
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

install_system_deps() {
    echo "Installing system dependencies for $OS..."
    case "$OS" in
        macos)
            if ! has_cmd brew; then
                echo "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                eval "$(/opt/homebrew/bin/brew shellenv)"
            fi
            brew update
            brew install python@3.11 llama.cpp hf
            ;;
        linux)
            if has_cmd apt; then
                sudo apt update
                sudo apt install -y python3.11 python3.11-venv python3.11-dev git curl wget build-essential
                if ! has_cmd llama-server; then
                    LLAMA_VER=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
                    curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VER}/llama-server-linux-x64" -o /usr/local/bin/llama-server
                    chmod +x /usr/local/bin/llama-server
                fi
                if ! has_cmd hf; then
                    curl -fsSL "https://huggingface.co/hf/hf/releases/latest/download/hf-linux-amd64" -o /usr/local/bin/hf
                    chmod +x /usr/local/bin/hf
                fi
            elif has_cmd dnf; then
                sudo dnf install -y python3.11 python3.11-venv git curl wget gcc gcc-c++ make
            elif has_cmd pacman; then
                sudo pacman -S --needed python git curl wget base-devel
            else
                echo "Unsupported Linux distro. Install python3.11, llama.cpp, hf manually."
                exit 1
            fi
            ;;
        windows)
            echo "Windows native not supported. Use WSL2:"
            echo "  wsl --install -d Ubuntu"
            echo "  Then run this script inside WSL."
            exit 1
            ;;
        *)
            echo "Unknown OS. Install python3.11, llama.cpp, hf manually."
            exit 1
            ;;
    esac
    echo "System dependencies installed"
}

setup_venv() {
    echo "Setting up Python virtual environment..."
    PYTHON_BIN="python3.11"
    if ! has_cmd "$PYTHON_BIN"; then PYTHON_BIN="python3"; fi
    if ! has_cmd "$PYTHON_BIN"; then
        echo "Python 3.11+ not found. Install it first."
        exit 1
    fi
    if [[ ! -d ".venv" ]]; then
        "$PYTHON_BIN" -m venv .venv
        echo "Virtual environment created"
    else
        echo "Virtual environment exists, reusing"
    fi
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    echo "Python dependencies installed"
}

download_models() {
    echo "Downloading GGUF models..."
    mkdir -p models
    declare -A EMBED_MODELS=(
        ["embeddinggemma-2-Q8_0.gguf"]="ggml-org/embeddinggemma-2-GGUF *Q8_0*"
        ["Qwen3-Embedding-4B-Q8_0.gguf"]="Qwen/Qwen3-Embedding-4B-GGUF *Q8_0*"
    )
    for file in "${!EMBED_MODELS[@]}"; do
        if [[ ! -f "models/$file" ]]; then
            echo "Downloading $file..."
            hf download ${EMBED_MODELS[$file]} --local-dir models || echo "Warning: Failed to download $file"
        else
            echo "$file already exists"
        fi
    done
    if [[ ! -f "models/gte-qwen2-1.5b-instruct-q8_0.gguf" ]]; then
        echo "Downloading gte-qwen2-1.5b-instruct..."
        hf download mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF --local-dir models/gte-qwen2-1.5b || echo "Warning: Failed"
        cp models/gte-qwen2-1.5b/gte-qwen2-1.5b-instruct-q8_0.gguf models/ 2>/dev/null || true
    fi
    if [[ ! -f "models/gte-qwen2-7b-instruct-q8_0.gguf" ]]; then
        echo "Downloading gte-qwen2-7b-instruct..."
        hf download fishbone64/gte-Qwen2-7B-instruct-Q8_0-GGUF --local-dir models/gte-qwen2-7b || echo "Warning: Failed"
        cp models/gte-qwen2-7b/gte-qwen2-7b-instruct-q8_0.gguf models/ 2>/dev/null || true
    fi
    if [[ ! -f "models/llama-3.2-3b-instruct-q8_0.gguf" ]]; then
        echo "Downloading llama-3.2-3b-instruct..."
        hf download bartowski/Llama-3.2-3B-Instruct-GGUF --include "*Q8_0*" --local-dir models || echo "Warning: Failed"
        mv models/Llama-3.2-3B-Instruct-Q8_0.gguf models/llama-3.2-3b-instruct-q8_0.gguf 2>/dev/null || true
    fi
    echo "Models downloaded"
}

run_benchmark() {
    echo "Running $BENCHMARK_TYPE benchmark for language: $LANG..."
    source .venv/bin/activate
    case "$BENCHMARK_TYPE" in
        quick)
            python -m embedding_benchmark.benchmarks.quick --lang "$LANG"
            ;;
        formats)
            python -m embedding_benchmark.benchmarks.formats --lang "$LANG"
            ;;
        full|*)
            python -m embedding_benchmark.benchmarks.full --lang "$LANG"
            ;;
    esac
    echo "Benchmark completed"
    echo "Results in .orchestrator/benchmark_results_${LANG}.json"
}

main() {
    echo "=================================================================="
    echo "  Embedding Benchmark - Automated Setup & Run"
    echo "  Language: $LANG | Benchmark: $BENCHMARK_TYPE"
    echo "=================================================================="
    detect_os
    install_system_deps
    setup_venv
    download_models
    run_benchmark
    echo
    echo "=================================================================="
    echo "  All done! Check .orchestrator/benchmark_results_${LANG}.json"
    echo "=================================================================="
}

main "$@"