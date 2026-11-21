# Embedding Benchmark for Local RAG

A comprehensive, automated benchmark suite for evaluating embedding models in Retrieval-Augmented Generation (RAG) pipelines. Tests retrieval quality, semantic metrics, cross-lingual alignment, and end-to-end generation quality via LLM-as-judge.

## Features

- **10 embedding models** tested: llama.cpp (GGUF) + fastembed (ONNX)
- **Retrieval metrics**: Recall@1/3/5, MRR@10, nDCG@5/10
- **Semantic metrics**: Cosine similarity query↔chunk, cross-lingual alignment (ES/CA/EN)
- **End-to-end evaluation**: LLM generation + LLM-as-judge (1-5 rubric)
- **Concurrency benchmarks**: 4 & 8 worker throughput
- **Fully automated**: One-command setup & run
- **Cross-platform**: macOS, Linux, Windows (WSL2)

---

## Quick Start

```bash
# Clone / navigate to benchmark directory
cd embedding-benchmark

# Run everything (installs deps, downloads models, runs benchmark)
./run_all.sh
```

That's it. Results in `.orchestrator/benchmark_results.json` and console table.

---

## Requirements

### System Dependencies

| Platform | Command |
|----------|---------|
| **macOS** | `brew install python@3.11 llama.cpp hf` |
| **Linux (Ubuntu/Debian)** | `sudo apt update && sudo apt install -y python3.11 python3.11-venv git curl && curl -fsSL https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-server-linux-x64 -o /usr/local/bin/llama-server && chmod +x /usr/local/bin/llama-server && curl -fsSL https://huggingface.co/hf/hf/releases/latest/download/hf-linux-amd64 -o /usr/local/bin/hf && chmod +x /usr/local/bin/hf` |
| **Windows (WSL2)** | Use Linux instructions inside WSL2 Ubuntu |

### Python Dependencies (auto-installed)

```txt
fastembed>=0.3.0
sqlite-vec>=0.1.6
httpx>=0.27.0
psutil>=5.9.0
numpy>=1.24.0
```

---

## Directory Structure

```
embedding-benchmark/
├── run_all.sh              # Master script: setup + download + run
├── run_benchmark.py        # Python wrapper (installs deps, runs benchmark)
├── benchmark.py            # Core benchmark logic
├── requirements.txt        # Python dependencies
├── models/                 # GGUF models (auto-downloaded)
│   ├── embeddinggemma-2-Q8_0.gguf
│   ├── Qwen3-Embedding-4B-Q8_0.gguf
│   ├── gte-qwen2-1.5b-instruct-q8_0.gguf
│   ├── gte-qwen2-7b-instruct-q8_0.gguf
│   └── llama-3.2-3b-instruct-q8_0.gguf
├── test_data/              # Test documents (14 files, 79 chunks)
│   ├── docs/
│   ├── agents/
│   └── tickets/
├── .orchestrator/          # Generated outputs
│   ├── index/              # SQLite+vec+FTS5 indexes per model
│   └── benchmark_results.json
└── README.md
```

---

## Models Tested

| Model | Type | Dim | Size | Source |
|-------|------|-----|------|--------|
| `embeddinggemma-2-Q8_0` | llama.cpp | 768 | 310 MB | `ggml-org/embeddinggemma-2-GGUF` |
| `Qwen3-Embedding-4B` | llama.cpp | 2560 | 4.3 GB | `Qwen/Qwen3-Embedding-4B-GGUF` |
| `gte-qwen2-1.5b-instruct` | llama.cpp | 1536 | 1.9 GB | `mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF` |
| `gte-qwen2-7b-instruct` | llama.cpp | 3584 | 8.1 GB | `fishbone64/gte-Qwen2-7B-instruct-Q8_0-GGUF` |
| `llama-3.2-3b-instruct` | llama.cpp | — | 3.4 GB | `bartowski/Llama-3.2-3B-Instruct-GGUF` |
| `mxbai-embed-large-v1` | fastembed | 1024 | — | `mixedbread-ai/mxbai-embed-large-v1` |
| `paraphrase-multilingual-MiniLM-L12-v2` | fastembed | 384 | — | `sentence-transformers/...` |
| `intfloat/multilingual-e5-large` | fastembed | 1024 | — | `intfloat/multilingual-e5-large` |
| `jina-embeddings-v2-base-es` | fastembed | 768 | — | `jinaai/jina-embeddings-v2-base-es` |
| `Qwen3-Embedding-0.6B` | fastembed | 1024 | — | `Qwen/Qwen3-Embedding-0.6B` |
| `Qwen3-Embedding-0.6B-Q` | fastembed | 1024 | — | `Qwen/Qwen3-Embedding-0.6B-Q` |
| `gte-base` | fastembed | 768 | — | `thenlper/gte-base` |
| `gte-large` | fastembed | 1024 | — | `thenlper/gte-large` |

> **Note**: fastembed models download automatically on first run (~1-2 GB each). llama.cpp models are pre-downloaded by `run_all.sh`.

---

## Usage

### Option 1: Fully Automated (Recommended)

```bash
./run_all.sh
```

This script:
1. Detects OS (macOS/Linux/WSL)
2. Installs system deps (Homebrew/apt)
3. Creates Python venv
4. Installs Python packages
5. Downloads all GGUF models via `hf` CLI
6. Runs benchmark

### Option 2: Step by Step

```bash
# 1. Install system deps manually (see Requirements)
# 2. Create venv
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install Python deps
pip install -r requirements.txt

# 4. Download GGUF models
mkdir -p models
hf download ggml-org/embeddinggemma-2-GGUF --include "*Q8_0*" --local-dir models
hf download Qwen/Qwen3-Embedding-4B-GGUF --include "*Q8_0*" --local-dir models
hf download mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF --local-dir models/gte-qwen2-1.5b
cp models/gte-qwen2-1.5b/gte-qwen2-1.5b-instruct-q8_0.gguf models/
hf download fishbone64/gte-Qwen2-7B-instruct-Q8_0-GGUF --local-dir models/gte-qwen2-7b
cp models/gte-qwen2-7b/gte-qwen2-7b-instruct-q8_0.gguf models/
hf download bartowski/Llama-3.2-3B-Instruct-GGUF --include "*Q8_0*" --local-dir models
mv models/Llama-3.2-3B-Instruct-Q8_0.gguf models/llama-3.2-3b-instruct-q8_0.gguf

# 5. Run benchmark
python run_benchmark.py
```

### Option 3: Custom Configuration

Edit `benchmark.py` to modify:
- `TEST_QUERIES` — add your own queries with expected answers
- `EMBEDDING_MODELS` — add/remove models
- `GENERATION_MODEL` / `JUDGE_MODEL` — change LLM
- `JUDGE_RUBRIC` — customize evaluation criteria

---

## Output

### Console Table

```
Model                       | Lat(ms) | QPS  | R@5 H | MRR H | nDCG5 H | Sim@5 H | X-Ling | Judge | CQPS4 | RAMq | BD
----------------------------------------------------------------------------------------------------------------
embeddinggemma-2-Q8_0       | 26      | 37.8 | 69%   | 0.436 | 0.765   | 0.742   | 0.897  | 2.62  | 117.3 | 111  | 3.1
intfloat/multilingual-e5... | 1267    | 0.8  | 77%   | 0.495 | 0.785   | 0.836   | 0.954  | 2.21  | 1.4   | 2511 | 4.1
...
```

### JSON Results (`.orchestrator/benchmark_results.json`)

```json
[
  {
    "model": "embeddinggemma-2-Q8_0",
    "avg_latency_ms": 26,
    "throughput_qps": 37.8,
    "recall_at_5_hybrid": 0.69,
    "mrr_at_10_hybrid": 0.436,
    "ndcg_at_5_hybrid": 0.765,
    "avg_sim_at_5_hybrid": 0.742,
    "cross_lingual_consistency": 0.897,
    "judge_score_avg": 2.62,
    "judge_score_dist": {"1": 8, "2": 25, "3": 0, "4": 7, "5": 8},
    "concurrent_qps_4": 117.3,
    "concurrent_p95_4": 46,
    "infer_ram_mb": 111,
    "index_size_mb": 3.1
  }
]
```

---

## Metrics Explained

| Metric | Description | Higher Better? |
|--------|-------------|----------------|
| `Lat(ms)` | Avg query latency (embed + search) | ❌ |
| `QPS` | Single-thread queries/second | ✅ |
| `R@5 H` | Recall@5 Hybrid (vector + BM25) | ✅ |
| `MRR@10 H` | Mean Reciprocal Rank @10 Hybrid | ✅ |
| `nDCG@5 H` | Normalized DCG @5 Hybrid (graded relevance) | ✅ |
| `Sim@5 H` | Avg cosine similarity query↔top5 chunks | ✅ |
| `X-Ling` | Cross-lingual alignment (ES/CA/EN query embeddings) | ✅ |
| `Judge` | LLM-as-judge score (1-5) on generated answers | ✅ |
| `CQPS4` | Concurrent QPS (4 workers) | ✅ |
| `RAMq` | Inference RAM (MB) | ❌ |
| `BD` | Index size (MB) | ❌ |

---

## Test Data

The benchmark uses 14 Markdown files from a restaurant project:
- **Business rules**: IVA, allergens, cancellation, hours, parking
- **Technical**: migrations, idempotency, availability, Server Actions
- **Process**: handoff, write sets, pre-commit, session rotation
- **Semantic**: parking, no-show, take away vs delivery, loyalty

Queries in 3 languages: Spanish (ES), Catalan (CA), English (EN).

---

## Customizing for Your Project

### 1. Replace Test Data

```bash
# Your docs go here
mkdir -p test_data/docs
cp -r /your/project/docs/* test_data/docs/
cp -r /your/project/agents/* test_data/agents/
cp -r /your/project/tickets/* test_data/tickets/
```

### 2. Add Your Queries

Edit `TEST_QUERIES` in `benchmark.py`:

```python
TEST_QUERIES = [
    {
        "es": "tu pregunta en español",
        "ca": "la teva pregunta en català",
        "en": "your question in english",
        "expected_file": "docs/tu-archivo.md",
        "expected_answer": "Respuesta esperada de referencia para el juez",
        "category": "tu_categoria"
    },
    # ...
]
```

### 3. Add Custom Models

```python
EMBEDDING_MODELS = [
    # ... existing ...
    {"name": "tu-modelo", "type": "fastembed", "model_name": "org/modelo", "dim": 768},
    {"name": "tu-gguf", "type": "llama_cpp", "gguf_path": "models/tu-modelo.gguf", "server_port": 8085, "dim": 1024, "ctx": 8192},
]
```

---

## Troubleshooting

### `llama-server` fails to start

```bash
# Check port conflicts
lsof -i :8081 :8082 :8083 :8084

# Check model compatibility
llama-server -m models/tu-modelo.gguf --embedding --port 8081 --ctx-size 8192
```

### `sqlite-vec` extension not found (macOS)

```bash
# Use Homebrew Python (not system Python)
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
```

### `fastembed` model download fails

```bash
# Set HF token for higher rate limits
export HF_TOKEN=your_token_here
# Or use offline cache
export HF_HOME=/path/to/cache
```

### Out of Memory (Qwen3-4B, GTE-7B)

```bash
# Reduce context size in benchmark.py
"ctx": 4096  # instead of 8192

# Or use quantized version (Q4_K_M instead of Q8_0)
```

### Windows: Use WSL2

```powershell
# In PowerShell
wsl --install -d Ubuntu
# Then follow Linux instructions inside WSL
```

---

## Performance Tips

| Scenario | Recommendation |
|----------|----------------|
| **Max speed** | `embeddinggemma-2-Q8_0` (llama.cpp) — 26ms, 187 QPS concurrent |
| **Max quality** | `intfloat/multilingual-e5-large` (fastembed) — best nDCG/MRR/Sim |
| **Best Spanish** | `mxbai-embed-large-v1` or `jina-embeddings-v2-base-es` |
| **Low RAM** | `embeddinggemma-2-Q8_0` (111 MB) or `gte-base` (fastembed) |
| **Cross-lingual** | `intfloat/multilingual-e5-large` (0.954 alignment) |

---

## License

MIT License — Free for commercial and personal use.

---

## Contributing

1. Fork the repo
2. Add your model/config
3. Run benchmark
4. Submit PR with results table

---

## Citation

If you use this benchmark in research:

```bibtex
@software{embedding_benchmark_2024,
  title = {Embedding Benchmark for Local RAG},
  author = {Adrian},
  year = {2024},
  url = {https://github.com/your-repo/embedding-benchmark}
}
```