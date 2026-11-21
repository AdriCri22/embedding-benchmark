# Embedding Benchmark Suite - Complete Summary

## Overview
Created a comprehensive, automated benchmark suite for evaluating embedding models in RAG pipelines. Tests retrieval quality across multiple document formats, languages, and content categories.

## Directory Structure

```
embedding-benchmark/
├── README.md                    # Full documentation (English, macOS/Linux/WSL2)
├── SUMMARY.md                   # This file
├── requirements.txt             # Python dependencies
├── run_all.sh                   # Master script: OS detection, deps, models, benchmark
├── run_quick.sh                 # Quick benchmark (3 models, 5 queries, ~5 min)
├── run_formats.sh               # Format-aware benchmark (7 models, 31 queries, ~10 min)
├── run_benchmark.py             # Python wrapper for full benchmark
├── benchmark.py                 # Full benchmark (retrieval + semantic + LLM judge)
├── benchmark_quick.py           # Quick benchmark (3 models, 5 queries)
├── benchmark_formats.py         # Format-aware benchmark (7 models, 31 queries)
├── models/                      # GGUF models (auto-downloaded)
│   ├── embeddinggemma-2-Q8_0.gguf
│   ├── Qwen3-Embedding-4B-Q8_0.gguf
│   ├── gte-qwen2-1.5b-instruct-q8_0.gguf
│   ├── gte-qwen2-7b-instruct-q8_0.gguf
│   └── llama-3.2-3b-instruct-q8_0.gguf
├── test_data/                   # Original test data (14 files, 79 chunks)
│   ├── docs/
│   ├── agents/
│   └── tickets/
└── test_data_pro/               # Professional test data (62 chunks, 4 formats)
    ├── docs/                    # Markdown (.md) - 18 chunks
    │   ├── en/company-overview.md
    │   ├── es/visión-misión.md
    │   └── ca/visió-missió.md
    ├── api/                     # JSON (.json) - 24 chunks
    │   └── en/openapi.json      # OpenAPI 3.1 spec
    ├── hr/                      # JSON (.json) - HR policies
    │   ├── en/policies.json
    │   ├── es/politicas.json
    │   └── ca/politiques.json
    ├── legal/                   # JSON (.json) - Contracts
    │   └── en/contracts.json
    ├── engineering/             # JSONL (.jsonl) - Runbooks
    │   └── en/runbooks.jsonl
    ├── ops/                     # JSONL (.jsonl) - Incidents
    │   └── en/incidents.jsonl
    └── finance/                 # JSON + CSV - Financial data
        ├── en/budget-2024.json
        └── en/quarterly-revenue.csv
```

## Benchmark Scripts

### 1. Full Benchmark (`run_all.sh` → `benchmark.py`)
- **Models**: 13 (5 llama.cpp GGUF + 8 fastembed)
- **Queries**: 16 × 3 languages = 48 evaluations
- **Metrics**: Recall@1/3/5, MRR@10, nDCG@5/10, Cosine Sim, Cross-lingual, LLM Judge (1-5)
- **Time**: ~1+ hour
- **Output**: Console table + `.orchestrator/benchmark_results.json`

### 2. Quick Benchmark (`run_quick.sh` → `benchmark_quick.py`)
- **Models**: 3 (embeddinggemma-2-Q8_0, gte-base, intfloat/multilingual-e5-large)
- **Queries**: 5 × 3 languages = 15 evaluations
- **Metrics**: Recall@5, MRR@10, nDCG@5, Cosine Sim, Cross-lingual, LLM Judge
- **Time**: ~5-6 minutes
- **Output**: Console table + `.orchestrator/benchmark_quick_results.json`

### 3. Format-Aware Benchmark (`run_formats.sh` → `benchmark_formats.py`)
- **Models**: 7 fastembed models
- **Queries**: 31 format-specific queries
- **Formats**: Markdown, JSON, JSONL, CSV
- **Categories**: Company Info, API Spec, HR Policy, Legal, Runbook, Incident, Finance
- **Languages**: EN, ES, CA
- **Time**: ~10 minutes
- **Output**: Console table + `.orchestrator/benchmark_format_results.json`

## Test Data - Professional Scenario (TechCorp)

### Company Profile
- **Name**: TechCorp - AI Infrastructure Platform
- **ARR**: $42.3M (+67% YoY)
- **Employees**: 218 across 6 departments
- **Languages**: EN, ES, CA

### Document Formats & Content

| Format | Files | Chunks | Content |
|--------|-------|--------|---------|
| **Markdown** | 3 | 18 | Company overview (EN/ES/CA) |
| **JSON** | 5 | 24 | OpenAPI 3.1 spec, HR policies (EN/ES/CA), Legal contracts, Finance budget |
| **JSONL** | 2 | 12 | Engineering runbooks (8), Incidents (4) |
| **CSV** | 1 | 8 | Quarterly revenue (8 quarters) |

### Query Categories (31 total)
| Category | Queries | Formats | Languages |
|----------|---------|---------|-----------|
| Company Info | 5 | MD | EN, ES, CA |
| API Spec | 5 | JSON | EN |
| HR Policy | 5 | JSON | EN, ES |
| Legal | 3 | JSON | EN |
| Runbook | 4 | JSONL | EN |
| Incident | 4 | JSONL | EN |
| Finance | 5 | JSON, CSV | EN |

## Benchmark Results Summary

### Format-Aware Benchmark (Latest Run)

| Model | Lat(ms) | QPS | MD | JSON | JSONL | CSV | API | HR | Legal | Runbook | Incident | Finance | EN | ES | CA |
|-------|---------|-----|-----|------|-------|-----|-----|----|-------|---------|----------|---------|----|----|----|
| **gte-base** | **241** | **4.2** | **100%** | **93%** | **100%** | **100%** | 0% | **100%** | **100%** | **100%** | **100%** | **100%** | 82% | **100%** | **100%** |
| jina-embeddings-v2-base-es | 365 | 2.7 | 100% | **100%** | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 82% | 100% | 100% |
| gte-large | 739 | 1.4 | 100% | 87% | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 82% | 100% | 100% |
| mxbai-embed-large-v1 | 757 | 1.3 | 100% | 87% | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 82% | 100% | 100% |
| intfloat/multilingual-e5-large | 1469 | 0.7 | 100% | 93% | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 82% | 100% | 100% |
| Qwen3-Embedding-0.6B | 2216 | 0.5 | 80% | 100% | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 79% | 100% | 100% |
| paraphrase-multilingual-MiniLM | 397 | 2.5 | 80% | 87% | 100% | 100% | 0% | 60% | 67% | 100% | 100% | 100% | 71% | 50% | 100% |

### Key Findings

1. **gte-base** = Best overall balance (speed + quality + format coverage)
2. **jina-embeddings-v2-base-es** = Best JSON recall (100%), optimized for Spanish
3. **All models** achieve 100% on JSONL, CSV, Runbook, Incident, Finance
4. **API Spec recall = 0%** for all models - OpenAPI spec needs better chunking/queries
5. **Spanish/Catalan** recall = 100% for most models (better than English!)
6. **Speed vs Quality tradeoff**: gte-base (241ms) vs intfloat/e5-large (1469ms)

## Quick Benchmark Results

| Model | Lat(ms) | QPS | R@5H | nDCG@5H | Judge | CQPS4 |
|-------|---------|-----|------|---------|-------|-------|
| embeddinggemma-2-Q8_0 | 21 | 46.8 | 80% | 0.914 | 2.93 | 104 |
| gte-base | 233 | 4.3 | 100% | 0.879 | 3.00 | 9.2 |
| intfloat/multilingual-e5-large | 1370 | 0.7 | 80% | 0.867 | 2.33 | 0.7 |

**Winner**: embeddinggemma-2-Q8_0 (llama.cpp) - 10-50x faster than fastembed models

## Metrics Measured

### Retrieval Metrics
- **Recall@K** (K=1,3,5): Binary hit/miss for expected file
- **MRR@10**: Mean Reciprocal Rank
- **nDCG@5/10**: Normalized Discounted Cumulative Gain (graded relevance)
- **Cosine Similarity**: Query↔chunk semantic similarity

### Format-Specific Metrics
- **Recall by Format**: MD, JSON, JSONL, CSV
- **Recall by Category**: Company, API, HR, Legal, Runbook, Incident, Finance
- **Recall by Language**: EN, ES, CA

### Semantic Metrics
- **Cross-lingual Consistency**: Cosine sim between EN/ES/CA query embeddings
- **Language Alignment**: Pairwise ES-CA, ES-EN, CA-EN similarity

### End-to-End Metrics
- **LLM Judge Score (1-5)**: LLM-as-judge evaluation of generated answers
- **Judge Distribution**: Count of scores 1-5

### Performance Metrics
- **Index Time**: Seconds to embed + index all chunks
- **Index RAM**: Peak RAM during indexing
- **Query Latency**: Avg/P50/P95 ms (embed + search)
- **Throughput**: QPS (single-thread)
- **Concurrent QPS**: 4 & 8 workers
- **Cold Start**: First query latency
- **Inference RAM**: RAM during querying
- **Index Size**: DB size on disk

### Cost Estimation
- **RAG Cost**: $/1K queries (embedding + LLM for 5 chunks × 200 tokens)
- **Full Context Cost**: $/1K queries (50K tokens)
- **Savings**: % vs full context

## Usage

### Quick Start (Recommended)
```bash
cd embedding-benchmark
./run_all.sh          # Full automated setup + benchmark (~1 hour)
# OR
./run_quick.sh        # Quick smoke test (~5 min)
# OR
./run_formats.sh      # Format-aware benchmark (~10 min)
```

### Manual Steps
```bash
# 1. Install system deps
brew install python@3.11 llama.cpp hf          # macOS
sudo apt install python3.11 python3.11-venv    # Linux

# 2. Create venv + install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download models
./run_all.sh  # Downloads all GGUF models automatically

# 4. Run benchmarks
python run_benchmark.py      # Full
python benchmark_quick.py    # Quick
python benchmark_formats.py  # Format-aware
```

## Customization

### Add Your Own Data
```bash
# Replace test_data_pro with your documents
mkdir -p test_data_pro/{docs,api,hr,legal,engineering,ops,finance}/{en,es,ca}
cp -r /your/project/docs/* test_data_pro/docs/
cp -r /your/project/api/* test_data_pro/api/
# ... etc
```

### Add Custom Queries
Edit `FORMAT_QUERIES` in `benchmark_formats.py`:
```python
{
    "query": "your specific question",
    "lang": "en",
    "expected_format": "json",  # md, json, jsonl, csv
    "expected_path_contains": "path/to/expected/file.json",
    "category": "your_category"
}
```

### Add Custom Models
```python
EMBEDDING_MODELS = [
    # ... existing ...
    {"name": "your-model", "type": "fastembed", "model_name": "org/model", "dim": 768},
    {"name": "your-gguf", "type": "llama_cpp", "gguf_path": "models/model.gguf", "server_port": 8085, "dim": 1024, "ctx": 8192},
]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `llama-server` 500 error | Use fastembed models only; GGUF embedding endpoint not fully supported |
| `sqlite-vec` not found | Use Homebrew Python 3.11+ on macOS |
| API Spec recall 0% | Improve OpenAPI chunking; add more specific queries |
| OOM on large models | Reduce `ctx` size; use Q4_K_M quantization |
| Windows | Use WSL2 Ubuntu |

## Requirements

### System
- **macOS**: Homebrew, Python 3.11+, llama.cpp, hf CLI
- **Linux**: apt/dnf, Python 3.11+, llama.cpp, hf CLI
- **Windows**: WSL2 Ubuntu

### Python (auto-installed)
```
fastembed>=0.3.0
sqlite-vec>=0.1.6
httpx>=0.27.0
psutil>=5.9.0
numpy>=1.24.0
```

## Output Files

| File | Description |
|------|-------------|
| `.orchestrator/benchmark_results.json` | Full benchmark results |
| `.orchestrator/benchmark_quick_results.json` | Quick benchmark results |
| `.orchestrator/benchmark_format_results.json` | Format-aware results |
| `.orchestrator/index/bench_*.db` | SQLite+vec+FTS5 indexes per model |

## License
MIT License - Free for commercial and personal use.