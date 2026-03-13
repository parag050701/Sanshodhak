# Sanshodhak v2 — Upgrade Guide

Adaptive Hybrid Retrieval (AHR) system with journal-level IR evaluation.

---

## Architecture

```
Query
  │
  ├─ Dense Retriever  (FAISS IndexFlatIP, BGE-M3 embeddings)
  ├─ Sparse Retriever (Okapi BM25)
  │       └─ RRF Fusion
  │               └─ Adaptive Graph Budget  ──► GraphIndex (1-hop / 2-hop / PPR / QBPR)
  │                                                  ├─ Vocabulary-Jaccard edges
  │                                                  ├─ Embedding-cosine edges
  │                                                  ├─ DOI co-citation edges
  │                                                  └─ Fuzzy title-match edges
  └─ [Optional] Cross-encoder Reranker
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For Oracle Free Tier (1 GB RAM), use a smaller embedding model:
```yaml
# config.yaml
embedding:
  model: "all-MiniLM-L6-v2"   # 80 MB, 384-dim
```

### 2. Start Ollama (required for /query endpoint)

```bash
# Install: https://ollama.ai
ollama pull bge-m3
ollama pull deepseek-r1:7b
ollama serve
```

---

## BEIR Evaluation (Part 1)

Evaluates 5 standard IR benchmarks with 4 systems.

```bash
# All 5 datasets (~30 min with bge-m3 on CPU)
python run_beir_eval.py

# Specific datasets (faster)
python run_beir_eval.py --datasets scifact nfcorpus

# Skip graph (Dense + Hybrid only)
python run_beir_eval.py --datasets scifact --skip-graph

# With cross-encoder reranker
python run_beir_eval.py --datasets scifact --reranker

# Custom expansion mode
python run_beir_eval.py --datasets nfcorpus --expansion-mode ppr
```

Results written to `beir_results/`:
- `summary.json` — metrics + Wilcoxon significance tests
- `summary.csv` — table for LaTeX / Excel import

### Expected results (approximate, CPU, BGE-M3)

| Dataset    | Dense NDCG@10 | Hybrid NDCG@10 | AHR NDCG@10 |
|------------|:-------------:|:--------------:|:-----------:|
| SciFact    | 0.63–0.68     | 0.65–0.70      | 0.66–0.72   |
| NFCorpus   | 0.30–0.33     | 0.31–0.34      | 0.32–0.35   |
| FiQA       | 0.34–0.38     | 0.35–0.39      | 0.35–0.40   |
| ArguAna    | 0.44–0.48     | 0.42–0.47      | 0.43–0.48   |
| TREC-COVID | 0.60–0.65     | 0.62–0.67      | graph skipped (>30k) |

---

## Scalability Analysis (Part 5)

```bash
# Default: 1k, 5k, 10k, 50k documents
python scalability_analysis.py

# Custom sizes
python scalability_analysis.py --sizes 1000 10000 100000

# Skip graph (fast mode)
python scalability_analysis.py --no-graph
```

Output: `scalability_results/`
- `scalability_results.json` — raw timings + memory
- `plots/build_time.png`
- `plots/memory.png`
- `plots/latency.png`
- `plots/graph_edges.png`

---

## API Server (Part 6)

### Start

```bash
# Development
uvicorn api.main:app --host 0.0.0.0 --port 8087 --reload

# Production (pre-built indices required)
INDEX_DIR=enhanced_rag_index python -m uvicorn api.main:app \
    --host 0.0.0.0 --port 8087 --workers 1
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Liveness probe |
| GET | /ready | Readiness probe (503 during startup) |
| GET | /stats | Index stats + graph topology |
| POST | /search | Retrieve top-k documents |
| POST | /query | Retrieve + generate answer |

### Example

```bash
# Search
curl -X POST http://localhost:8087/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "top_k": 5, "mode": "ahr"}'

# Query with generation
curl -X POST http://localhost:8087/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does self-attention work?", "top_k": 5}'
```

---

## Docker Deployment

```bash
# Build
docker build -t sanshodhak:latest .

# Run (mount pre-built index)
docker run -d \
  -p 8087:8087 \
  -v $(pwd)/enhanced_rag_index:/app/enhanced_rag_index \
  -e INDEX_DIR=/app/enhanced_rag_index \
  -e LOG_LEVEL=INFO \
  --name sanshodhak \
  sanshodhak:latest

# Check health
curl http://localhost:8087/health
```

---

## Module Reference (core/)

| Module | Key exports | Description |
|--------|-------------|-------------|
| `core/retrieval.py` | `EmbeddingBackend`, `BM25Retriever`, `DenseRetriever`, `HybridRetriever` | Dense + sparse retrieval |
| `core/graph.py` | `GraphIndex` | Knowledge graph with 4 edge types |
| `core/fusion.py` | `reciprocal_rank_fusion` | RRF fusion |
| `core/adaptive_budget.py` | `compute_graph_budget` | Dynamic graph slot allocation |
| `core/reranker.py` | `CrossEncoderReranker` | Cross-encoder reranking |
| `core/evaluation.py` | `evaluate_run`, `wilcoxon_test` | Metrics + significance tests |

---

## Configuration (config.yaml)

Key parameters:

```yaml
graph:
  alpha: 0.5          # Jaccard weight in edge formula
  beta: 0.5           # Embedding weight in edge formula
  expansion_mode: "1hop"   # 1hop | 2hop | ppr | qbpr

adaptive_budget:
  base_fraction: 0.25      # k/4 baseline graph slots

reranker:
  enabled: false           # Set true to enable (downloads 66 MB model)
```

---

## Reproducibility

All experiments are deterministic when using the same:
1. Embedding model version
2. BEIR dataset version (pinned by URL)
3. Random seed (42 for synthetic corpora)

Re-run with:
```bash
python run_beir_eval.py --datasets scifact nfcorpus fiqa arguana trec-covid
```
