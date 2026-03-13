# Research-Grade RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system with advanced features and complete evaluation framework.

## Features

### 🎯 Advanced RAG Pipeline
- **Hybrid Search**: Dense (BGE-M3) + Sparse (BM25) retrieval
- **Query Processing**:
  - Query expansion (generate alternatives)
  - Query rewriting (optimize for retrieval)
  - HyDE (Hypothetical Document Embeddings)
- **Reranking**: BGE-reranker-v2-m3 for improved relevance
- **Diversity**: MMR (Maximal Marginal Relevance) for result diversity
- **Semantic Chunking**: Sentence-aware chunking with overlap

### 📊 Comprehensive Evaluation

#### Retrieval Metrics
- **Precision@K**: Proportion of retrieved docs that are relevant
- **Recall@K**: Proportion of relevant docs that are retrieved
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision

#### Generation Metrics
- **Faithfulness**: Are answers grounded in context?
- **Answer Relevance**: Does answer match question?
- **Context Relevance**: Is retrieved context relevant?
- **BLEU**: N-gram overlap with reference
- **ROUGE**: Recall-oriented overlap metrics
- **BERTScore**: Semantic similarity using embeddings

#### End-to-End Metrics
- **Answer Correctness**: Combined lexical + semantic similarity
- **Hallucination Rate**: Proportion not grounded in context
- **Citation Accuracy**: Are citations valid?

#### Performance Metrics
- **Latency**: Mean, median, P95, P99
- **Throughput**: Queries per second
- **Token Efficiency**: Input/output token usage

## Installation

```bash
# Install dependencies
pip install -r rag/requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

### 1. Build RAG Index

```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel
python rag/scripts/build_index.py
```

This will:
- Load 36 research papers from `ingestion/raw_text/`
- Chunk documents semantically
- Generate BGE-M3 embeddings
- Build FAISS vector database
- Save index to `rag/index/`

### 2. Generate Test Dataset

```bash
python rag/evaluation/generate_test_data.py
```

Generates evaluation questions from papers with reference answers.

### 3. Run Evaluation

```bash
python rag/scripts/run_evaluation.py
```

Runs complete evaluation suite and generates:
- `rag/evaluation/results.json` - Summary metrics
- `rag/evaluation/detailed_results.json` - Full test case results

### 4. Query System

```bash
python rag/scripts/query_rag.py
```

Interactive interface for querying the RAG system.

## Configuration

Edit `rag/config/rag_config.yaml` to customize:

```yaml
embeddings:
  dense:
    model: "BAAI/bge-m3"
    device: "cuda"

retrieval:
  hybrid:
    enabled: true
    dense_weight: 0.7
    sparse_weight: 0.3
  
  reranking:
    enabled: true
    model: "BAAI/bge-reranker-v2-m3"
    top_k: 20
  
  diversity:
    enabled: true
    method: "mmr"
    lambda_param: 0.7
```

## Architecture

```
┌─────────────┐
│    Query    │
└──────┬──────┘
       │
       ├─ Query Rewrite
       ├─ Query Expansion
       └─ HyDE Generation
       │
       v
┌─────────────────┐
│ Hybrid Retrieval│
├─────────────────┤
│ Dense (BGE-M3)  │ ← 70% weight
│ Sparse (BM25)   │ ← 30% weight
│ RRF Fusion      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Reranking     │ ← BGE-reranker-v2-m3
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Diversity      │ ← MMR
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Generation    │ ← LLM with context
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Answer + Cites │
└─────────────────┘
```

## Advanced Usage

### Programmatic API

```python
from rag.core.pipeline import AdvancedRAG
import asyncio

# Initialize
rag = AdvancedRAG(config_path="rag/config/rag_config.yaml")
rag.load_index("rag/index")

# Query with all features
async def query():
    result = await rag.query(
        "What are the main approaches to prompt engineering?",
        use_expansion=True,
        use_rewrite=True,
        use_hyde=True,
        use_reranking=True,
        use_diversity=True
    )
    
    print(result.answer)
    for ctx in result.contexts:
        print(f"Paper: {ctx['paper_id']}, Score: {ctx['score']}")

asyncio.run(query())
```

### Custom Evaluation

```python
from rag.evaluation.metrics import RAGEvaluator

evaluator = RAGEvaluator()

# Retrieval metrics
precision = evaluator.precision_at_k(retrieved, relevant, k=10)
mrr = evaluator.mean_reciprocal_rank([retrieved], [relevant])

# Generation metrics
faithfulness = evaluator.faithfulness(answer, contexts)
correctness = evaluator.answer_correctness(answer, reference)
```

## Components

### Core Modules
- `embeddings.py` - Dense (BGE-M3) and sparse (BM25) embedders
- `chunking.py` - Semantic chunking with overlap
- `vectordb.py` - FAISS vector database with GPU support
- `reranking.py` - BGE reranker and MMR diversity
- `query_processing.py` - Query expansion, rewriting, HyDE, RRF fusion
- `generation.py` - LLM answer generation with citations
- `pipeline.py` - End-to-end RAG pipeline

### Evaluation
- `metrics.py` - Comprehensive evaluation metrics
- `generate_test_data.py` - Test dataset generation

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for embeddings and reranking
2. **Batch Size**: Adjust in config for memory/speed trade-off
3. **Index Type**: Use `IVF` for large datasets (>100K chunks)
4. **Chunking**: Tune `chunk_size` and `overlap` for your domain
5. **Weights**: Adjust `dense_weight`/`sparse_weight` for hybrid search

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM
- ~5GB disk space for models
