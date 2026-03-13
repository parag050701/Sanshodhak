#!/bin/bash
# Quick reference for RAG system usage

cat << 'EOF'
╔════════════════════════════════════════════════════════════════════════════╗
║                      🚀 RAG SYSTEM QUICK REFERENCE                         ║
╚════════════════════════════════════════════════════════════════════════════╝

📦 SYSTEMS AVAILABLE
────────────────────────────────────────────────────────────────────────────
1. ollama_rag.py       - Simple & fast (3,383 chunks)
2. simple_rag.py       - With transformer fallback
3. enhanced_rag.py     - Full pipeline (hybrid + rerank + eval)

🎯 QUICK START
────────────────────────────────────────────────────────────────────────────
# Interactive mode (recommended)
python interactive_rag.py

# Run comprehensive tests
python test_enhanced_rag.py

# Simple Ollama RAG
python ollama_rag.py

🔍 QUERY MODES
────────────────────────────────────────────────────────────────────────────
Dense only:   Fast, semantic search only
Hybrid:       Dense + BM25, better coverage
Full:         Hybrid + reranking, best quality

📊 EVALUATION
────────────────────────────────────────────────────────────────────────────
from enhanced_rag import EnhancedRAG, RAGEvaluator

rag = EnhancedRAG()
rag.load("enhanced_rag_index")

evaluator = RAGEvaluator(rag)
results = evaluator.full_evaluation(test_questions)

💻 PYTHON API
────────────────────────────────────────────────────────────────────────────
from enhanced_rag import EnhancedRAG

# Initialize with Ollama fallback
rag = EnhancedRAG()

# Load index
rag.load("enhanced_rag_index")

# Query (full pipeline)
answer, results = rag.query(
    "Your question here",
    use_hybrid=True,
    use_rerank=True,
    top_k=5
)

# Print results
print(f"Answer: {answer}")
for r in results:
    print(f"- {r['metadata']['file']}: {r['score']:.3f}")

🎛️ CONFIGURATION
────────────────────────────────────────────────────────────────────────────
# Ollama only (no PyTorch)
rag = EnhancedRAG(
    use_ollama_fallback=True,
    ollama_embed="bge-m3",
    ollama_llm="deepseek-r1:7b"
)

# SentenceTransformers (requires PyTorch ≥2.6)
rag = EnhancedRAG(
    embed_model="BAAI/bge-m3",
    rerank_model="BAAI/bge-reranker-v2-m3",
    use_ollama_fallback=False
)

🔧 INTERACTIVE MODE OPTIONS
────────────────────────────────────────────────────────────────────────────
Run: python interactive_rag.py

Commands:
  dense: <query>    - Use dense search only
  hybrid: <query>   - Use hybrid (dense + BM25)
  full: <query>     - Use full pipeline (default)
  quit              - Exit

📈 PERFORMANCE
────────────────────────────────────────────────────────────────────────────
Build time:    5-8 minutes (36 papers)
Query time:    0.5-1.2 seconds
Generation:    3-5 seconds per answer

Dense only:    ~0.5s
Hybrid:        ~0.8s
Full:          ~1.2s

📚 DOCUMENTATION
────────────────────────────────────────────────────────────────────────────
RAG_README.md      - Comprehensive guide
SUMMARY.md         - Implementation summary
This file          - Quick reference

🐛 TROUBLESHOOTING
────────────────────────────────────────────────────────────────────────────
Problem: PyTorch security error
Solution: Enhanced RAG auto-falls back to Ollama

Problem: Ollama connection error
Solution: Check Ollama is running
  curl http://localhost:11434/api/tags
  ollama list

Problem: Low retrieval scores
Solution: Add relevant_docs to test_dataset.json

📁 FILE STRUCTURE
────────────────────────────────────────────────────────────────────────────
paper-intel/
├── ollama_rag.py              ← Simple RAG
├── enhanced_rag.py            ← Full RAG + evaluator
├── interactive_rag.py         ← Interactive demo
├── test_enhanced_rag.py       ← Comprehensive tests
├── rag_index/                 ← Ollama index
├── enhanced_rag_index/        ← Enhanced index
├── ingestion/raw_text/        ← 36 papers
├── rag/evaluation/test_dataset.json
└── evaluation_results.json

✅ STATUS: PRODUCTION READY
────────────────────────────────────────────────────────────────────────────
Built: December 6, 2025
Papers: 36
Chunks: 3,383 (Ollama) / 514 (Enhanced)
Test questions: 27
Dependencies: Minimal (works with Ollama only)

EOF
