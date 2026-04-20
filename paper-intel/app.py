"""
Flask Web Application for Research Assistant
Integrates: Paper Search → RAG → Resource Recommendations
"""

# Must be set before numpy/torch/OpenBLAS are imported to prevent
# threading deadlocks on Windows when multiple models are loaded.
import os as _os
import sys as _sys
import io as _io
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Disable Hugging Face fast-tokenizer parallelism — avoids deadlocks
# when encode() is called from the main thread of a multi-process server.
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Force UTF-8 stdout/stderr on Windows so emoji in print() don't crash.
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import time
import os
import sys
import secrets
from datetime import datetime
from typing import Dict, List

# Set working directory to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(SCRIPT_DIR, '.env'))

# Import our components
from graph_rag import GraphRAG
from resource_recommender_v2 import EnhancedResourceRecommender

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# Global instances
rag_system = None
recommender = None


def _normalize_scores(results: list) -> list:
    """
    Normalize raw retrieval/rerank scores to a 0–100 relevance scale.

    Cross-encoder logits are unbounded (e.g. -10 to +10).  RRF scores are
    tiny positives (~0.016).  Both are unreadable as-is.  Min-max normalize
    the batch so the best result = 100 and the worst = 0.
    """
    raw = [r.get('rerank_score', r.get('score', 0.0)) for r in results]
    lo, hi = min(raw, default=0.0), max(raw, default=1.0)
    span = hi - lo if hi != lo else 1.0
    for r, s in zip(results, raw):
        r['_display_score'] = round((s - lo) / span * 100, 1)
    return results


@app.route('/')
def index():
    """Main research assistant interface."""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search_papers():
    """Search research papers."""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        start_time = time.time()
        
        # Search papers
        results = _normalize_scores(rag_system.search(query, top_k=top_k))

        response = {
            'query': query,
            'results': [
                {
                    'rank': i + 1,
                    'file': r['metadata']['file'],
                    'score': r['_display_score'],
                    'snippet': r['text'][:300] + '...',
                    'full_text': r['text']
                }
                for i, r in enumerate(results)
            ],
            'count': len(results),
            'processing_time': round(time.time() - start_time, 3),
            'graph_stats': rag_system.get_graph_stats(),
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query_rag():
    """Generate answer using RAG."""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        start_time = time.time()
        
        # Get search results
        search_results = _normalize_scores(rag_system.search(query, top_k=5))

        # Generate answer
        answer, _ = rag_system.query(query)

        response = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'file': r['metadata']['file'],
                    'score': r['_display_score'],
                    'snippet': r['text'][:200]
                }
                for r in search_results[:3]
            ],
            'processing_time': round(time.time() - start_time, 3),
            'graph_stats': rag_system.get_graph_stats(),
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get resource recommendations."""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        start_time = time.time()
        
        # Get recommendations
        recs = recommender.recommend_resources(query, top_k_papers=3)

        response = {
            'query': query,
            'recommendations': {
                'papers': recs['relevant_papers'][:3],
                'huggingface_models': recs['huggingface']['models'][:5],
                'huggingface_datasets': recs['huggingface']['datasets'][:5],
                'github_repos': recs['github_repositories'][:5],
                'web_resources': recs['web_resources'][:5],
                'keywords': recs['search_keywords']
            },
            'summary': recs['summary'],
            'processing_time': round(time.time() - start_time, 3),
        }
        
        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/full_pipeline', methods=['POST'])
def full_pipeline():
    """Complete pipeline: Search → RAG → Recommendations."""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        start_time = time.time()
        
        # Step 1: Search papers
        search_results = _normalize_scores(rag_system.search(query, top_k=5))

        # Step 2: Generate answer
        answer, _ = rag_system.query(query)

        # Step 3: Get recommendations
        recs = recommender.recommend_resources(query, top_k_papers=3)

        response = {
            'query': query,
            'pipeline': {
                'search': {
                    'results': [
                        {
                            'rank': i + 1,
                            'file': r['metadata']['file'],
                            'score': r['_display_score'],
                            'snippet': r['text'][:200]
                        }
                        for i, r in enumerate(search_results[:5])
                    ],
                    'count': len(search_results)
                },
                'answer': {
                    'text': answer,
                    'sources': [r['metadata']['file'] for r in search_results[:3]]
                },
                'recommendations': {
                    'huggingface_models': recs['huggingface']['models'][:3],
                    'github_repos': recs['github_repositories'][:3],
                    'web_resources': recs['web_resources'][:3]
                }
            },
            'total_processing_time': round(time.time() - start_time, 3),
            'graph_stats': rag_system.get_graph_stats(),
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history():
    """Get historical metrics (from session)."""
    history = session.get('metrics_history', [])
    return jsonify({'history': history[-10:]})  # Last 10 queries


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'rag_loaded': rag_system is not None,
        'recommender_loaded': recommender is not None,
        'timestamp': datetime.now().isoformat()
    })


def initialize_systems():
    """Initialize RAG and recommender systems."""
    global rag_system, recommender

    print("Initializing Research Assistant...")

    # Load GraphRAG system (HGR: BM25 + FAISS + KG via RRF with slot reservation)
    print("Loading GraphRAG system...")
    rag_system = GraphRAG(
        similarity_threshold=0.30,
        use_bm25=True,
        use_adaptive_budget=True,
        expansion_mode="qbpr",
        max_graph_hops=2,
    )
    rag_system.load('rag_index')
    print("GraphRAG loaded")

    # Warm up cross-encoder reranker at startup (avoids cold-start on first request)
    print("Warming up cross-encoder reranker...")
    reranker = rag_system._get_reranker()
    if reranker.available:
        print("Cross-encoder reranker ready")
    else:
        print("Cross-encoder reranker unavailable (sentence-transformers not installed) — skipping")

    # Create recommender
    print("Initializing resource recommender...")
    # Prefer OpenRouter when Ollama is unavailable; falls back gracefully either way
    import requests as _req
    _ollama_up = False
    try:
        _ollama_up = _req.get("http://localhost:11434/api/tags", timeout=3).ok
    except Exception:
        pass
    _llm_api = "ollama" if _ollama_up else "openrouter"
    recommender = EnhancedResourceRecommender(rag_system, llm_api=_llm_api)
    print("Resource recommender ready")

    stats = rag_system.get_graph_stats()
    print("\n" + "="*60)
    print("  RESEARCH ASSISTANT READY  (GraphRAG / HGR)")
    print("="*60)
    print(f"  Server:      http://localhost:5000")
    print(f"  Chunks:      {len(rag_system.chunks)}")
    print(f"  Graph nodes: {stats.get('nodes', 0)}")
    print(f"  Graph edges: {stats.get('edges', 0)}  (tau=0.30)")
    print("="*60 + "\n")


if __name__ == '__main__':
    initialize_systems()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
