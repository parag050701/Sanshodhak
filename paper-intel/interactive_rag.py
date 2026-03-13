#!/usr/bin/env python3
"""Interactive RAG comparison and demo."""

from enhanced_rag import EnhancedRAG
from ollama_rag import OllamaRAG
import time

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def interactive_mode():
    """Interactive query mode."""
    print_header("🎯 INTERACTIVE RAG DEMO")
    
    print("\n📦 Loading Enhanced RAG...")
    rag = EnhancedRAG()
    
    try:
        rag.load("enhanced_rag_index")
        print("✅ Index loaded")
    except:
        print("⚠️  Index not found, building...")
        rag.build_index("ingestion/raw_text")
        rag.save("enhanced_rag_index")
    
    print("\n" + "="*80)
    print("📝 INSTRUCTIONS:")
    print("  - Enter your question")
    print("  - Prefix with 'dense:', 'hybrid:', or 'full:' to select mode")
    print("  - Default mode: full (hybrid + reranking)")
    print("  - Type 'quit' to exit")
    print("="*80)
    
    while True:
        print("\n" + "-"*80)
        query = input("❓ Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Parse mode
        use_hybrid = True
        use_rerank = True
        mode_name = "Full Pipeline (Hybrid + Reranking)"
        
        if query.lower().startswith("dense:"):
            use_hybrid = False
            use_rerank = False
            query = query[6:].strip()
            mode_name = "Dense Only"
        elif query.lower().startswith("hybrid:"):
            use_hybrid = True
            use_rerank = False
            query = query[7:].strip()
            mode_name = "Hybrid (Dense + BM25)"
        elif query.lower().startswith("full:"):
            query = query[5:].strip()
        
        print(f"\n🔍 Mode: {mode_name}")
        print("-"*80)
        
        # Query
        try:
            start = time.time()
            answer, results = rag.query(
                query,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                top_k=5,
                verbose=False
            )
            elapsed = time.time() - start
            
            print(f"\n📊 Top {len(results)} Results:")
            for i, r in enumerate(results):
                print(f"\n[{i+1}] 📄 {r['metadata']['file']}")
                print(f"    Score: {r['score']:.4f}")
                if 'scores' in r:
                    print(f"    Details: Dense={r['scores']['dense_score']:.3f}, "
                          f"BM25={r['scores'].get('bm25_score', 0):.3f}, "
                          f"RRF={r['scores'].get('rrf_score', 0):.3f}")
                print(f"    Text: {r['text'][:150]}...")
            
            print(f"\n✨ ANSWER:")
            print("-"*80)
            print(answer)
            print("-"*80)
            print(f"⏱️  Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def benchmark_mode():
    """Benchmark different configurations."""
    print_header("⚡ PERFORMANCE BENCHMARK")
    
    print("\n📦 Loading RAG...")
    rag = EnhancedRAG()
    rag.load("enhanced_rag_index")
    
    test_queries = [
        "What machine learning techniques are discussed?",
        "What datasets were used in the experiments?",
        "What are the main research contributions?"
    ]
    
    configs = [
        ("Dense Only", {"use_hybrid": False, "use_rerank": False}),
        ("Hybrid", {"use_hybrid": True, "use_rerank": False}),
        ("Full", {"use_hybrid": True, "use_rerank": True})
    ]
    
    results_table = []
    
    for query in test_queries:
        print(f"\n📝 Query: {query[:60]}...")
        row = {"query": query}
        
        for config_name, config in configs:
            start = time.time()
            answer, results = rag.query(query, verbose=False, top_k=5, **config)
            elapsed = time.time() - start
            
            row[config_name] = elapsed
            print(f"  {config_name:20s}: {elapsed:.2f}s")
        
        results_table.append(row)
    
    print("\n" + "="*80)
    print("📊 SUMMARY (Average Times)")
    print("="*80)
    
    avg_times = {name: 0 for name, _ in configs}
    for row in results_table:
        for name, _ in configs:
            avg_times[name] += row[name]
    
    for name, total in avg_times.items():
        avg = total / len(results_table)
        print(f"  {name:20s}: {avg:.2f}s")

def comparison_mode():
    """Compare Ollama RAG vs Enhanced RAG."""
    print_header("⚖️  SYSTEM COMPARISON")
    
    query = "What machine learning techniques are discussed in these papers?"
    
    print(f"\n📝 Query: {query}")
    print("\n" + "-"*80)
    
    # Ollama RAG
    print("\n1️⃣  OLLAMA RAG (Simple)")
    print("-"*40)
    try:
        rag_ollama = OllamaRAG()
        rag_ollama.load("rag_index")
        
        start = time.time()
        answer, results = rag_ollama.query(query, top_k=3, verbose=False)
        elapsed = time.time() - start
        
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Results: {len(results)}")
        print(f"💬 Answer: {answer[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Enhanced RAG
    print("\n2️⃣  ENHANCED RAG (Full Pipeline)")
    print("-"*40)
    try:
        rag_enhanced = EnhancedRAG()
        rag_enhanced.load("enhanced_rag_index")
        
        start = time.time()
        answer, results = rag_enhanced.query(
            query, 
            top_k=3, 
            use_hybrid=True, 
            use_rerank=True,
            verbose=False
        )
        elapsed = time.time() - start
        
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Results: {len(results)}")
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r['metadata']['file']}: {r['score']:.3f}")
        print(f"💬 Answer: {answer[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    import sys
    
    print("\n" + "="*80)
    print("  🚀 RAG SYSTEM DEMO")
    print("="*80)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nSelect mode:")
        print("  1. Interactive (query by query)")
        print("  2. Benchmark (performance comparison)")
        print("  3. Comparison (Ollama vs Enhanced)")
        choice = input("\nChoice (1/2/3) [1]: ").strip() or "1"
        mode = {"1": "interactive", "2": "benchmark", "3": "comparison"}.get(choice, "interactive")
    
    if mode == "benchmark":
        benchmark_mode()
    elif mode == "comparison":
        comparison_mode()
    else:
        interactive_mode()
    
    print("\n" + "="*80)
    print("✅ Demo complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
