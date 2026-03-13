#!/usr/bin/env python3
"""Interactive Resource Recommendation Interface"""

from resource_recommender_v2 import EnhancedResourceRecommender
from ollama_rag import OllamaRAG
import json
import sys


def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def interactive_mode():
    """Interactive resource search."""
    
    print_header("🔍 INTERACTIVE RESOURCE RECOMMENDER")
    
    # Load RAG
    print("\n📦 Loading RAG system...")
    rag = OllamaRAG()
    rag.load("rag_index")
    print("✅ RAG loaded")
    
    # Create recommender
    print("\n🤖 Initializing recommender...")
    recommender = EnhancedResourceRecommender(rag, llm_api="ollama")
    print("✅ Recommender ready")
    
    print("\n" + "="*80)
    print("Enter your research query to get recommendations")
    print("Commands: 'quit' to exit, 'save' to save last results")
    print("="*80)
    
    last_recommendations = None
    
    while True:
        print("\n" + "-"*80)
        query = input("🔍 Research Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if query.lower() == 'save' and last_recommendations:
            filename = input("💾 Filename (default: recommendations.json): ").strip()
            if not filename:
                filename = "recommendations.json"
            if not filename.endswith('.json'):
                filename += '.json'
            recommender.save_recommendations(last_recommendations, filename)
            continue
        
        if not query:
            continue
        
        try:
            # Get recommendations
            recommendations = recommender.recommend_resources(query, top_k_papers=3)
            
            # Display
            recommender.print_recommendations(recommendations)
            
            # Save for later
            last_recommendations = recommendations
            
        except KeyboardInterrupt:
            print("\n\n⏸️  Interrupted")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def batch_mode(queries_file: str):
    """Batch process multiple queries."""
    
    print_header("📋 BATCH MODE")
    
    # Load queries
    with open(queries_file, 'r') as f:
        data = json.load(f)
    
    queries = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                queries.append(item.get('query', str(item)))
            else:
                queries.append(str(item))
    else:
        queries = [data]
    
    print(f"\n📄 Loaded {len(queries)} queries")
    
    # Load RAG
    print("\n📦 Loading RAG system...")
    rag = OllamaRAG()
    rag.load("rag_index")
    
    # Create recommender
    recommender = EnhancedResourceRecommender(rag, llm_api="ollama")
    
    all_results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(queries)}] {query[:60]}...")
        print('='*80)
        
        try:
            recommendations = recommender.recommend_resources(query, top_k_papers=3)
            all_results.append(recommendations)
            
            # Save individual result
            filename = f"batch_recommendations_{i:02d}.json"
            recommender.save_recommendations(recommendations, filename)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results.append({
                'query': query,
                'error': str(e)
            })
    
    # Save combined results
    with open("all_batch_recommendations.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Batch processing complete!")
    print(f"💾 All results saved to all_batch_recommendations.json")


def demo_mode():
    """Run demo with preset queries."""
    
    print_header("🎬 DEMO MODE")
    
    # Load RAG
    print("\n📦 Loading RAG system...")
    rag = OllamaRAG()
    rag.load("rag_index")
    
    # Create recommender
    recommender = EnhancedResourceRecommender(rag, llm_api="ollama")
    
    # Demo queries
    demo_queries = [
        "Retrieval-Augmented Generation with hybrid search",
        "Transformer models for semantic embeddings",
        "Vector databases for ML applications"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"DEMO {i}/{len(demo_queries)}")
        print(f"{'='*80}")
        
        recommendations = recommender.recommend_resources(query, top_k_papers=3)
        recommender.print_recommendations(recommendations)
        
        # Save
        filename = f"demo_{i}.json"
        recommender.save_recommendations(recommendations, filename)
        
        if i < len(demo_queries):
            input("\n⏸️  Press Enter to continue...")
    
    print(f"\n✅ Demo complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "batch" and len(sys.argv) > 2:
            batch_mode(sys.argv[2])
        elif mode == "demo":
            demo_mode()
        else:
            print("Usage:")
            print("  python interactive.py              # Interactive mode")
            print("  python interactive.py demo         # Demo mode")
            print("  python interactive.py batch <file> # Batch mode")
    else:
        interactive_mode()
