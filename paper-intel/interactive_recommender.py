"""Interactive Resource Recommendation System."""

from resource_recommender import ResourceRecommender
from ollama_rag import OllamaRAG
import json


def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def display_detailed_recommendations(recommendations: dict):
    """Display recommendations in a user-friendly format."""
    
    print_header("📚 RELEVANT RESEARCH PAPERS")
    for paper in recommendations['relevant_papers']:
        print(f"\n[{paper['rank']}] {paper['file']}")
        print(f"    Relevance Score: {paper['score']:.3f}")
        print(f"    Snippet: {paper['snippet'][:150]}...")
    
    print_header("💻 GITHUB REPOSITORIES")
    if recommendations['github_repositories']:
        for i, repo in enumerate(recommendations['github_repositories'], 1):
            print(f"\n[{i}] {repo['name']}")
            print(f"    ⭐ Stars: {repo['stars']:,}")
            print(f"    🔧 Language: {repo['language']}")
            print(f"    📝 {repo['description']}")
            print(f"    🔗 {repo['url']}")
            if repo['topics']:
                print(f"    🏷️  Topics: {', '.join(repo['topics'][:5])}")
    else:
        print("  No GitHub repositories found.")
    
    print_header("🤗 HUGGINGFACE MODELS")
    if recommendations['huggingface']['models']:
        for i, model in enumerate(recommendations['huggingface']['models'], 1):
            print(f"\n[{i}] {model['name']}")
            print(f"    💙 Likes: {model['likes']:,} | ⬇️  Downloads: {model['downloads']:,}")
            print(f"    🎯 Task: {model['task']}")
            print(f"    📚 Library: {model['library']}")
            print(f"    🔗 {model['url']}")
    else:
        print("  No HuggingFace models found.")
    
    print_header("📊 HUGGINGFACE DATASETS")
    if recommendations['huggingface']['datasets']:
        for i, dataset in enumerate(recommendations['huggingface']['datasets'], 1):
            print(f"\n[{i}] {dataset['name']}")
            print(f"    💙 Likes: {dataset['likes']:,} | ⬇️  Downloads: {dataset['downloads']:,}")
            print(f"    🔗 {dataset['url']}")
    else:
        print("  No HuggingFace datasets found.")
    
    print_header("📄 PAPERS WITH CODE")
    if recommendations['papers_with_code']:
        for i, paper in enumerate(recommendations['papers_with_code'], 1):
            print(f"\n[{i}] {paper['title']}")
            if paper.get('arxiv_id'):
                print(f"    📝 ArXiv: {paper['arxiv_id']}")
            print(f"    🔗 {paper['paper_url']}")
            if paper['implementations']:
                print(f"    💻 Implementations ({len(paper['implementations'])}):")
                for impl in paper['implementations']:
                    stars = impl.get('stars', 0)
                    print(f"      • {impl['url']} ({impl['framework']}) - {stars}⭐")
    else:
        print("  No Papers with Code results found.")
    
    print_header("📚 TUTORIALS & COURSES")
    if recommendations['additional_resources']['tutorials']:
        print("\n📖 Tutorials:")
        for tutorial in recommendations['additional_resources']['tutorials']:
            print(f"  • {tutorial['title']}")
            print(f"    {tutorial['url']}")
    
    if recommendations['additional_resources']['courses']:
        print("\n🎓 Courses:")
        for course in recommendations['additional_resources']['courses'][:3]:
            print(f"  • {tutorial['title']}")
            print(f"    {course['url']}")
    
    if recommendations['additional_resources']['documentation']:
        print("\n📘 Documentation:")
        for doc in recommendations['additional_resources']['documentation']:
            print(f"  • {doc['framework'].upper()} - {doc['url']}")
    
    print_header("🔧 EXTRACTED TECHNICAL TERMS")
    terms = recommendations['technical_terms']
    for category, items in terms.items():
        if items:
            print(f"\n{category.capitalize()}:")
            print(f"  {', '.join(items[:10])}")


def interactive_mode():
    """Interactive resource recommendation."""
    
    print("="*80)
    print("  🔍 INTERACTIVE RESOURCE RECOMMENDER")
    print("="*80)
    
    # Load RAG system
    print("\n📦 Loading RAG system...")
    rag = OllamaRAG()
    rag.load("rag_index")
    print("✅ RAG system loaded")
    
    # Create recommender
    recommender = ResourceRecommender(rag)
    
    print("\n" + "="*80)
    print("Enter your research query to get recommendations")
    print("Type 'quit' to exit, 'save' to save last results")
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
            print("\n⏳ Searching for resources...")
            recommendations = recommender.recommend_resources(query, top_k_papers=3)
            
            # Display
            display_detailed_recommendations(recommendations)
            
            # Save for later
            last_recommendations = recommendations
            
            print("\n" + "="*80)
            print(f"✅ Found {recommendations['summary']['total_github_repos']} GitHub repos, "
                  f"{recommendations['summary']['total_hf_models']} HF models, "
                  f"{recommendations['summary']['pwc_papers']} papers with code")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def batch_mode(queries_file: str):
    """Batch process multiple queries."""
    
    print("="*80)
    print("  📋 BATCH RESOURCE RECOMMENDATION")
    print("="*80)
    
    # Load queries
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    
    print(f"\n📄 Loaded {len(queries)} queries")
    
    # Load RAG system
    print("\n📦 Loading RAG system...")
    rag = OllamaRAG()
    rag.load("rag_index")
    
    # Create recommender
    recommender = ResourceRecommender(rag)
    
    all_results = []
    
    for i, query_item in enumerate(queries, 1):
        query = query_item.get('query', query_item)
        
        print(f"\n[{i}/{len(queries)}] Processing: {query[:60]}...")
        
        try:
            recommendations = recommender.recommend_resources(query, top_k_papers=3)
            all_results.append(recommendations)
            
            # Save individual result
            filename = f"recommendations_{i:02d}.json"
            recommender.save_recommendations(recommendations, filename)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results.append({
                'query': query,
                'error': str(e)
            })
    
    # Save combined results
    with open("all_recommendations.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Batch processing complete!")
    print(f"💾 All results saved to all_recommendations.json")


def demo_mode():
    """Run demo with preset queries."""
    
    print("="*80)
    print("  🎬 DEMO MODE")
    print("="*80)
    
    # Load RAG system
    print("\n📦 Loading RAG system...")
    rag = OllamaRAG()
    rag.load("rag_index")
    
    # Create recommender
    recommender = ResourceRecommender(rag)
    
    # Demo queries
    demo_queries = [
        "Retrieval-Augmented Generation with hybrid search",
        "Neural machine translation models",
        "Database query optimization using deep learning"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"DEMO {i}/{len(demo_queries)}")
        print(f"{'='*80}")
        
        recommendations = recommender.recommend_resources(query, top_k_papers=3)
        display_detailed_recommendations(recommendations)
        
        # Save
        filename = f"demo_recommendations_{i}.json"
        recommender.save_recommendations(recommendations, filename)
        
        input("\n⏸️  Press Enter to continue...")
    
    print(f"\n✅ Demo complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "batch" and len(sys.argv) > 2:
            batch_mode(sys.argv[2])
        elif mode == "demo":
            demo_mode()
        else:
            print("Usage:")
            print("  python interactive_recommender.py              # Interactive mode")
            print("  python interactive_recommender.py demo         # Demo mode")
            print("  python interactive_recommender.py batch <file> # Batch mode")
    else:
        interactive_mode()
