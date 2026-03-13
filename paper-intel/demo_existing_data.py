#!/usr/bin/env python3
"""
✅ DEMO WITH EXISTING DATA - INSTANT RESULTS
============================================

Uses the 4 PDFs already downloaded and indexed from "Transformers" topic.
Shows RAG querying, answer generation, and resource recommendations.

Perfect for presentations - runs in 30 seconds!
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ollama_rag import OllamaRAG
from resource_recommender import ResourceRecommender


def demo():
    print("\n" + "█"*80)
    print("█" + " "*20 + "SANSHODHAK RAG DEMO - EXISTING DATA" + " "*25 + "█")
    print("█"*80)
    
    # Check if data exists
    index_dir = Path("research_papers/rag_index")
    text_dir = Path("research_papers/extracted_text")
    pdfs = list(Path("research_papers").glob("*.pdf"))
    
    if not index_dir.exists() or not (index_dir / "faiss.index").exists():
        print("\n❌ No RAG index found!")
        print("\nPlease run the pipeline first:")
        print("  ./RUN.sh")
        print("\nOr use existing data:")
        print("  Topic: 'Transformers' (already has 4 PDFs)")
        return
    
    print(f"\n✅ Found existing data:")
    print(f"   • PDFs: {len(pdfs)}")
    print(f"   • Texts: {len(list(text_dir.glob('*.txt')))}")
    print(f"   • Index: {index_dir}")
    
    # Show PDFs
    print(f"\n📄 Papers in index:")
    for pdf in pdfs:
        size = pdf.stat().st_size / 1024 / 1024
        print(f"   • {pdf.name} ({size:.1f} MB)")
    
    # Load RAG system
    print("\n" + "="*80)
    print("LOADING RAG SYSTEM")
    print("="*80)
    
    print("\n🤖 Loading Ollama RAG from disk...")
    rag = OllamaRAG(embed_model='bge-m3', llm_model='qwen3:4b', use_openrouter=True)
    rag.load(str(index_dir))
    
    print(f"✅ Loaded: {rag.index.ntotal:,} chunks indexed")
    
    # Interactive Q&A with Resource Recommendations
    print("\n" + "="*80)
    print("INTERACTIVE Q&A + RESOURCE RECOMMENDATIONS")
    print("="*80)
    
    questions = [
        "What are transformers in machine learning?",
        "What are the main applications of transformers?",
        "How do transformers compare to other neural networks?",
        "What are the key innovations in transformer architecture?"
    ]
    
    print("\n📝 Sample questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    
    print("\n💡 Commands:")
    print("  • Type your question")
    print("  • Type a number (1-4) for sample questions")
    print("  • Type 'resources <topic>' for recommendations")
    print("  • Type 'quit' to exit")
    
    recommender = ResourceRecommender(rag)
    
    while True:
        print("\n" + "-"*80)
        query = input("\n💬 Your input: ").strip()
        
        if query.lower() in ['quit', 'q', 'exit']:
            break
        
        if not query:
            continue
        
        # Check for resource recommendation request
        if query.lower().startswith('resources'):
            parts = query.split(maxsplit=1)
            topic = parts[1] if len(parts) > 1 else "Deep Learning"
            
            print(f"\n🔍 Finding resources for: {topic}")
            resources = recommender.recommend_resources(topic)
            
            print("\n" + "="*80)
            print(f"🎯 RESOURCE RECOMMENDATIONS: {topic}")
            print("="*80)
            
            if resources.get('huggingface'):
                print(f"\n🤗 HuggingFace ({len(resources['huggingface'])} items):")
                for item in resources['huggingface'][:5]:
                    print(f"   • {item['name']}")
                    print(f"     {item['url']}")
            
            if resources.get('github'):
                print(f"\n💻 GitHub ({len(resources['github'])} repos):")
                for repo in resources['github'][:5]:
                    stars = repo.get('stars', 0)
                    print(f"   • {repo['name']} ⭐ {stars:,}")
                    print(f"     {repo['url']}")
            
            if resources.get('web'):
                print(f"\n🌐 Web ({len(resources['web'])} links):")
                for link in resources['web'][:5]:
                    print(f"   • {link['title']}")
                    print(f"     {link['url']}")
            
            total = sum(len(v) for v in resources.values())
            print(f"\n✅ Found {total} resources")
            continue
        
        # Check if it's a number (sample question)
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(questions):
                query = questions[idx]
                print(f"   → {query}")
            else:
                print("   Invalid number")
                continue
        
        print("\n🔍 Searching...")
        answer, sources = rag.query(query, top_k=5)
        
        print("\n" + "─"*80)
        print("📝 ANSWER:")
        print("─"*80)
        
        # Show answer (max 500 chars)
        display_answer = answer[:500] + "..." if len(answer) > 500 else answer
        for line in display_answer.split('\n'):
            print(f"  {line}")
        
        print("─"*80)
        
        print(f"\n📚 Sources ({len(sources)} chunks):")
        for i, src in enumerate(sources[:3], 1):
            print(f"  [{i}] {src['metadata']['file']}")
            print(f"      Score: {src['score']:.4f}")
            print(f"      Text: {src['text'][:100]}...")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    
    pdfs = list(Path("research_papers").glob("*.pdf"))
    print("\n📊 Summary:")
    print(f"   • Papers indexed: {len(pdfs)}")
    print(f"   • Chunks searched: {rag.index.ntotal:,}")
    print(f"   • Questions answered: Interactive")
    
    print("\n🎉 RAG system is working perfectly!")


if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
