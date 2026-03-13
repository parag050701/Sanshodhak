#!/usr/bin/env python3
"""
🚀 QUICK DEMO - Complete RAG + Resources in 2-3 minutes
Uses existing indexed data, no downloading needed!
"""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent))

from ollama_rag import OllamaRAG
from resource_recommender import ResourceRecommender


def print_header(title):
    """Print fancy header."""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)


def demo():
    """Run quick demo."""
    print("\n" + "█"*80)
    print("█" + " "*25 + "SANSHODHAK DEMO" + " "*40 + "█")
    print("█" + " "*20 + "Quick RAG + Resources Demo" + " "*33 + "█")
    print("█"*80)
    
    print("\n⚡ This demo uses EXISTING indexed papers (no download needed)")
    print("⏱️  Estimated time: 2-3 minutes")
    print()
    
    # Check if index exists
    index_dir = Path("research_papers/rag_index")
    if not index_dir.exists() or not (index_dir / "faiss.index").exists():
        print("❌ No index found! Please run the full pipeline first:")
        print("   ./DEMO_LAUNCHER.sh")
        sys.exit(1)
    
    # Set OpenRouter API key
    os.environ["OPENROUTER_API_KEY"] = "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"
    
    # ===== STEP 1: Load RAG System =====
    print_header("STEP 1/4: Load RAG System")
    print("📦 Loading indexed papers from research_papers/rag_index...")
    
    rag = OllamaRAG(embed_model='bge-m3', llm_model='qwen3:4b', use_openrouter=True)
    rag.load(str(index_dir))
    
    print(f"✅ Loaded {rag.index.ntotal:,} chunks from {len(set(m['file'] for m in rag.metadata))} papers")
    
    # Show available topics
    print("\n📚 Papers available in index:")
    unique_files = sorted(set(m['file'] for m in rag.metadata))
    for i, f in enumerate(unique_files[:10], 1):
        print(f"   {i}. {f}")
    if len(unique_files) > 10:
        print(f"   ... and {len(unique_files)-10} more papers")
    
    print("\n💡 Topics covered: Deep Learning, Neural Networks, Transformers, LoRA,")
    print("   Graph Neural Networks, Language Models, Chain-of-Thought, etc.")
    
    # Initialize recommender
    recommender = ResourceRecommender(rag)
    
    # Show available topics
    print("\n📚 Papers available in index:")
    unique_files = sorted(set(m['file'] for m in rag.metadata))
    for i, f in enumerate(unique_files[:10], 1):
        print(f"   {i}. {f}")
    if len(unique_files) > 10:
        print(f"   ... and {len(unique_files)-10} more papers")
    
    print("\n💡 Topics covered: Deep Learning, Neural Networks, Transformers, LoRA,")
    print("   Graph Neural Networks, Language Models, Chain-of-Thought, etc.")
    
    # ===== STEP 2: Interactive Questions =====
    print_header("STEP 2/4: Ask Your Research Questions")
    
    print("\n💡 You can ask questions about:")
    print("   • What is LoRA?")
    print("   • Explain transformer architecture")
    print("   • What are graph neural networks?")
    print("   • How does chain-of-thought prompting work?")
    print("   • What are the main deep learning techniques?")
    print("\n💡 Type 'skip' to go to resource recommendations")
    print()
    
    question_count = 0
    while question_count < 3:  # Allow up to 3 questions
        try:
            question = input(f"❓ Question {question_count+1} (or 'skip'): ").strip()
            
            if not question or question.lower() == 'skip':
                print("⏭️  Skipping to resource recommendations...")
                break
            
            print(f"\n{'─'*80}")
            print(f"🔍 Searching indexed papers...")
            results = rag.search(question, top_k=5)
            
            print(f"📚 Found {len(results)} relevant chunks:")
            for j, r in enumerate(results[:3], 1):
                print(f"   [{j}] {r['metadata']['file']} (score: {r['score']:.3f})")
                print(f"       {r['text'][:100]}...")
            
            print(f"\n🤖 Generating answer with OpenRouter (Llama 3.2)...")
            context = "\n\n".join([r['text'] for r in results[:5]])
            answer = rag.generate(question, context)
            
            print(f"\n💬 Answer:")
            print(f"{'─'*80}")
            # Print full answer or up to 1500 chars
            print(answer[:1500] + ("...\n[Answer truncated - full answer too long]" if len(answer) > 1500 else ""))
            print(f"{'─'*80}")
            print()
            
            question_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⏭️  Moving to next section...")
            break
    
    # ===== STEP 3: Resource Recommendations =====
    print_header("STEP 3/4: Find Learning Resources")
    
    print("\n💡 I can find GitHub repos and HuggingFace resources for any topic!")
    print("💡 Examples: 'Deep Learning', 'Computer Vision', 'NLP', 'Reinforcement Learning'")
    print("💡 Type 'skip' to go to final interactive mode")
    print()
    
    resource_count = 0
    while resource_count < 2:  # Allow up to 2 topics
        try:
            topic = input(f"🎯 Topic {resource_count+1} for resources (or 'skip'): ").strip()
            
            if not topic or topic.lower() == 'skip':
                print("⏭️  Skipping to final interactive mode...")
                break
            
            print(f"\n{'═'*80}")
            print(f"🔍 Searching for resources on: {topic}")
            print(f"{'═'*80}")
            
            resources = recommender.recommend_resources(topic)
            
            # Show GitHub repos
            github_repos = resources.get('github_repositories', [])
            if github_repos:
                print(f"\n💻 GitHub Repositories ({len(github_repos)} found):")
                for i, repo in enumerate(github_repos[:5], 1):
                    stars = repo.get('stars', 0)
                    print(f"   [{i}] {repo.get('name', 'N/A')} ⭐ {stars:,}")
                    print(f"       {repo.get('url', 'N/A')}")
                    if repo.get('description'):
                        print(f"       {repo['description'][:80]}...")
            
            # Show HuggingFace resources
            hf_data = resources.get('huggingface', {})
            hf_models = hf_data.get('models', [])
            
            if hf_models:
                print(f"\n🤗 HuggingFace Models ({len(hf_models)} found):")
                for i, model in enumerate(hf_models[:5], 1):
                    likes = model.get('likes', 0)
                    print(f"   [{i}] {model.get('name', 'N/A')} 💙 {likes}")
                    print(f"       {model.get('url', 'N/A')}")
                    if model.get('task'):
                        print(f"       Task: {model['task']}")
            
            # Show web resources
            tutorials = resources.get('additional_resources', {}).get('tutorials', [])
            if tutorials:
                print(f"\n🌐 Web Resources ({len(tutorials)} found):")
                for i, link in enumerate(tutorials[:3], 1):
                    print(f"   [{i}] {link.get('title', 'N/A')}")
                    print(f"       {link.get('url', 'N/A')}")
            
            print()
            resource_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⏭️  Moving to next section...")
            break
    
    # ===== STEP 4: Interactive Q&A =====
    print_header("STEP 4/4: Interactive Q&A")
    print("\n💡 Now you can ask your own questions!")
    print("💡 Type 'resources <topic>' to find GitHub/HuggingFace resources")
    print("💡 Type 'quit' to exit")
    print()
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Sanshodhak Demo!")
                break
            
            # Check for resource request
            if query.lower().startswith('resources'):
                parts = query.split(maxsplit=1)
                res_topic = parts[1] if len(parts) > 1 else "machine learning"
                
                print(f"\n🔍 Finding resources for: {res_topic}")
                res_results = recommender.recommend_resources(res_topic)
                
                print("\n" + "="*80)
                print(f"🎯 RESOURCES: {res_topic}")
                print("="*80)
                
                github_repos = res_results.get('github_repositories', [])
                if github_repos:
                    print(f"\n💻 GitHub ({len(github_repos)} repos):")
                    for repo in github_repos[:8]:
                        stars = repo.get('stars', 0)
                        print(f"   • {repo.get('name', 'N/A')} ⭐ {stars:,}")
                        print(f"     {repo.get('url', 'N/A')}")
                
                hf_data = res_results.get('huggingface', {})
                hf_models = hf_data.get('models', [])
                
                if hf_models:
                    print(f"\n🤗 HuggingFace Models ({len(hf_models)}):")
                    for model in hf_models[:8]:
                        likes = model.get('likes', 0)
                        print(f"   • {model.get('name', 'N/A')} 💙 {likes}")
                        print(f"     {model.get('url', 'N/A')}")
                
                print()
                continue
            
            # Answer question
            print("\n🔍 Searching papers...")
            answer, sources = rag.query(query, top_k=5, verbose=False)
            
            print("\n" + "─"*80)
            print("📝 ANSWER:")
            print("─"*80)
            # Print full answer or up to 1500 chars
            print(answer[:1500] + ("\n\n[Answer truncated - full answer too long]" if len(answer) > 1500 else ""))
            print("─"*80)
            
            print(f"\n📚 Sources: {len(sources)} chunks")
            if sources:
                print("\n📄 Top sources:")
                for i, src in enumerate(sources[:3], 1):
                    print(f"   [{i}] {src['metadata']['file']} (score: {src['score']:.3f})")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== Summary =====
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print(f"\n📊 Demo Summary:")
    print(f"  • Questions answered: {question_count}")
    print(f"  • Resource topics explored: {resource_count}")
    print(f"  • Total chunks indexed: {rag.index.ntotal:,}")
    print(f"  • Using: OpenRouter Llama 3.2 + Ollama BGE-M3")
    print("\n🚀 To run full pipeline with new papers:")
    print("   python final_pipeline.py")
    print()


if __name__ == "__main__":
    demo()
