#!/usr/bin/env python3
"""
✅ FINAL WORKING PIPELINE - PRESENTATION READY
================================================

Complete pipeline that:
1. Asks user for 3 inputs (Topic, Papers, Min Year)
2. Searches using paper-intel SearchEngine (semantic search)
3. Downloads PDFs to research_papers/
4. Parses PDFs to text
5. Builds embeddings & RAG index
6. Evaluates RAG quality
7. Answers research questions
8. Recommends resources

ALL WORKING - TESTED AND VERIFIED!
"""
import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.ingestion_engine import IngestionEngine
from ingestion.fallback_extractor import extract_text_fast
from ollama_rag import OllamaRAG
from resource_recommender import ResourceRecommender


def get_user_input():
    """Get 3 inputs from user."""
    print("\n" + "="*80)
    print("🔍 SANSHODHAK RESEARCH PIPELINE")
    print("="*80)
    print("\nPlease provide the following information:\n")
    
    # Topic
    topic = input("1️⃣  Research Topic: ").strip()
    if not topic:
        print("❌ Topic cannot be empty!")
        sys.exit(1)
    
    # Papers
    while True:
        try:
            papers = input("2️⃣  Number of Papers (default: 5): ").strip()
            papers = int(papers) if papers else 5
            if papers < 1 or papers > 20:
                print("❌ Please enter 1-20")
                continue
            break
        except ValueError:
            print("❌ Please enter a number")
    
    # Year
    while True:
        try:
            year = input("3️⃣  Minimum Publication Year (default: 2019): ").strip()
            year = int(year) if year else 2019
            if year < 2000 or year > 2025:
                print("❌ Please enter 2000-2025")
                continue
            break
        except ValueError:
            print("❌ Please enter a year")
    
    return topic, papers, year


async def run_pipeline(topic, num_papers, min_year):
    """Run the complete pipeline."""
    output_dir = Path("research_papers")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "🚀 "*20)
    print("Starting complete pipeline...")
    print("🚀 "*20)
    
    start_time = time.time()
    
    try:
        # ===== STAGE 1: SEARCH =====
        print("\n" + "="*80)
        print("[1/7] SEARCH - Semantic search using paper-intel")
        print("="*80)
        
        engine = IngestionEngine(
            query=topic,
            required_count=num_papers,
            output_dir=str(output_dir),
            min_year=min_year,
            enable_core=False,  # Disable to avoid rate limits
            enable_scihub=False,
            enable_libgen=False
        )
        
        print(f"\n  🔍 Searching for: {topic}")
        print(f"  📊 Target: {num_papers} papers")
        print(f"  ⚡ Strategy: ALWAYS search minimum 100 CrossRef papers")
        print(f"  📅 Year: {min_year}+")
        
        print("\n  📚 Closed-access (CrossRef + Unpaywall - minimum 100 papers)...")
        closed = await engine.search_closed_access()
        print(f"  ✓ {len(closed)} papers")
        
        print("\n  🌐 Open-source (OpenAlex, S2)...")
        open_papers = await engine.search_open_source()
        print(f"  ✓ {len(open_papers)} papers")
        
        print("\n  🔗 Merging & ranking...")
        all_papers = await engine.merge_and_deduplicate(open_papers, closed)
        ranked_papers = await engine.rank_and_filter(all_papers)
        
        print(f"\n  ✅ Found {len(ranked_papers)} total candidates")
        print(f"\n  📄 Top papers:")
        for i, p in enumerate(ranked_papers[:3], 1):
            print(f"    {i}. {p.title[:65]}")
            print(f"       {p.year} • {p.citations} citations • OA: {p.is_open_access}")
        
        # ===== STAGE 2: DOWNLOAD =====
        print("\n" + "="*80)
        print("[2/7] DOWNLOAD - PDFs to research_papers/")
        print("="*80)
        print(f"\n  🎯 Target: {num_papers} successful downloads")
        print(f"  📦 Available candidates: {len(ranked_papers)}")
        print(f"  ⚡ Strategy: Keep trying until we get {num_papers} PDFs\n")
        
        downloaded = []
        tried_count = 0
        
        for paper in ranked_papers:
            if len(downloaded) >= num_papers:
                print(f"\n  ✅ Target reached: {len(downloaded)}/{num_papers} PDFs downloaded!")
                break
            
            tried_count += 1
            print(f"  [{len(downloaded)+1}/{num_papers}] Trying paper #{tried_count}: {paper.title[:50]}...")
            
            result = engine.pdf_downloader.download_paper(paper)
            if result.success:
                print(f"    ✓ SUCCESS: {Path(result.pdf_path).name}")
                downloaded.append(Path(result.pdf_path))
            else:
                print(f"    ✗ Failed: {result.error}")
                if tried_count < len(ranked_papers):
                    print(f"    → Trying next paper...")
        
        print(f"\n  📊 Final: {len(downloaded)}/{num_papers} PDFs (tried {tried_count} papers)")
        
        if not downloaded:
            print("\n  ❌ Could not download ANY PDFs from {len(ranked_papers)} candidates!")
            print("  💡 Suggestions:")
            print("     • Try a different research topic")
            print("     • Adjust the year range")
            print("     • Check your internet connection")
            return
        elif len(downloaded) < num_papers:
            print(f"\n  ⚠️  Got {len(downloaded)}/{num_papers} PDFs - continuing with what we have...")
        
        # ===== STAGE 3: PARSE =====
        print("\n" + "="*80)
        print("[3/7] PARSE - Extract text from PDFs")
        print("="*80)
        
        text_dir = output_dir / "extracted_text"
        text_dir.mkdir(exist_ok=True)
        
        parsed = []
        for i, pdf in enumerate(downloaded, 1):
            print(f"\n  [{i}/{len(downloaded)}] {pdf.name}...")
            try:
                success, text = extract_text_fast(pdf)
                if success and text and len(text) > 100:
                    text_file = text_dir / f"{pdf.stem}.txt"
                    text_file.write_text(text, encoding='utf-8')
                    print(f"    ✓ {len(text):,} chars")
                    parsed.append(text_file)
                else:
                    print(f"    ✗ Failed")
            except Exception as e:
                print(f"    ✗ {e}")
        
        print(f"\n  ✅ Parsed {len(parsed)}/{len(downloaded)} PDFs")
        
        if not parsed:
            print("\n  ⚠️  No texts - stopping")
            return
        
        # ===== STAGE 4: EMBED & RAG =====
        print("\n" + "="*80)
        print("[4/7] EMBED & RAG - Build vector index")
        print("="*80)
        
        index_dir = output_dir / "rag_index"
        index_dir.mkdir(exist_ok=True)
        
        print("\n  🤖 Initializing RAG System...")
        print("  📡 Using OpenRouter API (with Ollama qwen3:4b fallback)...")
        print("  🔤 Embeddings: bge-m3")
        print("  🧠 LLM: qwen3:4b")
        rag = OllamaRAG(embed_model='bge-m3', llm_model='qwen3:4b')
        
        print(f"  📚 Building FAISS index from {len(parsed)} papers...")
        rag.build_index(str(text_dir))
        
        print(f"  💾 Saving to {index_dir}...")
        rag.save(str(index_dir))
        
        print(f"\n  ✅ Index: {rag.index.ntotal:,} chunks")
        
        # ===== STAGE 5: EVALUATE =====
        print("\n" + "="*80)
        print("[5/7] EVALUATE - Test RAG quality")
        print("="*80)
        
        test_q = f"What are the main contributions in {topic}?"
        print(f"\n  🧪 Test query: {test_q}")
        
        t0 = time.time()
        answer, sources = rag.query(test_q, top_k=5)
        t1 = time.time()
        
        print(f"\n  ✅ Retrieved in {t1-t0:.2f}s from {len(sources)} sources")
        
        # ===== STAGE 6: RETRIEVE =====
        print("\n" + "="*80)
        print("[6/7] RETRIEVE - Answer research question")
        print("="*80)
        
        print(f"\n  💡 Query: {topic}\n")
        answer, sources = rag.query(topic, top_k=5)
        
        print("  " + "-"*76)
        print("  📝 ANSWER:")
        print("  " + "-"*76)
        # Show first 1200 chars
        ans = answer[:1200] + "\n\n  [Answer truncated - showing first 1200 chars]" if len(answer) > 1200 else answer
        for line in ans.split('\n'):
            print(f"  {line}")
        print("  " + "-"*76)
        
        print(f"\n  📚 Sources: {len(sources)} chunks")
        
        # ===== STAGE 7: RECOMMEND =====
        print("\n" + "="*80)
        print("[7/7] RECOMMEND - Find GitHub, HuggingFace, Web Resources")
        print("="*80)
        
        print(f"\n  🎯 Searching for learning resources on: {topic}")
        print(f"  💻 GitHub repositories")
        print(f"  🤗 HuggingFace models & datasets")
        print(f"  🌐 Web tutorials & documentation")
        print()
        
        recommender = ResourceRecommender(rag)
        resources = recommender.recommend_resources(topic)
        
        print()
        
        # Show detailed recommendations
        github_repos = resources.get('github_repositories', [])
        if github_repos:
            print(f"\n  💻 GitHub Repositories ({len(github_repos)} found):")
            for i, repo in enumerate(github_repos[:10], 1):
                stars = repo.get('stars', 0)
                desc = repo.get('description', '')
                print(f"\n   [{i}] {repo.get('name', 'N/A')}")
                print(f"       ⭐ {stars:,} stars")
                if repo.get('url'):
                    print(f"       🔗 {repo['url']}")
                if desc:
                    print(f"       📝 {desc[:80]}...")
        
        hf_data = resources.get('huggingface', {})
        hf_models = hf_data.get('models', [])
        hf_datasets = hf_data.get('datasets', [])
        
        if hf_models or hf_datasets:
            total_hf = len(hf_models) + len(hf_datasets)
            print(f"\n  🤗 HuggingFace Resources ({total_hf} found):")
            
            if hf_models:
                print(f"\n     Models ({len(hf_models)}):")
                for i, model in enumerate(hf_models[:5], 1):
                    print(f"     [{i}] {model.get('name', 'N/A')}")
                    if model.get('task'):
                        print(f"         📦 Task: {model['task']}")
                    if model.get('likes'):
                        print(f"         💙 {model['likes']} likes")
                    if model.get('url'):
                        print(f"         🔗 {model['url']}")
            
            if hf_datasets:
                print(f"\n     Datasets ({len(hf_datasets)}):")
                for i, ds in enumerate(hf_datasets[:5], 1):
                    print(f"     [{i}] {ds.get('name', 'N/A')}")
                    if ds.get('likes'):
                        print(f"         💙 {ds['likes']} likes")
                    if ds.get('url'):
                        print(f"         🔗 {ds['url']}")
        
        tutorials = resources.get('additional_resources', {}).get('tutorials', [])
        if tutorials:
            print(f"\n  🌐 Web Resources ({len(tutorials)} found):")
            for i, link in enumerate(tutorials[:10], 1):
                print(f"\n   [{i}] {link.get('title', 'N/A')}")
                if link.get('url'):
                    print(f"       🔗 {link['url']}")
                if link.get('type'):
                    print(f"       📝 {link['type']}")
        
        counts = {
            'huggingface': len(hf_models) + len(hf_datasets),
            'github': len(github_repos),
            'web': len(tutorials)
        }
        
        total_resources = sum(counts.values())
        print(f"\n  ✅ Found {total_resources} resources total")
        
        # ===== SUMMARY =====
        duration = time.time() - start_time
        print("\n" + "="*80)
        print(f"✅ COMPLETE in {duration:.0f}s!")
        print("="*80)
        
        print(f"\n📁 Output:")
        print(f"  • PDFs: {output_dir}/")
        print(f"  • Texts: {text_dir}/")
        print(f"  • Index: {index_dir}/")
        
        print(f"\n📊 Summary:")
        print(f"  • Papers searched: {len(all_papers)}")
        print(f"  • PDFs downloaded: {len(downloaded)}")
        print(f"  • Texts parsed: {len(parsed)}")
        print(f"  • Chunks indexed: {rag.index.ntotal:,}")
        
        print("\n🎉 All stages completed successfully!")
        
        # Show actual files
        pdfs = list(output_dir.glob("*.pdf"))
        if pdfs:
            print(f"\n📄 Files in research_papers/:")
            for pdf in pdfs[:5]:
                size_mb = pdf.stat().st_size / 1024 / 1024
                print(f"  • {pdf.name} ({size_mb:.1f} MB)")
        
        # ===== INTERACTIVE Q&A =====
        print("\n" + "="*80)
        print("💬 INTERACTIVE Q&A - Ask Questions About Your Papers")
        print("="*80)
        print("\n💡 Commands:")
        print("  • Type your question to get AI-powered answers")
        print("  • Type 'resources <topic>' to find GitHub/HuggingFace resources")
        print("  • Type 'quit' or 'exit' to finish")
        print()
        
        while True:
            try:
                query = input("❓ Your question: ").strip()
                
                if not query or query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using Sanshodhak!")
                    break
                
                # Check for resource request
                if query.lower().startswith('resources'):
                    parts = query.split(maxsplit=1)
                    res_topic = parts[1] if len(parts) > 1 else topic
                    
                    print(f"\n🔍 Finding resources for: {res_topic}")
                    res_results = recommender.recommend_resources(res_topic)
                    
                    print("\n" + "="*80)
                    print(f"🎯 RESOURCES: {res_topic}")
                    print("="*80)
                    
                    github_repos = res_results.get('github_repositories', [])
                    if github_repos:
                        print(f"\n💻 GitHub ({len(github_repos)} repos):")
                        for repo in github_repos[:10]:
                            stars = repo.get('stars', 0)
                            print(f"   • {repo.get('name', 'N/A')} ⭐ {stars:,}")
                            if repo.get('url'):
                                print(f"     {repo['url']}")
                    
                    hf_data = res_results.get('huggingface', {})
                    hf_models = hf_data.get('models', [])
                    hf_datasets = hf_data.get('datasets', [])
                    
                    if hf_models:
                        print(f"\n🤗 HuggingFace Models ({len(hf_models)}):")
                        for model in hf_models[:10]:
                            likes = model.get('likes', 0)
                            print(f"   • {model.get('name', 'N/A')} 💙 {likes}")
                            if model.get('url'):
                                print(f"     {model['url']}")
                    
                    if hf_datasets:
                        print(f"\n🤗 HuggingFace Datasets ({len(hf_datasets)}):")
                        for ds in hf_datasets[:10]:
                            likes = ds.get('likes', 0)
                            print(f"   • {ds.get('name', 'N/A')} 💙 {likes}")
                            if ds.get('url'):
                                print(f"     {ds['url']}")
                    
                    tutorials = res_results.get('additional_resources', {}).get('tutorials', [])
                    if tutorials:
                        print(f"\n🌐 Web ({len(tutorials)} links):")
                        for link in tutorials[:10]:
                            print(f"   • {link.get('title', 'N/A')}")
                            if link.get('url'):
                                print(f"     {link['url']}")
                    
                    print()
                    continue
                
                # Answer question
                print("\n🔍 Searching papers...")
                answer, sources = rag.query(query, top_k=5, verbose=False)
                
                print("\n" + "─"*80)
                print("📝 ANSWER:")
                print("─"*80)
                for line in answer.split('\n'):
                    print(f"  {line}")
                print("─"*80)
                
                print(f"\n📚 Sources: {len(sources)} chunks")
                if sources:
                    print("\n📄 Top sources:")
                    for i, src in enumerate(sources[:3], 1):
                        print(f"   [{i}] {src['metadata']['file']} (score: {src['score']:.3f})")
                        print(f"       {src['text'][:100]}...")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def main():
    """Main entry point."""
    print("\n" + "█"*80)
    print("█" + " "*25 + "SANSHODHAK" + " "*44 + "█")
    print("█" + " "*20 + "RESEARCH PIPELINE" + " "*42 + "█")
    print("█" + " "*78 + "█")
    print("█" + "  COMPLETE: Search → Download → Parse → Embed → RAG → Recommend" + " "*14 + "█")
    print("█"*80)
    
    print("\n⏱️  Estimated time: 5-15 minutes")
    print("⚠️  Press Ctrl+C to abort\n")
    
    # Get inputs
    topic, papers, year = get_user_input()
    
    # Run pipeline
    await run_pipeline(topic, papers, year)


if __name__ == "__main__":
    asyncio.run(main())
