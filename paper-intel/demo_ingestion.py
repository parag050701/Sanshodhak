"""
Sanshodhak Ingestion Demo
=========================

This script demonstrates the complete Phase-1 ingestion pipeline step-by-step.

Shows:
- Two-layer search (open-source + closed-access)
- Deduplication and ranking
- Multi-source PDF download with fallback
- Iterative expansion if needed
"""

import asyncio
import logging
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import IngestionEngine
from ingestion.models import PaperMetadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_quick():
    """Quick demo: Simple one-liner usage."""
    print("\n" + "="*70)
    print("DEMO 1: Quick Start (One-Liner)")
    print("="*70 + "\n")
    
    engine = IngestionEngine(
        query="quantum computing applications",
        required_count=10,
        min_year=2020
    )
    
    results = await engine.run()
    
    print(f"\n✅ Results:")
    print(f"   Papers found: {len(results['papers'])}")
    print(f"   PDFs downloaded: {len(results['pdf_paths'])}")
    print(f"   Sources used: {results['sources_used']}")
    print(f"   Success rate: {results['success_rate']:.1%}")
    print(f"   Time: {results['elapsed_time']:.1f}s")


async def demo_detailed():
    """Detailed demo: Step-by-step pipeline execution."""
    print("\n" + "="*70)
    print("DEMO 2: Step-by-Step Pipeline")
    print("="*70 + "\n")
    
    # Initialize
    engine = IngestionEngine(
        query="machine learning healthcare",
        required_count=15,
        min_year=2020
    )
    
    # Step 1: Open-source layer
    print("📚 STEP 1: Open-Source Discovery")
    print("   Searching: OpenAlex, CORE, Semantic Scholar, DOAJ")
    open_papers = await engine.search_open_source()
    print(f"   ✅ Found {len(open_papers)} open-access papers")
    
    # Show top 3
    for i, paper in enumerate(open_papers[:3], 1):
        print(f"      {i}. {paper.title[:60]}...")
        print(f"         Year: {paper.year}, Citations: {paper.citations}, Source: {paper.source}")
    
    # Step 2: Closed-access layer
    print(f"\n📖 STEP 2: Closed-Access Enrichment")
    print("   Searching: CrossRef → Unpaywall → arXiv enrichment")
    closed_papers = await engine.search_closed_access()
    print(f"   ✅ Found {len(closed_papers)} additional papers")
    
    # Step 3: Merge
    print(f"\n🔗 STEP 3: Merge & Deduplicate")
    all_papers = engine.merge_and_deduplicate(open_papers, closed_papers)
    print(f"   Original: {len(open_papers) + len(closed_papers)} papers")
    print(f"   ✅ After deduplication: {len(all_papers)} unique papers")
    
    # Step 4: Rank
    print(f"\n📊 STEP 4: Rank by Quality Metrics")
    print("   Scoring: log(citations) + recency + OA bonus + source score")
    ranked = engine.rank_and_filter(all_papers)
    print(f"   ✅ Top {len(ranked)} papers selected")
    
    # Show ranking
    for i, paper in enumerate(ranked[:5], 1):
        oa_status = "🔓 OA" if paper.is_open_access else "🔒 Closed"
        print(f"      {i}. [{oa_status}] {paper.title[:50]}...")
        print(f"         Citations: {paper.citations}, Year: {paper.year}")
    
    # Step 5: Download
    print(f"\n⬇️  STEP 5: Download PDFs (Multi-Source Fallback)")
    print("   Trying: Direct OA → arXiv → Sci-Hub (if enabled) → LibGen")
    results = await engine.download_pdfs(ranked)
    
    success_count = sum(1 for r in results if r.success)
    print(f"   ✅ Downloaded: {success_count}/{len(results)} ({success_count/len(results):.1%})")
    
    # Show download attempts
    for i, result in enumerate(results[:3], 1):
        if result.success:
            print(f"      {i}. ✅ {result.pdf_path.name}")
            print(f"         Source: {result.source}, Size: {result.file_size_bytes/1024:.0f} KB")
        else:
            print(f"      {i}. ❌ Failed: {result.error}")
            print(f"         Attempts: {len(result.attempts)}")


async def demo_expansion():
    """Demo: Iterative expansion when initial results insufficient."""
    print("\n" + "="*70)
    print("DEMO 3: Iterative Expansion")
    print("="*70 + "\n")
    
    # Use narrow query to trigger expansion
    engine = IngestionEngine(
        query="quantum annealing D-Wave systems",
        required_count=20,
        min_year=2023  # Very recent + narrow topic
    )
    
    print("🔍 Initial search with narrow query...")
    results = await engine.run()
    
    print(f"\n📈 Expansion Summary:")
    print(f"   Iterations needed: {results.get('iterations', 1)}")
    print(f"   Final papers: {len(results['papers'])}")
    print(f"   Success rate: {results['success_rate']:.1%}")


async def demo_api_comparison():
    """Demo: Compare different API sources."""
    print("\n" + "="*70)
    print("DEMO 4: API Source Comparison")
    print("="*70 + "\n")
    
    from ingestion.discovery import (
        OpenAlexClient, COREClient, SemanticScholarClient, 
        DOAJClient, CrossRefClient
    )
    
    query = "deep learning"
    limit = 5
    
    clients = [
        ("OpenAlex", OpenAlexClient(email="demo@sanshodhak.org")),
        ("CORE", COREClient()),
        ("Semantic Scholar", SemanticScholarClient()),
        ("DOAJ", DOAJClient()),
        ("CrossRef", CrossRefClient(email="demo@sanshodhak.org")),
    ]
    
    for name, client in clients:
        try:
            print(f"\n📚 {name}:")
            papers = client.search(query, limit=limit)
            print(f"   ✅ Found {len(papers)} papers")
            
            if papers:
                paper = papers[0]
                print(f"   Top result: {paper.title[:60]}...")
                print(f"   Year: {paper.year}, OA: {paper.is_open_access}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def demo_pdf_fallback():
    """Demo: PDF download fallback chain."""
    print("\n" + "="*70)
    print("DEMO 5: PDF Download Fallback Chain")
    print("="*70 + "\n")
    
    from ingestion.discovery import PDFDownloader
    
    # Create test papers with different availability
    papers = [
        PaperMetadata(
            doi="10.1038/nature14539",
            title="Deep learning",
            pdf_urls=["https://www.nature.com/articles/nature14539.pdf"],
            is_open_access=False
        ),
        PaperMetadata(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            pdf_urls=[],
            is_open_access=True
        ),
    ]
    
    downloader = PDFDownloader(output_dir="demo_papers/")
    
    for i, paper in enumerate(papers, 1):
        print(f"\n📄 Paper {i}: {paper.title[:50]}...")
        
        result = downloader.download_paper(paper)
        
        if result.success:
            print(f"   ✅ Success!")
            print(f"   Source: {result.source}")
            print(f"   Path: {result.pdf_path}")
            print(f"   Size: {result.file_size_bytes/1024:.0f} KB")
        else:
            print(f"   ❌ Failed: {result.error}")
            print(f"   Attempts: {len(result.attempts)}")
            for attempt in result.attempts:
                print(f"      - {attempt['source']}: {attempt['error']}")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🚀 SANSHODHAK INGESTION ENGINE - DEMO SUITE")
    print("="*70)
    
    demos = [
        ("Quick Start", demo_quick),
        ("Detailed Pipeline", demo_detailed),
        ("Iterative Expansion", demo_expansion),
        ("API Comparison", demo_api_comparison),
        ("PDF Fallback", demo_pdf_fallback),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"   {i}. {name}")
    print("   0. Run all demos")
    
    choice = input("\nSelect demo (0-5): ").strip()
    
    if choice == "0":
        for name, demo_func in demos:
            try:
                await demo_func()
            except Exception as e:
                print(f"\n❌ Demo '{name}' failed: {e}")
                import traceback
                traceback.print_exc()
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        name, demo_func = demos[int(choice) - 1]
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*70)
    print("✅ Demo complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
