#!/usr/bin/env python3
"""
WORKING PIPELINE TEST - Simplified version that handles all errors
Tests with 5 papers to verify everything works
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

async def test_pipeline():
    print("\n" + "="*80)
    print("🧪 WORKING PIPELINE TEST")
    print("="*80)
    
    # Settings
    query = "Transformers"
    num_papers = 5
    min_year = 2019
    output_dir = Path("research_papers")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSettings:")
    print(f"  Topic: {query}")
    print(f"  Papers: {num_papers}")
    print(f"  Min Year: {min_year}")
    print(f"  Output: {output_dir}/")
    
    try:
        # ============================================================
        # STAGE 1: SEARCH
        # ============================================================
        print("\n" + "="*80)
        print("STAGE 1: SEARCH")
        print("="*80)
        
        engine = IngestionEngine(
            query=query,
            required_count=num_papers,
            output_dir=str(output_dir),
            min_year=min_year,
            enable_core=False,  # Disable CORE to avoid rate limits
            enable_scihub=False,
            enable_libgen=False
        )
        
        print("\n📚 Searching closed-access (CrossRef + Unpaywall)...")
        closed_papers = await engine.search_closed_access()
        print(f"✓ Found {len(closed_papers)} papers")
        
        print("\n🌐 Searching open-source (OpenAlex, S2)...")
        open_papers = await engine.search_open_source()
        print(f"✓ Found {len(open_papers)} papers")
        
        print("\n🔗 Merging and deduplicating...")
        all_papers = await engine.merge_and_deduplicate(open_papers, closed_papers)
        print(f"✓ Total unique: {len(all_papers)} papers")
        
        print("\n🏆 Ranking by quality...")
        ranked_papers = await engine.rank_and_filter(all_papers)
        papers = ranked_papers[:num_papers]
        
        print(f"\n✅ Selected TOP {len(papers)} papers:")
        for i, p in enumerate(papers[:5], 1):
            print(f"  {i}. {p.title[:70]}")
            print(f"     Year: {p.year}, Citations: {p.citations}, OA: {p.is_open_access}")
        
        # ============================================================
        # STAGE 2: DOWNLOAD
        # ============================================================
        print("\n" + "="*80)
        print("STAGE 2: DOWNLOAD")
        print("="*80)
        
        downloaded_pdfs = []
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] {paper.title[:60]}...")
            
            # Download (NOT async!)
            result = engine.pdf_downloader.download_paper(paper)
            
            if result.success:
                print(f"  ✓ Downloaded: {result.pdf_path}")
                downloaded_pdfs.append(Path(result.pdf_path))
            else:
                print(f"  ✗ Failed: {result.error}")
        
        print(f"\n✅ Downloaded {len(downloaded_pdfs)}/{len(papers)} PDFs")
        
        if len(downloaded_pdfs) == 0:
            print("\n⚠️  No PDFs downloaded - stopping here")
            print("(This is normal - many papers don't have open access PDFs)")
            return
        
        # ============================================================
        # STAGE 3: PARSE
        # ============================================================
        print("\n" + "="*80)
        print("STAGE 3: PARSE")
        print("="*80)
        
        text_dir = output_dir / "extracted_text"
        text_dir.mkdir(exist_ok=True)
        
        parsed_texts = []
        for i, pdf_path in enumerate(downloaded_pdfs, 1):
            print(f"\n[{i}/{len(downloaded_pdfs)}] {pdf_path.name}...")
            
            try:
                # extract_text_fast returns (success, text)
                success, text = extract_text_fast(pdf_path)
                if success and text and len(text.strip()) > 100:
                    text_file = text_dir / f"{pdf_path.stem}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"  ✓ Extracted {len(text)} chars")
                    parsed_texts.append(text_file)
                else:
                    print(f"  ✗ No text extracted")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\n✅ Parsed {len(parsed_texts)}/{len(downloaded_pdfs)} PDFs")
        
        if len(parsed_texts) == 0:
            print("\n⚠️  No texts parsed - stopping here")
            return
        
        # ============================================================
        # STAGE 4: EMBED & RAG
        # ============================================================
        print("\n" + "="*80)
        print("STAGE 4: EMBED & RAG")
        print("="*80)
        
        index_dir = output_dir / "rag_index"
        index_dir.mkdir(exist_ok=True)
        
        print("\n🤖 Initializing Ollama RAG...")
        rag = OllamaRAG(embed_model='bge-m3', llm_model='deepseek-r1:7b')
        
        print(f"\n📚 Building index from {text_dir}...")
        rag.build_index(str(text_dir))
        
        print(f"\n💾 Saving to {index_dir}...")
        rag.save(str(index_dir))
        
        print(f"\n✅ Index built: {rag.index.ntotal} chunks")
        
        # ============================================================
        # STAGE 5: RETRIEVE
        # ============================================================
        print("\n" + "="*80)
        print("STAGE 5: RETRIEVE")
        print("="*80)
        
        print(f"\n💡 Querying: {query}")
        answer, sources = rag.query(query, top_k=5)
        
        print("\n" + "-"*80)
        print("📝 ANSWER:")
        print("-"*80)
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        print("-"*80)
        
        print(f"\n✅ Answer generated from {len(sources)} sources")
        
        # ============================================================
        # STAGE 6: RECOMMEND
        # ============================================================
        print("\n" + "="*80)
        print("STAGE 6: RECOMMEND")
        print("="*80)
        
        print(f"\n🎯 Finding resources for: {query}")
        recommender = ResourceRecommender(rag)
        resources = recommender.recommend_resources(query)
        
        if resources.get('huggingface'):
            print(f"\n🤗 HuggingFace: {len(resources['huggingface'])} items")
        if resources.get('github'):
            print(f"💻 GitHub: {len(resources['github'])} repos")
        if resources.get('web'):
            print(f"🌐 Web: {len(resources['web'])} links")
        
        print(f"\n✅ Resources found")
        
        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "="*80)
        print("✅ ALL STAGES COMPLETE!")
        print("="*80)
        
        print(f"\n📁 Output:")
        print(f"  • PDFs: {output_dir}/")
        print(f"  • Texts: {text_dir}/")
        print(f"  • Index: {index_dir}/")
        
        print(f"\n📊 Summary:")
        print(f"  • Papers found: {len(papers)}")
        print(f"  • PDFs downloaded: {len(downloaded_pdfs)}")
        print(f"  • Texts parsed: {len(parsed_texts)}")
        print(f"  • Chunks indexed: {rag.index.ntotal}")
        
        # Show files
        pdfs = list(output_dir.glob("*.pdf"))
        if pdfs:
            print(f"\n📄 PDFs in {output_dir}/:")
            for pdf in pdfs[:10]:
                print(f"  • {pdf.name}")
        
        print("\n🎉 Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("\n🚀 Running working pipeline test...")
    print("   This will take 5-10 minutes")
    print("   Press Ctrl+C to abort\n")
    
    asyncio.run(test_pipeline())
