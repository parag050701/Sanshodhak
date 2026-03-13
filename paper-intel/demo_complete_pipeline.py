#!/usr/bin/env python3
"""
QUICK DEMO - Shows the complete pipeline is ready
Runs a mini version with 3 papers to verify everything works
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from complete_search_pipeline import CompletePipeline

async def demo():
    print("\n" + "="*80)
    print("QUICK DEMO - Complete Pipeline Test")
    print("="*80)
    print("\nThis will test ALL 7 stages with 3 papers:")
    print("  1. Search (10X strategy)")
    print("  2. Download PDFs")
    print("  3. Parse text")
    print("  4. Build embeddings & RAG")
    print("  5. Evaluate quality")
    print("  6. Answer questions")
    print("  7. Recommend resources")
    print("\nEstimated time: 3-5 minutes\n")
    
    # Override user input for demo
    pipeline = CompletePipeline(output_dir="research_papers")
    
    # Demo settings
    demo_topic = "Graph Neural Networks"
    demo_papers = 3
    demo_year = 2020
    
    print(f"Demo settings:")
    print(f"  Topic: {demo_topic}")
    print(f"  Papers: {demo_papers}")
    print(f"  Min Year: {demo_year}")
    print()
    
    try:
        # Stage 1: Search
        await pipeline.stage1_search(demo_topic, demo_papers, demo_year)
        
        # Stage 2: Download
        if pipeline.papers:
            await pipeline.stage2_download()
        else:
            print("❌ No papers found")
            return
        
        # Stage 3: Parse
        if pipeline.downloaded_pdfs:
            await pipeline.stage3_parse()
        else:
            print("⚠️  No PDFs downloaded, but continuing...")
            return
        
        # Stage 4: Embed & RAG
        if pipeline.parsed_texts:
            await pipeline.stage4_embed_and_rag()
        else:
            print("❌ No texts parsed")
            return
        
        # Stage 5: Evaluate
        await pipeline.stage5_evaluate()
        
        # Stage 6: Retrieve
        await pipeline.stage6_retrieve(demo_topic)
        
        # Stage 7: Recommend
        await pipeline.stage7_recommend(demo_topic)
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETE - All 7 stages working!")
        print("="*80)
        print("\n🎯 Ready for full pipeline with your inputs!")
        print("\nRun: ./run_complete_pipeline.sh")
        print("Or:  python3 complete_search_pipeline.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo())
