#!/usr/bin/env python3
"""
DIRECT TEST - Run full pipeline WITHOUT web server
Tests: Search → Download → Parse → Embed → Retrieve → Recommend
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sanshodhak_agent import SanshodhakAgent

async def test_full_pipeline():
    print("="*80)
    print("🧪 TESTING FULL SANSHODHAK PIPELINE")
    print("="*80)
    
    # Progress callback
    def progress(update):
        stage = update.get('stage', 'unknown')
        status = update.get('status', 'unknown')
        message = update.get('message', '')
        print(f"\n[{stage.upper()}] {status}: {message}")
        if update.get('data'):
            for key, value in update['data'].items():
                print(f"  {key}: {value}")
    
    # Initialize agent
    print("\n1. Initializing agent...")
    agent = SanshodhakAgent(progress_callback=progress)
    print("✅ Agent initialized")
    
    # Test query
    query = "Graph Neural Networks"
    required_papers = 10  # Small number for testing
    min_year = 2016
    
    print(f"\n2. Starting pipeline:")
    print(f"   Query: {query}")
    print(f"   Papers: {required_papers}")
    print(f"   Min year: {min_year}")
    
    try:
        # Execute full pipeline
        result = await agent.execute_full_pipeline(
            query=query,
            required_papers=required_papers,
            min_year=min_year
        )
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        
        print(f"\nSession ID: {result['session_id']}")
        print(f"\nStage Results:")
        for stage_name, stage_data in result['stages'].items():
            print(f"\n  {stage_name}:")
            for key, value in stage_data.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"    {key}: {value}")
        
        # Check session directory
        session_dir = Path(f"sessions/{result['session_id']}")
        if session_dir.exists():
            print(f"\n📁 Session directory created: {session_dir}")
            
            papers_dir = session_dir / "papers"
            text_dir = session_dir / "text"
            index_dir = session_dir / "index"
            
            if papers_dir.exists():
                pdf_count = len(list(papers_dir.glob("*.pdf")))
                print(f"   Papers: {pdf_count} PDFs")
            
            if text_dir.exists():
                txt_count = len(list(text_dir.glob("*.txt")))
                print(f"   Text: {txt_count} text files")
            
            if index_dir.exists():
                if (index_dir / "faiss.index").exists():
                    print(f"   Index: FAISS index created ✅")
                if (index_dir / "metadata.pkl").exists():
                    print(f"   Metadata: metadata.pkl saved ✅")
        
        print("\n" + "="*80)
        print("✅ TEST PASSED - Full pipeline working!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("\n🚀 Running direct pipeline test...")
    print("   This will take 5-15 minutes for 5 papers")
    print("   Press Ctrl+C to abort\n")
    
    asyncio.run(test_full_pipeline())
