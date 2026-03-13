"""
Simple example: Fetch 10 papers on quantum computing
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import IngestionEngine


async def main():
    print("🔍 Sanshodhak Paper Ingestion Demo\n")
    
    # Create engine
    engine = IngestionEngine(
        query="quantum computing applications",
        required_count=10,
        min_year=2020
    )
    
    print(f"Query: {engine.query}")
    print(f"Target: {engine.required_count} papers")
    print(f"Year range: {engine.min_year}-{engine.max_year or 'present'}\n")
    
    # Run pipeline
    print("Running ingestion pipeline...\n")
    results = await engine.run()
    
    # Print results
    print("\n" + "="*60)
    print("✅ RESULTS")
    print("="*60)
    
    print(f"\n📊 Papers found: {len(results['papers'])}")
    print(f"⬇️  PDFs downloaded: {len(results['pdf_paths'])}")
    print(f"✨ Success rate: {results['success_rate']:.1%}")
    print(f"⏱️  Time: {results['elapsed_time']:.1f}s")
    print(f"🔍 Sources used: {', '.join(results['sources_used'])}")
    
    if results['pdf_paths']:
        print(f"\n📁 Downloaded PDFs:")
        for i, path in enumerate(results['pdf_paths'][:5], 1):
            print(f"   {i}. {Path(path).name}")
        if len(results['pdf_paths']) > 5:
            print(f"   ... and {len(results['pdf_paths']) - 5} more")
    
    print("\n" + "="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
