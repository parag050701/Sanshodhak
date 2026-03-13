#!/usr/bin/env python3
"""
Quick Research - Simple command-line interface
Usage: python quick_research.py "topic" [num_papers] [min_year]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from research_orchestrator import ResearchOrchestrator

def main():
    if len(sys.argv) < 2:
        print("\n Usage: python quick_research.py \"topic\" [num_papers] [min_year]")
        print("\n Examples:")
        print("   python quick_research.py \"machine learning\" 10 2023")
        print("   python quick_research.py \"quantum computing\" 5")
        print("   python quick_research.py \"deep learning\"")
        sys.exit(1)
    
    topic = sys.argv[1]
    num_papers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2020
    
    orchestrator = ResearchOrchestrator()
    
    try:
        result = orchestrator.conduct_research(
            topic=topic,
            num_papers=num_papers,
            min_year=min_year,
            open_access_only=False,
            download_pdfs=True
        )
        
        if result['success']:
            print(f"\n✅ Complete! Location: {result['output_directory']}")
            print(f"   {result['papers_found']} papers found")
            print(f"   {result['pdfs_downloaded']} PDFs downloaded")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
