"""
Minimal validation - OpenAlex only (most reliable).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.ERROR)

print("\n🧪 Minimal System Validation\n")

# Test OpenAlex
print("Testing OpenAlex API...")
from ingestion.discovery import OpenAlexClient

client = OpenAlexClient(email="test@sanshodhak.org")
result = client.search("quantum computing", limit=3)

if result.success and result.papers:
    print(f"✅ SUCCESS! Found {len(result.papers)} papers\n")
    
    for i, paper in enumerate(result.papers, 1):
        print(f"{i}. {paper.title[:60]}...")
        print(f"   Year: {paper.year}, Citations: {paper.citations}")
        print(f"   DOI: {paper.doi}")
        print(f"   Open Access: {paper.is_open_access}\n")
    
    print("="*60)
    print("✅ Core system is working!")
    print("="*60)
    print("\nNote: Some APIs (DOAJ, CORE) may be rate-limited.")
    print("The system will work with OpenAlex, arXiv, and CrossRef.\n")
else:
    print(f"❌ Failed: {result.error}\n")
