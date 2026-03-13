"""Quick CORE test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.DEBUG)

print("\n🧪 Testing CORE API\n")

from ingestion.discovery import COREClient

CORE_API_KEY = "fU7PRKOIE5Nz3roXDSvCwVZ46Wd2TpYi"

client = COREClient(api_key=CORE_API_KEY)
result = client.search("vision transformer", limit=5)

if result.success and result.papers:
    print(f"✅ CORE SUCCESS! Found {len(result.papers)} papers\n")
    
    for i, paper in enumerate(result.papers, 1):
        print(f"{i}. {paper.title[:60]}...")
        print(f"   Year: {paper.year}, OA: {paper.is_open_access}")
        print(f"   DOI: {paper.doi or 'N/A'}\n")
else:
    print(f"❌ CORE FAILED: {result.error}\n")
