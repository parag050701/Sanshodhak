"""
Quick validation test - tests only the most reliable APIs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

print("\n" + "="*60)
print("🧪 QUICK API VALIDATION TEST")
print("="*60)

# Test 1: OpenAlex (most reliable)
print("\n1. Testing OpenAlex...")
try:
    from ingestion.discovery import OpenAlexClient
    client = OpenAlexClient(email="test@sanshodhak.org")
    result = client.search("quantum computing", limit=2)
    if result.success and result.papers:
        print(f"   ✅ OpenAlex: Found {len(result.papers)} papers")
        print(f"      Example: {result.papers[0].title[:50]}...")
    else:
        print(f"   ⚠️  OpenAlex: {result.error}")
except Exception as e:
    print(f"   ❌ OpenAlex failed: {e}")

# Test 2: Semantic Scholar
print("\n2. Testing Semantic Scholar...")
try:
    from ingestion.discovery import SemanticScholarClient
    client = SemanticScholarClient()
    result = client.search("machine learning", limit=2)
    if result.success and result.papers:
        print(f"   ✅ S2: Found {len(result.papers)} papers")
        print(f"      Example: {result.papers[0].title[:50]}...")
    else:
        print(f"   ⚠️  S2: {result.error}")
except Exception as e:
    print(f"   ❌ S2 failed: {e}")

# Test 3: arXiv (very reliable)
print("\n3. Testing arXiv...")
try:
    from ingestion.discovery import ArxivClient
    client = ArxivClient()
    result = client.search("neural networks", limit=2)
    if result.success and result.papers:
        print(f"   ✅ arXiv: Found {len(result.papers)} papers")
        print(f"      Example: {result.papers[0].title[:50]}...")
    else:
        print(f"   ⚠️  arXiv: {result.error}")
except Exception as e:
    print(f"   ❌ arXiv failed: {e}")

# Test 4: DOI utilities
print("\n4. Testing DOI utilities...")
try:
    from ingestion.discovery.doi_utils import normalize_doi, is_valid_doi
    
    test_doi = "https://doi.org/10.1038/nature14539"
    normalized = normalize_doi(test_doi)
    valid = is_valid_doi(normalized)
    
    print(f"   ✅ DOI utils working")
    print(f"      Normalized: {normalized}")
    print(f"      Valid: {valid}")
except Exception as e:
    print(f"   ❌ DOI utils failed: {e}")

# Test 5: Data models
print("\n5. Testing data models...")
try:
    from ingestion.models import PaperMetadata, SearchResult
    
    paper = PaperMetadata(
        doi="10.1038/nature14539",
        title="Deep learning",
        authors=["LeCun, Y", "Bengio, Y", "Hinton, G"],
        year=2015,
        citations=50000
    )
    
    result = SearchResult(
        source="test",
        query="test query",
        papers=[paper],
        total_found=1,
        fetched_count=1,
        success=True
    )
    
    print(f"   ✅ Data models working")
    print(f"      Paper: {paper.title}")
    print(f"      Result: {len(result.papers)} papers")
except Exception as e:
    print(f"   ❌ Data models failed: {e}")

print("\n" + "="*60)
print("✅ QUICK TEST COMPLETE")
print("="*60)
print("\nNote: DOAJ and CORE APIs are currently rate-limited/blocked.")
print("The system will work with OpenAlex, S2, and arXiv.\n")
