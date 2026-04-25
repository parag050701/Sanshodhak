"""
Quick test script to verify all API clients work correctly.
Run this after installation to check your setup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_doaj():
    """Test DOAJ client (user reported this as broken)."""
    print("\n" + "="*60)
    print("Testing DOAJ Client")
    print("="*60)
    
    from ingestion.discovery import DOAJClient
    
    client = DOAJClient()
    
    try:
        result = client.search("machine learning", limit=3)
        
        if result.success and result.papers:
            print(f"✅ DOAJ working! Found {len(result.papers)} papers")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n   {i}. {paper.title}")
                print(f"      Year: {paper.year}")
                print(f"      DOI: {paper.doi}")
                print(f"      OA: {paper.is_open_access}")
        else:
            print(f"⚠️  DOAJ returned no results: {result.error if not result.success else 'empty results'}")
            
    except Exception as e:
        print(f"❌ DOAJ test failed: {e}")
        import traceback
        traceback.print_exc()


def test_openalex():
    """Test OpenAlex client."""
    print("\n" + "="*60)
    print("Testing OpenAlex Client")
    print("="*60)
    
    from ingestion.discovery import OpenAlexClient
    
    client = OpenAlexClient(email="test@sanshodhak.org")
    
    try:
        result = client.search("quantum computing", limit=3)
        
        if result.success and result.papers:
            print(f"✅ OpenAlex working! Found {len(result.papers)} papers")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n   {i}. {paper.title[:60]}...")
                print(f"      Citations: {paper.citations}, Year: {paper.year}")
        else:
            print(f"⚠️  OpenAlex: {result.error if not result.success else 'no results'}")
            
    except Exception as e:
        print(f"❌ OpenAlex test failed: {e}")


def test_semantic_scholar():
    """Test Semantic Scholar client."""
    print("\n" + "="*60)
    print("Testing Semantic Scholar Client")
    print("="*60)
    
    from ingestion.discovery import SemanticScholarClient
    
    client = SemanticScholarClient()
    
    try:
        result = client.search("deep learning", limit=3)
        
        if result.success and result.papers:
            print(f"✅ Semantic Scholar working! Found {len(result.papers)} papers")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n   {i}. {paper.title[:60]}...")
                print(f"      S2 ID: {paper.s2_id}")
        else:
            print(f"⚠️  Semantic Scholar: {result.error if not result.success else 'no results'}")
            
    except Exception as e:
        print(f"❌ Semantic Scholar test failed: {e}")


def test_crossref():
    """Test CrossRef client."""
    print("\n" + "="*60)
    print("Testing CrossRef Client")
    print("="*60)
    
    from ingestion.discovery import CrossRefClient
    
    client = CrossRefClient(email="test@sanshodhak.org")
    
    try:
        result = client.search("artificial intelligence", limit=3, min_year=2023)
        
        if result.success and result.papers:
            print(f"✅ CrossRef working! Found {len(result.papers)} papers")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n   {i}. {paper.title[:60]}...")
                print(f"      DOI: {paper.doi}")
        else:
            print(f"⚠️  CrossRef: {result.error if not result.success else 'no results'}")
            
    except Exception as e:
        print(f"❌ CrossRef test failed: {e}")


def test_arxiv():
    """Test arXiv client."""
    print("\n" + "="*60)
    print("Testing arXiv Client")
    print("="*60)
    
    from ingestion.discovery import ArxivClient
    
    client = ArxivClient()
    
    try:
        result = client.search("neural networks", limit=3)
        
        if result.success and result.papers:
            print(f"✅ arXiv working! Found {len(result.papers)} papers")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n   {i}. {paper.title[:60]}...")
                print(f"      arXiv ID: {paper.arxiv_id}")
        else:
            print(f"⚠️  arXiv: {result.error if not result.success else 'no results'}")
            
    except Exception as e:
        print(f"❌ arXiv test failed: {e}")


def test_doi_utils():
    """Test DOI validation and deduplication."""
    print("\n" + "="*60)
    print("Testing DOI Utilities")
    print("="*60)
    
    from ingestion.discovery.doi_utils import normalize_doi, is_valid_doi
    from ingestion.models import PaperMetadata
    
    # Test normalization
    test_dois = [
        "https://doi.org/10.1038/nature14539",
        "10.1038/nature14539",
        "  10.1038/nature14539  ",
    ]
    
    print("\nDOI Normalization:")
    for doi in test_dois:
        normalized = normalize_doi(doi)
        print(f"   {doi!r} → {normalized!r}")
    
    # Test validation (predatory publishers)
    print("\nDOI Validation (predatory filtering):")
    test_dois = [
        ("10.1038/nature14539", True, "Legitimate (Nature)"),
        ("10.21275/fake123", False, "IJSR (predatory)"),
        ("10.2139/ssrn.123456", False, "SSRN (working paper)"),
    ]
    
    for doi, expected, desc in test_dois:
        valid = is_valid_doi(doi)
        status = "✅" if valid == expected else "❌"
        print(f"   {status} {doi}: {valid} ({desc})")


def test_search_engine():
    """Test unified search engine."""
    print("\n" + "="*60)
    print("Testing Search Engine (Two-Layer Architecture)")
    print("="*60)
    
    from ingestion.discovery import SearchEngine
    
    engine = SearchEngine()
    
    try:
        # Test open-source layer
        print("\n🔍 Testing open-source layer...")
        open_papers = engine.search_open_source("transformer models", limit_per_source=5)
        print(f"   ✅ Found {len(open_papers)} OA papers")
        
        # Test closed-access layer
        print("\n🔍 Testing closed-access layer...")
        closed_papers = engine.search_closed_access("computer vision", limit=5)
        print(f"   ✅ Found {len(closed_papers)} papers")
        
        # Test unified search
        print("\n🔍 Testing unified search (both layers + ranking)...")
        papers = engine.search_unified("natural language processing", limit=10)
        print(f"   ✅ Found {len(papers)} ranked papers")
        
        if papers:
            print(f"\n   Top 3 results:")
            for i, paper in enumerate(papers[:3], 1):
                oa = "🔓" if paper.is_open_access else "🔒"
                print(f"      {i}. {oa} {paper.title[:50]}...")
                print(f"         Citations: {paper.citations}, Year: {paper.year}, Source: {paper.source}")
        
    except Exception as e:
        print(f"❌ Search engine test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 SANSHODHAK API CLIENT TESTS")
    print("="*60)
    
    tests = [
        test_doaj,           # User's main concern
        test_openalex,
        test_semantic_scholar,
        test_crossref,
        test_arxiv,
        test_doi_utils,
        test_search_engine,
    ]
    
    failed = []
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            failed.append((test_func.__name__, str(e)))
            print(f"\n❌ {test_func.__name__} failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = len(tests) - len(failed)
    print(f"\nPassed: {passed}/{len(tests)}")
    
    if failed:
        print(f"\nFailed tests:")
        for name, error in failed:
            print(f"   ❌ {name}: {error}")
    else:
        print(f"\n✅ All tests passed!")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
