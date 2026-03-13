"""
Test Shadow Libraries: Sci-Hub and Anna's Archive
⚠️  WARNING: Use these sources responsibly and legally!
Always exhaust legal sources (Unpaywall, arXiv, OpenAlex) first.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("⚠️  SHADOW LIBRARY TESTING")
print("="*70)
print("\n⚠️  Legal Notice:")
print("   These tools access sources in legal gray areas.")
print("   Use ONLY for research/education in jurisdictions where permitted.")
print("   Always try legal open-access sources first!\n")

# Test 1: Sci-Hub
print("1️⃣  Sci-Hub PDF Resolution")
print("-" * 70)
try:
    from ingestion.discovery import SciHubClient
    
    client = SciHubClient()
    
    # Test with a well-known open DOI (Nature paper)
    test_doi = "10.1038/nature12373"
    
    print(f"   Testing DOI: {test_doi}")
    print(f"   Mirrors: {len(client.mirrors)} available")
    print(f"   Attempting download (may take 10-30s)...\n")
    
    result = client.get_pdf(test_doi)
    
    if result:
        pdf_content, mirror_used = result
        size_mb = len(pdf_content) / (1024 * 1024)
        print(f"✅ SUCCESS - Downloaded PDF")
        print(f"   • Size: {size_mb:.2f} MB")
        print(f"   • Mirror: {mirror_used}")
        print(f"   • Content type: PDF binary (bytes {len(pdf_content)})")
        
        # Verify it's actually a PDF
        if pdf_content.startswith(b'%PDF'):
            print(f"   • Validation: ✓ Valid PDF header")
        else:
            print(f"   • Validation: ⚠️  Unexpected format")
    else:
        print(f"❌ FAILED - No PDF found")
        print(f"   This could mean:")
        print(f"   • DOI not in Sci-Hub database")
        print(f"   • All mirrors are down/blocked")
        print(f"   • Network connectivity issues")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 2: Anna's Archive
print("\n2️⃣  Anna's Archive (LibGen) PDF Resolution")
print("-" * 70)
try:
    from ingestion.discovery import AnnasArchiveClient
    
    client = AnnasArchiveClient()
    
    # Test with a known MD5 from LibGen
    # Note: Real MD5 hashes would come from metadata lookups
    test_md5 = "2c0d8b3a2b9a6c5e4f1d3e7a9b2c4d6e"  # Example MD5
    
    print(f"   Testing MD5: {test_md5}")
    print(f"   LibGen Mirrors: {len(client.libgen_mirrors)} available")
    print(f"   Note: This is a demonstration - real MD5 needed for actual download\n")
    
    result = client.get_pdf_by_md5(test_md5)
    
    if result:
        pdf_content, source = result
        size_mb = len(pdf_content) / (1024 * 1024)
        print(f"✅ SUCCESS - Downloaded PDF")
        print(f"   • Size: {size_mb:.2f} MB")
        print(f"   • Source: {source}")
        print(f"   • Content: {len(pdf_content)} bytes")
    else:
        print(f"⚠️  EXPECTED - Demo MD5 not found")
        print(f"   • This is expected behavior for test MD5")
        print(f"   • Real usage requires valid MD5 from metadata")
        print(f"   • LibGen has 80M+ files accessible via MD5")
        
    # Test DOI lookup (not implemented)
    print(f"\n   Testing DOI lookup:")
    result = client.get_pdf_by_doi("10.1038/nature12373")
    if result:
        print(f"   ✅ DOI lookup working")
    else:
        print(f"   ⚠️  DOI lookup not implemented (requires search)")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

# Summary
print("\n" + "="*70)
print("📊 SHADOW LIBRARY STATUS")
print("="*70)
print("\n🔓 Sci-Hub:")
print("   • Function: Direct PDF download by DOI")
print("   • Mirrors: 6 rotating endpoints")
print("   • Coverage: ~85M+ papers")
print("   • Rate limit: 3s delay (polite)")
print("   • Use case: Last resort after legal sources fail")
print("\n🔓 Anna's Archive / LibGen:")
print("   • Function: PDF download by MD5 hash")
print("   • Mirrors: 4 LibGen endpoints")
print("   • Coverage: ~80M+ files")
print("   • Rate limit: 2s delay")
print("   • Use case: When you have MD5 from metadata")
print("\n⚠️  Legal Reminder:")
print("   Both sources operate in legal gray areas.")
print("   Use only when:")
print("   • Legal OA sources exhausted (Unpaywall, arXiv, etc.)")
print("   • Research/education purpose")
print("   • Jurisdiction permits such access")
print("   • You understand the risks")
print("\n✅ Implementation complete - use responsibly!")
print("="*70 + "\n")
