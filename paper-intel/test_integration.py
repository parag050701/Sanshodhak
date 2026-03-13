#!/usr/bin/env python3
"""
Test integrated workflow: Download → Auto-process
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.auto_processor import get_auto_processor

def test_auto_processing():
    """Test auto-processing on existing PDFs."""
    
    # Get processor
    processor = get_auto_processor()
    
    # Find existing PDFs
    pdf_dir = Path("research_papers")
    if not pdf_dir.exists():
        print("❌ No research_papers directory found")
        return
    
    pdfs = list(pdf_dir.glob("**/*.pdf"))
    if not pdfs:
        print("❌ No PDFs found in research_papers/")
        return
    
    print(f"\n📄 Found {len(pdfs)} PDFs")
    print("="*80)
    
    # Test on first 3 PDFs
    test_pdfs = pdfs[:3]
    
    for pdf_path in test_pdfs:
        print(f"\n🔄 Processing: {pdf_path.name}")
        print("-"*80)
        
        try:
            result = processor.process_downloaded_pdf(str(pdf_path))
            
            if result['success']:
                print(f"✅ Success!")
                print(f"   JSON: {Path(result['json_path']).name}")
                print(f"   Text: {Path(result['text_path']).name}")
                print(f"   Method: {result['extraction_method']}")
                
                # Show JSON size
                json_size = Path(result['json_path']).stat().st_size / 1024
                text_size = Path(result['text_path']).stat().st_size / 1024
                print(f"   Sizes: JSON={json_size:.1f}KB, Text={text_size:.1f}KB")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "="*80)
    print("✅ Integration test complete!")
    
    # Show processed files
    json_dir = Path("research_papers/processed_json")
    if json_dir.exists():
        json_files = list(json_dir.glob("*.json"))
        print(f"\n📊 Total processed: {len(json_files)} JSON files")


if __name__ == "__main__":
    test_auto_processing()
