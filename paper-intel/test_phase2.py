#!/usr/bin/env python3
"""
Test Phase 2 extraction on sample data.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from extraction.extraction_engine import ExtractionEngine

logging.basicConfig(level=logging.INFO)


async def test_phase2():
    """Test Phase 2 extraction."""
    
    print("\n" + "="*80)
    print("🧪 TESTING PHASE 2: SEMANTIC EXTRACTION")
    print("="*80)
    
    # Check Phase 1 outputs exist
    json_dir = Path("ingestion/output_json")
    text_dir = Path("ingestion/raw_text")
    
    if not json_dir.exists():
        print(f"\n❌ Phase 1 JSON directory not found: {json_dir}")
        print("   Run Phase 1 first: python ingestion/pdf_to_text.py")
        return
    
    if not text_dir.exists():
        print(f"\n❌ Phase 1 text directory not found: {text_dir}")
        return
    
    # Count papers
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"\n❌ No papers found in {json_dir}")
        return
    
    print(f"\n📄 Found {len(json_files)} papers from Phase 1")
    
    # Test on first 3 papers
    test_papers = [f.stem for f in json_files[:3]]
    
    print(f"\n🔬 Testing extraction on {len(test_papers)} papers")
    print("-"*80)
    
    # Create engine
    engine = ExtractionEngine(
        json_dir=json_dir,
        text_dir=text_dir,
        output_dir=Path("extraction"),
        use_graffiti=True,
        use_llm_fallback=True,
        workers=2
    )
    
    async with engine:
        # Test each paper
        for i, paper_id in enumerate(test_papers, 1):
            print(f"\n[{i}/{len(test_papers)}] {paper_id}")
            print("-"*80)
            
            success, error = await engine.extract_single(paper_id)
            
            if success:
                # Load and show results
                entities_file = engine.entities_dir / f"{paper_id}.json"
                relations_file = engine.relations_dir / f"{paper_id}.json"
                triples_file = engine.triples_dir / f"{paper_id}.json"
                
                with open(entities_file) as f:
                    entities = json.load(f)
                
                with open(relations_file) as f:
                    relations = json.load(f)
                
                with open(triples_file) as f:
                    triples_data = json.load(f)
                    triples = triples_data['triples']
                
                print(f"✅ Success!")
                print(f"   Entities: {len(entities)}")
                print(f"   Relations: {len(relations)}")
                print(f"   Triples: {len(triples)}")
                
                # Show sample entities
                print(f"\n   Sample Entities:")
                for entity in entities[:5]:
                    print(f"      {entity['type']}: {entity['name']}")
                
                # Show sample triples
                print(f"\n   Sample Triples:")
                for triple in triples[:5]:
                    print(f"      {triple[0]} -[{triple[1]}]-> {triple[2]}")
            else:
                print(f"❌ Failed: {error}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ Phase 2 test complete!")
    print("="*80)
    
    # Check outputs
    entities_dir = Path("extraction/output_entities")
    if entities_dir.exists():
        entity_files = list(entities_dir.glob("*.json"))
        print(f"\n📊 Total extracted: {len(entity_files)} papers")
        print(f"   Entities: extraction/output_entities/")
        print(f"   Relations: extraction/output_relations/")
        print(f"   Triples: extraction/output_triples/")


if __name__ == "__main__":
    asyncio.run(test_phase2())
