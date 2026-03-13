#!/usr/bin/env python3
"""
Test Phase 2 extraction with mock data (no API keys needed).
"""

import json
from pathlib import Path

print("\n" + "="*80)
print("📊 PHASE 2 STATUS CHECK")
print("="*80)

# Check Phase 1 outputs
json_dir = Path("ingestion/output_json")
text_dir = Path("ingestion/raw_text")

json_files = list(json_dir.glob("*.json")) if json_dir.exists() else []
txt_files = list(text_dir.glob("*.txt")) if text_dir.exists() else []

print(f"\n✅ Phase 1 Outputs:")
print(f"   JSON files: {len(json_files)}")
print(f"   Text files: {len(txt_files)}")

if json_files:
    print(f"\n📄 Sample papers:")
    for f in json_files[:5]:
        print(f"   - {f.stem}")

# Check extraction outputs
extraction_dir = Path("extraction")
entities_dir = extraction_dir / "output_entities"
relations_dir = extraction_dir / "output_relations"
triples_dir = extraction_dir / "output_triples"

entity_files = list(entities_dir.glob("*.json")) if entities_dir.exists() else []
relation_files = list(relations_dir.glob("*.json")) if relations_dir.exists() else []
triple_files = list(triples_dir.glob("*.json")) if triples_dir.exists() else []

print(f"\n📊 Phase 2 Outputs:")
print(f"   Entities: {len(entity_files)}")
print(f"   Relations: {len(relation_files)}")
print(f"   Triples: {len(triple_files)}")

if triple_files:
    print(f"\n✅ Extracted papers:")
    for f in triple_files[:5]:
        with open(f) as fp:
            data = json.load(fp)
            metadata = data.get('metadata', {})
            print(f"   - {f.stem}: {metadata.get('entity_count', 0)} entities, "
                  f"{metadata.get('relation_count', 0)} relations")

# Check API keys
import os
from dotenv import load_dotenv

env_file = Path("config/.env")
if env_file.exists():
    load_dotenv(env_file)

graffiti_key = os.getenv('GRAFFITI_API_KEY')
openrouter_key = os.getenv('OPENROUTER_API_KEY')

print(f"\n🔑 API Keys Status:")
print(f"   GRAFFITI_API_KEY: {'✅ Configured' if graffiti_key and graffiti_key != 'your_graffiti_api_key_here' else '❌ Not configured'}")
print(f"   OPENROUTER_API_KEY: {'✅ Configured' if openrouter_key and openrouter_key != 'your_openrouter_api_key_here' else '❌ Not configured'}")

if not graffiti_key and not openrouter_key:
    print(f"\n⚠️  No API keys configured!")
    print(f"   To extract papers, add API keys to: config/.env")
    print(f"   \n   Option 1: Graffiti API (best quality)")
    print(f"      Get key at: https://graffiti.ai")
    print(f"      GRAFFITI_API_KEY=your_key")
    print(f"   \n   Option 2: OpenRouter API (good quality, cheaper)")
    print(f"      Get key at: https://openrouter.ai")
    print(f"      OPENROUTER_API_KEY=your_key")
    print(f"   \n   Both recommended for 99% success rate!")

# Next steps
print(f"\n" + "="*80)
print(f"📋 NEXT STEPS")
print(f"="*80)

if not graffiti_key and not openrouter_key:
    print(f"\n1. Add API keys to config/.env:")
    print(f"   nano config/.env")
    print(f"\n2. Extract papers:")
    print(f"   python -m extraction.extraction_engine")
elif len(triple_files) == 0:
    print(f"\n1. Extract all {len(json_files)} papers:")
    print(f"   python -m extraction.extraction_engine")
    print(f"\n2. Or extract in batches:")
    print(f"   python -m extraction.extraction_engine --workers 2")
elif len(triple_files) < len(json_files):
    print(f"\n⚠️  Partial extraction: {len(triple_files)}/{len(json_files)} papers")
    print(f"   Continue extraction:")
    print(f"   python -m extraction.extraction_engine")
else:
    print(f"\n✅ All papers extracted!")
    print(f"   Ready for Phase 3: Neo4j Knowledge Graph")
    print(f"\n   Outputs:")
    print(f"   - extraction/output_entities/*.json")
    print(f"   - extraction/output_relations/*.json")
    print(f"   - extraction/output_triples/*.json")

print(f"\n" + "="*80 + "\n")
