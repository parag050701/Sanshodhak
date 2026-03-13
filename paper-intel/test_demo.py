#!/usr/bin/env python3
"""
Quick test of demo features
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ollama_rag import OllamaRAG
from resource_recommender import ResourceRecommender

print("=" * 80)
print("🧪 TESTING DEMO FEATURES")
print("=" * 80)

# Load RAG
print("\n1️⃣ Loading RAG system...")
rag = OllamaRAG(embed_model='bge-m3', llm_model='deepseek-r1:7b')
rag.load('research_papers/rag_index')
print(f"✅ Loaded {rag.index.ntotal:,} chunks")

# Test query
print("\n2️⃣ Testing query...")
query = "What are transformers?"
print(f"Query: {query}")
answer, sources = rag.query(query, top_k=3)
print(f"✅ Got answer ({len(answer)} chars) from {len(sources)} sources")

# Test resource recommender
print("\n3️⃣ Testing resource recommendations...")
recommender = ResourceRecommender(rag)
resources = recommender.recommend_resources("Deep Learning")
total = sum(len(v) for v in resources.values())
print(f"✅ Found {total} resources")

if resources.get('github'):
    print(f"   • GitHub: {len(resources['github'])} repos")
if resources.get('huggingface'):
    print(f"   • HuggingFace: {len(resources['huggingface'])} items")
if resources.get('web'):
    print(f"   • Web: {len(resources['web'])} links")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - Demo is ready!")
print("=" * 80)
print("\nRun the demo:")
print("  python3 demo_existing_data.py")
