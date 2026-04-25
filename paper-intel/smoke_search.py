"""
End-to-end smoke test: load GraphRAG with the new NIM-embedded 2048-dim index,
run a single query, print top-5 results.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / ".env")

from graph_rag import GraphRAG

QUERY = "How does retrieval-augmented generation handle queries with rare scientific terms?"

t0 = time.perf_counter()
rag = GraphRAG()
rag.load(str(SCRIPT_DIR / "rag_index"))
t_load = time.perf_counter() - t0
print(f"\nload time: {t_load:.1f}s")
print(f"index dim: {rag.index.d}  (expect 2048)")
assert rag.index.d == 2048, f"index dimension mismatch: got {rag.index.d}"

print(f"\nQUERY: {QUERY}\n")
t1 = time.perf_counter()
results = rag.search(QUERY, top_k=5)
t_search = time.perf_counter() - t1
print(f"search time: {t_search*1000:.0f}ms\n")

for i, r in enumerate(results, 1):
    src = r.get("metadata", {}).get("file", "?")
    score = r.get("score", 0.0)
    text = r.get("chunk", r.get("text", ""))[:160].replace("\n", " ")
    print(f"  #{i}  score={score:.4f}  file={src}")
    print(f"      {text}...")
    print()

print("SMOKE SEARCH PASSED")
