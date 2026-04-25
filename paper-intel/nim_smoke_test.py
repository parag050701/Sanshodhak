"""
Smoke test for NVIDIA NIM embedding endpoint.
Confirms: API reachability, output dimension, query vs passage input_type, batching, L2-norm state.
"""
import os
import time
import math
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
BASE_URL = os.getenv("NIM_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"

assert API_KEY, "NVIDIA_NIM_API_KEY missing from .env"
print(f"Endpoint : {BASE_URL}")
print(f"Model    : {MODEL}")
print(f"Key      : {API_KEY[:10]}...{API_KEY[-4:]}  (len={len(API_KEY)})")
print()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def embed(texts, input_type):
    t0 = time.perf_counter()
    resp = client.embeddings.create(
        model=MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    dt = time.perf_counter() - t0
    return [d.embedding for d in resp.data], dt


def l2(v):
    return math.sqrt(sum(x * x for x in v))


# 1. single query
vecs, dt = embed(["What is retrieval-augmented generation?"], "query")
print(f"[query x1   ] dim={len(vecs[0])}  L2={l2(vecs[0]):.4f}  latency={dt*1000:.0f}ms")

# 2. single passage
vecs, dt = embed(["RAG combines retrieval over a corpus with generation by an LLM."], "passage")
print(f"[passage x1 ] dim={len(vecs[0])}  L2={l2(vecs[0]):.4f}  latency={dt*1000:.0f}ms")

# 3. batched passages
batch = [
    "Dense retrieval uses learned embeddings.",
    "BM25 is a lexical sparse retriever.",
    "RRF fuses ranked lists from multiple retrievers.",
]
vecs, dt = embed(batch, "passage")
print(f"[passage x3 ] dim={len(vecs[0])}  count={len(vecs)}  latency={dt*1000:.0f}ms  (per-item avg={dt*1000/len(vecs):.0f}ms)")

print("\nSMOKE TEST PASSED")
