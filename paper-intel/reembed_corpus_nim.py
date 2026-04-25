"""
Re-embed corpus chunks via NVIDIA NIM and rebuild the FAISS index at 2048-dim.

Reads:  rag_index/metadata.pkl  (dict with 'chunks' list)
Writes: rag_index/faiss.index   (new IndexFlatIP, 2048-dim)
Backs up the prior 1024-dim index to rag_index/faiss.index.bge-m3.bak.
metadata.pkl is unchanged (chunk text/order preserved).
"""
import os
import sys
import time
import pickle
import shutil
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_DIR = SCRIPT_DIR / "rag_index"
META_PATH = INDEX_DIR / "metadata.pkl"
FAISS_PATH = INDEX_DIR / "faiss.index"
BACKUP_PATH = INDEX_DIR / "faiss.index.bge-m3.bak"

load_dotenv(SCRIPT_DIR / ".env")
API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
BASE_URL = os.getenv("NIM_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
EMBED_DIM = 2048
BATCH_SIZE = 32

assert API_KEY, "NVIDIA_NIM_API_KEY missing from .env"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def embed_batch(texts, input_type="passage", retries=3):
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(
                model=MODEL,
                input=texts,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}: {e}  (sleeping {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries on batch of size {len(texts)}")


def main():
    if not META_PATH.exists():
        sys.exit(f"missing {META_PATH}")

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    chunks = meta["chunks"]
    n = len(chunks)
    print(f"corpus: {n} chunks")

    if FAISS_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(FAISS_PATH, BACKUP_PATH)
        print(f"backed up old index -> {BACKUP_PATH.name}")
    elif BACKUP_PATH.exists():
        print(f"backup already exists at {BACKUP_PATH.name} (not overwriting)")

    vectors = np.zeros((n, EMBED_DIM), dtype=np.float32)
    t_start = time.perf_counter()

    for i in range(0, n, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embs = embed_batch(batch, input_type="passage")
        vectors[i : i + len(embs)] = np.asarray(embs, dtype=np.float32)
        done = i + len(embs)
        elapsed = time.perf_counter() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"  {done}/{n}  ({100*done/n:5.1f}%)  rate={rate:.1f}/s  eta={eta:.0f}s")

    norms = np.linalg.norm(vectors, axis=1)
    print(f"\nvector L2 norms: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}")
    if not np.allclose(norms, 1.0, atol=1e-2):
        print("  -> renormalizing (L2)")
        vectors = vectors / norms[:, None]

    print(f"building IndexFlatIP({EMBED_DIM})...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_PATH))

    total = time.perf_counter() - t_start
    print(f"\nDONE  ntotal={index.ntotal}  d={index.d}  total={total:.1f}s ({total/60:.1f} min)")
    print(f"new index -> {FAISS_PATH}")


if __name__ == "__main__":
    main()
