"""
rebuild_rag_index.py — Rebuild rag_index/ from all files in ingestion/raw_text/

Reads every .txt file, chunks it (512 tokens / 64 overlap), embeds with BGE-M3,
and saves a fresh rag_index/faiss.index + rag_index/metadata.pkl that graph_rag.py
can load directly.

Usage:
    D:\\Sanshodhak\\.venv\\Scripts\\python.exe rebuild_rag_index.py

Output:
    rag_index/faiss.index        (FAISS IndexFlatIP, 1024-dim BGE-M3 vectors)
    rag_index/metadata.pkl       ({"chunks": [...], "metadata": [...]})
    rag_index/faiss.index.bak    (backup of previous index)
    rag_index/metadata.pkl.bak   (backup of previous metadata)
"""

import os
import sys
import pickle
import shutil
import logging
from pathlib import Path

import numpy as np

# Prevent OpenBLAS thread conflicts on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
RAW_TEXT_DIR = BASE_DIR / "ingestion" / "raw_text"
INDEX_DIR = BASE_DIR / "rag_index"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.pkl"

# ---------------------------------------------------------------------------
# Chunking (tiktoken, same params as production)
# ---------------------------------------------------------------------------

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64


def chunk_text(text: str, filename: str) -> list[dict]:
    """
    Split text into overlapping token windows.
    Returns list of {"text": str, "metadata": {"file": str, "chunk_index": int}}.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        # Fallback: rough character-based splitting (~4 chars/token)
        logger.warning("tiktoken not found — using character-based chunking")
        char_size = CHUNK_TOKENS * 4
        char_overlap = OVERLAP_TOKENS * 4
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + char_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {"file": filename, "chunk_index": idx},
                })
                idx += 1
            start += char_size - char_overlap
        return chunks

    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens).strip()
        if chunk_str:
            chunks.append({
                "text": chunk_str,
                "metadata": {"file": filename, "chunk_index": idx},
            })
            idx += 1
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks


# ---------------------------------------------------------------------------
# Embedding (BGE-M3, same model as GraphRAG)
# ---------------------------------------------------------------------------

EMBED_BATCH = 32  # chunks per batch — keep memory manageable on CPU


def load_embedding_model():
    logger.info("Loading BAAI/bge-m3 via sentence-transformers (CPU) …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    logger.info("Model loaded.")
    return model


def embed_chunks(model, texts: list[str]) -> np.ndarray:
    """Embed a list of texts in batches. Returns L2-normalised float32 array."""
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i: i + EMBED_BATCH]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs.astype(np.float32))
        if (i // EMBED_BATCH + 1) % 5 == 0:
            done = min(i + EMBED_BATCH, len(texts))
            logger.info("  Embedded %d / %d chunks …", done, len(texts))
    return np.vstack(all_vecs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import faiss

    # 1 — Collect all .txt files
    txt_files = sorted(RAW_TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {RAW_TEXT_DIR}")
        sys.exit(1)

    logger.info("Found %d text files in %s", len(txt_files), RAW_TEXT_DIR)

    # 2 — Chunk all files
    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    for txt_path in txt_files:
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Cannot read %s: %s", txt_path.name, e)
            continue

        text = text.strip()
        if len(text) < 200:
            logger.debug("Skipping short file: %s (%d chars)", txt_path.name, len(text))
            continue

        file_chunks = chunk_text(text, txt_path.name)
        for c in file_chunks:
            all_chunks.append(c["text"])
            all_metadata.append(c["metadata"])

    logger.info(
        "Total chunks: %d from %d papers",
        len(all_chunks),
        len({m["file"] for m in all_metadata}),
    )

    if not all_chunks:
        print("No chunks produced — check raw_text files.")
        sys.exit(1)

    # 3 — Embed
    model = load_embedding_model()
    logger.info("Embedding %d chunks (batch=%d) …", len(all_chunks), EMBED_BATCH)
    vectors = embed_chunks(model, all_chunks)
    logger.info("Embedding done. Shape: %s", vectors.shape)

    dim = vectors.shape[1]  # should be 1024 for BGE-M3

    # 4 — Build FAISS index
    logger.info("Building FAISS IndexFlatIP (dim=%d) …", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info("FAISS index: %d vectors", index.ntotal)

    # 5 — Backup old index
    if FAISS_PATH.exists():
        shutil.copy2(FAISS_PATH, FAISS_PATH.with_suffix(".index.bak"))
        logger.info("Backed up old faiss.index → faiss.index.bak")
    if META_PATH.exists():
        shutil.copy2(META_PATH, META_PATH.with_suffix(".pkl.bak"))
        logger.info("Backed up old metadata.pkl → metadata.pkl.bak")

    # 6 — Save
    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    papers = len({m["file"] for m in all_metadata})
    print("\n" + "=" * 60)
    print("Index rebuilt successfully")
    print(f"  Papers   : {papers}")
    print(f"  Chunks   : {len(all_chunks)}")
    print(f"  Vectors  : {index.ntotal}  (dim={dim})")
    print(f"  Saved to : {INDEX_DIR}")
    print("=" * 60)
    print("\nNext step: start the app")
    print(
        "  $env:OPENBLAS_NUM_THREADS='1'\n"
        "  D:\\Sanshodhak\\.venv\\Scripts\\python.exe app.py"
    )


if __name__ == "__main__":
    main()
