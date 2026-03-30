#!/usr/bin/env python3
"""
GPU-accelerated FAISS embedding builder for 10k corpus.

Builds dense embeddings for entire merged corpus, batches them via GPU,
and exports indexed FAISS index for fast retrieval.

Usage:
  python build_gpu_embeddings.py --corpus data/corpus_merged.json \
    --output indexes/faiss_10k \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --batch_size 256
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

# CPU device by default; GPU available via torch device config
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU FAISS embeddings for corpus")
    parser.add_argument("--corpus", required=True, help="Path to merged corpus JSON")
    parser.add_argument("--output", required=True, help="Output FAISS index directory")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--device", default="cpu", help="torch device (cpu/cuda/mps)")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="FAISS metric")
    return parser.parse_args()


def load_corpus(path: str) -> List[dict]:
    """Load merged corpus JSON."""
    with open(path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    if isinstance(corpus_data, list):
        return corpus_data
    if isinstance(corpus_data, dict) and "papers" in corpus_data:
        return corpus_data["papers"]
    raise ValueError(f"Unexpected corpus format in {path}")


def embed_batch(model, texts: List[str], device: str = "cpu") -> np.ndarray:
    """Embed a batch of texts using SentenceTransformer."""
    embeddings = model.encode(texts, device=device, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, metric: str = "cosine", use_gpu: bool = False) -> faiss.Index:
    """Build FAISS index from embeddings."""
    d = embeddings.shape[1]
    logger.info(f"Building FAISS index: {embeddings.shape[0]} vectors × {d} dims, metric={metric}")

    if metric == "cosine":
        # Normalize for cosine similarity (already done in embed_batch)
        index = faiss.IndexFlatIP(d)  # Inner product on normalized vectors = cosine
    else:  # l2
        index = faiss.IndexFlatL2(d)

    if use_gpu:
        logger.info("Moving index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    logger.info("Adding vectors to index...")
    index.add(embeddings)

    logger.info(f"Index built: {index.ntotal} vectors indexed")
    return index


def main():
    if not _FAISS_AVAILABLE or not _ST_AVAILABLE:
        logger.error("Required packages not found. Install: pip install faiss-cpu sentence-transformers")
        return

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    logger.info(f"Loaded {len(papers)} papers")

    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    model.to(args.device)

    logger.info(f"Extracting texts from papers...")
    texts = []
    paper_ids = []
    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
        if text:
            texts.append(text)
            paper_ids.append(paper.get("doi") or paper.get("title") or f"unknown_{len(texts)}")

    logger.info(f"Will embed {len(texts)} papers in batches of {args.batch_size}...")

    all_embeddings = []
    start_time = time.time()

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Batching embeddings"):
        batch_texts = texts[i : i + args.batch_size]
        batch_embeddings = embed_batch(model, batch_texts, device=args.device)
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - start_time
    logger.info(f"Embedded {len(texts)} papers in {elapsed:.1f}s ({len(texts)/elapsed:.1f} papers/s)")

    logger.info("Building FAISS index...")
    use_gpu = args.device.startswith("cuda")
    index = build_faiss_index(all_embeddings, metric=args.metric, use_gpu=use_gpu)

    index_path = output_dir / "index.faiss"
    logger.info(f"Saving index to {index_path}...")
    faiss.write_index(faiss.index_gpu_to_cpu(index) if use_gpu else index, str(index_path))

    metadata_path = output_dir / "metadata.json"
    logger.info(f"Saving metadata to {metadata_path}...")
    metadata = {
        "model": args.model,
        "metric": args.metric,
        "num_vectors": all_embeddings.shape[0],
        "embedding_dim": all_embeddings.shape[1],
        "paper_ids": paper_ids,
        "corpus_path": args.corpus,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ FAISS index built and saved to {output_dir}")
    logger.info(f"   - Index: {index_path}")
    logger.info(f"   - Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
