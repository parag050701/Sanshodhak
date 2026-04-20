"""
Build the Sanshodhak knowledge graph in Neo4j from an existing rag_index.

Usage:
    python build_neo4j_graph.py [--index rag_index]

What it does:
  1. Loads chunks + metadata from rag_index/metadata.pkl
  2. Reconstructs embeddings from rag_index/faiss.index
  3. Aggregates chunks → papers (one node per unique file)
  4. Builds the knowledge graph in Neo4j via Neo4jGraphIndex.build()
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from core.neo4j_graph import Neo4jGraphIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="rag_index",
                        help="Path to the rag_index directory")
    args = parser.parse_args()

    index_dir = Path(args.index)
    pkl_path   = index_dir / "metadata.pkl"
    faiss_path = index_dir / "faiss.index"

    # ── 1. Load chunks & metadata ─────────────────────────────────────────
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    chunks:   list = data["chunks"]
    metadata: list = data["metadata"]
    n_chunks = len(chunks)
    print(f"  {n_chunks} chunks loaded")

    # ── 2. Reconstruct embeddings from FAISS ──────────────────────────────
    print(f"Loading {faiss_path}...")
    faiss_index = faiss.read_index(str(faiss_path))
    print(f"  FAISS index: {faiss_index.ntotal} vectors, dim={faiss_index.d}")

    print("Reconstructing embeddings...")
    all_embeddings = np.zeros((n_chunks, faiss_index.d), dtype="float32")
    batch = 512
    for start in range(0, n_chunks, batch):
        end = min(start + batch, n_chunks)
        all_embeddings[start:end] = faiss_index.reconstruct_n(start, end - start)
    print(f"  Embeddings shape: {all_embeddings.shape}")

    # ── 3. Aggregate chunks → papers ──────────────────────────────────────
    print("Grouping chunks by paper...")
    paper_chunks:  dict = defaultdict(list)   # paper_id → [chunk indices]
    for idx, meta in enumerate(metadata):
        paper_chunks[meta["file"]].append(idx)

    paper_ids  = sorted(paper_chunks.keys())
    n_papers   = len(paper_ids)
    print(f"  {n_papers} unique papers")

    # For each paper: concatenate all chunk texts, average chunk embeddings
    paper_texts      = []
    paper_embeddings = []
    paper_metadata   = []

    for paper_id in paper_ids:
        indices = paper_chunks[paper_id]
        text    = " ".join(chunks[i] for i in indices)
        emb     = all_embeddings[indices].mean(axis=0)
        # L2-normalise (FAISS uses inner product on normalised vectors)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        paper_texts.append(text)
        paper_embeddings.append(emb)
        paper_metadata.append({"title": paper_id.replace("_", " ").replace(".txt", "")})

    paper_embeddings_arr = np.stack(paper_embeddings).astype("float32")

    # ── 4. Build Neo4j graph ───────────────────────────────────────────────
    print("\nConnecting to Neo4j...")
    kg = Neo4jGraphIndex(jaccard_threshold=0.30)

    print(f"Building graph for {n_papers} papers (τ=0.30)...")
    kg.build(
        doc_ids   = paper_ids,
        texts     = paper_texts,
        embeddings= paper_embeddings_arr,
        metadata  = paper_metadata,
    )

    stats = kg.get_stats()
    print("\n" + "="*50)
    print("  Neo4j Graph Built")
    print("="*50)
    print(f"  Nodes:       {stats['nodes']}")
    print(f"  Edges:       {stats['edges']}")
    print(f"  Avg degree:  {stats.get('avg_degree', 0):.3f}")
    print(f"  Density:     {stats.get('density', 0):.6f}")
    print(f"  Edge types:  {stats.get('edge_types', {})}")
    print("="*50)
    print("\nDone. Open Neo4j Browser and run:")
    print("  MATCH (n:Document) RETURN count(n)")
    print("to verify the graph.")


if __name__ == "__main__":
    main()
