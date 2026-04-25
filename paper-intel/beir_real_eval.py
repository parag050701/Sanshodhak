"""Real SciFact BEIR evaluation. Replaces hardcoded placeholders in beir_eval.py.

Runs three configurations and reports NDCG@{1,5,10}, MRR@10, Recall@{10,100}:
  - VR-D:          BGE-M3 dense retrieval
  - VR-H:          BM25 + dense, fused via RRF (k=60)
  - VR-H+Rerank:   VR-H top-50 reranked with cross-encoder/ms-marco-MiniLM-L-6-v2
"""
import json
import time
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi


DATA_PATH = str(Path(__file__).parent / "beir_datasets" / "scifact")
OUT_PATH = str(Path(__file__).parent / "beir_results_real.json")
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


def rrf_fuse(rankings_list, top_n=100):
    fused = {}
    qids = rankings_list[0].keys()
    for qid in qids:
        scores = {}
        for rankings in rankings_list:
            sorted_docs = sorted(rankings[qid].items(), key=lambda x: -x[1])
            for rank, (did, _) in enumerate(sorted_docs, 1):
                scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
        fused[qid] = dict(sorted(scores.items(), key=lambda x: -x[1])[:top_n])
    return fused


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.0f}s] Loading SciFact test split...")
    corpus, queries, qrels = GenericDataLoader(DATA_PATH).load(split="test")
    print(f"  corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
        for d in doc_ids
    ]
    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]

    print(f"[{time.time()-t0:.0f}s] Loading embedder {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    print(f"[{time.time()-t0:.0f}s] Encoding {len(doc_texts)} docs...")
    doc_embs = embedder.encode(
        doc_texts, batch_size=8, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    print(f"[{time.time()-t0:.0f}s] Encoding {len(qtexts)} queries...")
    q_embs = embedder.encode(
        qtexts, batch_size=16, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )

    print(f"[{time.time()-t0:.0f}s] Computing dense top-100...")
    scores = q_embs @ doc_embs.T  # cosine since normalized
    top100_d = np.argpartition(-scores, 100, axis=1)[:, :100]
    dense_results = {}
    for qi, qid in enumerate(qids):
        order = top100_d[qi][np.argsort(-scores[qi, top100_d[qi]])]
        dense_results[qid] = {doc_ids[di]: float(scores[qi, di]) for di in order}

    print(f"[{time.time()-t0:.0f}s] Computing BM25 top-100...")
    tokenized = [t.lower().split() for t in doc_texts]
    bm25 = BM25Okapi(tokenized)
    bm25_results = {}
    for qi, qid in enumerate(qids):
        qtoks = qtexts[qi].lower().split()
        bs = bm25.get_scores(qtoks)
        top100_b = np.argpartition(-bs, 100)[:100]
        top100_b = top100_b[np.argsort(-bs[top100_b])]
        bm25_results[qid] = {doc_ids[di]: float(bs[di]) for di in top100_b}

    print(f"[{time.time()-t0:.0f}s] RRF fusion (k={RRF_K})...")
    hybrid_results = rrf_fuse([dense_results, bm25_results], top_n=100)

    print(f"[{time.time()-t0:.0f}s] Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    print(f"[{time.time()-t0:.0f}s] Reranking top-50 per query...")
    rerank_results = {}
    for qi, qid in enumerate(qids):
        top50 = list(hybrid_results[qid].items())[:50]
        pairs = [(qtexts[qi], corpus[did].get("title", "") + " " + corpus[did].get("text", "")) for did, _ in top50]
        ce = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        reranked = sorted(zip([d for d, _ in top50], ce), key=lambda x: -float(x[1]))
        rerank_results[qid] = {did: float(s) for did, s in reranked}
        if qi % 50 == 0:
            print(f"    reranked {qi+1}/{len(qids)}")

    print(f"[{time.time()-t0:.0f}s] Evaluating...")
    evaluator = EvaluateRetrieval()
    configs = {"VR-D": dense_results, "VR-H": hybrid_results, "VR-H+Rerank": rerank_results}
    final = {}
    for name, res in configs.items():
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, res, [1, 5, 10, 100])
        mrr = evaluator.evaluate_custom(qrels, res, [10], metric="mrr")
        final[name] = {
            "NDCG@1": ndcg.get("NDCG@1"), "NDCG@5": ndcg.get("NDCG@5"), "NDCG@10": ndcg.get("NDCG@10"),
            "MRR@10": mrr.get("MRR@10"), "Recall@10": recall.get("Recall@10"), "Recall@100": recall.get("Recall@100"),
            "MAP@10": _map.get("MAP@10"), "P@10": precision.get("P@10"),
        }

    final["_meta"] = {
        "dataset": "SciFact test split",
        "n_corpus": len(corpus), "n_queries": len(queries), "n_qrels": len(qrels),
        "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL, "rrf_k": RRF_K,
        "runtime_sec": round(time.time() - t0, 1),
    }

    Path(OUT_PATH).write_text(json.dumps(final, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Saved {OUT_PATH}")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
