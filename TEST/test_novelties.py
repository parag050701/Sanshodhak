import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../paper-intel')))
import numpy as np

print("--- Testing CWRRF ---")
try:
    from core.fusion import confidence_weighted_rrf
    dense = [("d1", 0.95), ("d2", 0.40), ("d3", 0.38)]
    bm25 = [("d2", 12.0), ("d1", 11.5), ("d4", 11.3)]
    res = confidence_weighted_rrf([dense, bm25])
    print("CWRRF output:", res)
    assert res[0][0] == "d1", "CWRRF ranking unexpected"
    print("CWRRF OK\n")
except Exception as e:
    print("CWRRF Error:", e)

print("--- Testing MMR ---")
try:
    from core.diversity import mmr_rerank
    results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}, {"id": 3, "score": 0.7}]
    query = np.array([1, 0, 0])
    doc1 = np.array([1, 0, 0])
    doc2 = np.array([0.9, 0.1, 0])
    doc3 = np.array([0, 1, 0])
    res = mmr_rerank(results, query, 0.5, [doc1, doc2, doc3])
    print("MMR output:", [r["id"] for r in res])
    assert res[0]["id"] == 1, "MMR ranking unexpected"
    print("MMR OK\n")
except Exception as e:
    print("MMR Error:", e)

print("--- Testing Query Classifier ---")
try:
    from core.query_classifier import classify_query
    q1 = classify_query("transformer attention")
    print("q1:", q1)
    assert q1["class"] == "keyword_focused"
    q2 = classify_query("what is the impact of multi-head attention on transformers in NLP?")
    print("q2:", q2)
    assert q2["class"] == "conceptual"
    print("Classifier OK\n")
except Exception as e:
    print("Classifier Error:", e)

print("--- Testing GraphRAG ---")
try:
    from graph_rag import GraphRAG
    rag = GraphRAG(expansion_mode="qbpr")
    print("GraphRAG instantiated successfully with expansion_mode=qbpr")
    print("GraphRAG OK\n")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("GraphRAG Error:", e)
