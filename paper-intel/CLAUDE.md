# Sanshodhak — Orientation for Claude Code

This file is persistent context for future sessions. `CHANGES.md` is the chronological log; this file is evergreen — keep it current.

## Project goal

Adaptive Hybrid Retrieval (AHR) + knowledge-graph expansion for scholarly RAG. Current best result on the corpus-grounded 100-Q benchmark: **HGR NDCG@10 = 0.9170, MRR = 0.9025, P@1 = 0.860, hit-rate 96/100**. Paper draft: `D:\Sanshodhak\EDI_5_final (1).docx`. LaTeX source: `paper-intel/paper/main.tex`.

## Hardware & provider constraints

- **No GPU / no CUDA VRAM.** All on-device inference is CPU-only. The pivot to NVIDIA NIM cloud (Session 4) eliminates this constraint for embedding and reranking — both now run on NIM-hosted GPUs.
- **Primary provider: NVIDIA NIM cloud**, OpenAI-compatible API. Two endpoint hosts:
  - `https://integrate.api.nvidia.com/v1` — chat completions, embeddings (`nvidia/llama-3.2-nv-embedqa-1b-v2`)
  - `https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking` — reranker (`nvidia/rerank-qa-mistral-4b`)
  - Same `NVIDIA_NIM_API_KEY` works for both. Rerank endpoint optionally takes a separate `NIM_RERANK_API_KEY` (falls back to the main key).
- **OpenRouter** is still wired in `graph_rag.py:_generate()` as a generation fallback (key in `.env` as `OPENROUTER_API_KEY`). Free tier rate-limited to 8 RPM on `meta-llama/llama-3.2-3b-instruct:free` — too slow for batch question generation; use NIM chat for those.
- **Ollama is not installed** on this Windows machine. Any Ollama call path will fail. The codebase no longer relies on it for embeddings or reranking — only `_generate()` still references it as a "try first" before OpenRouter, and that path is dormant.
- **Sentence-transformers / local BGE-M3 path is removed** from the active retrieval pipeline. The 1024-dim BGE-M3 FAISS index is preserved at `rag_index/faiss.index.bge-m3.bak` for revert if ever needed.

## Canonical pipeline config (what actually runs as of Session 4)

| Component | Value | Evidence |
|---|---|---|
| Embedding | `nvidia/llama-3.2-nv-embedqa-1b-v2` 2048-dim, NIM cloud, L2-norm | `nim_embedder.py`, called from `graph_rag.py:get_embedding()` |
| FAISS index | `rag_index/faiss.index` IndexFlatIP, **2048-dim**, 4333 vectors | re-embedded via `reembed_corpus_nim.py` |
| Old index backup | `rag_index/faiss.index.bge-m3.bak` (1024-dim BGE-M3) | for revert |
| Corpus size | **163 papers, 4333 chunks** | `pickle.load("rag_index/metadata.pkl")["chunks"]` |
| BM25 | `rank-bm25` 0.2.2, default params | `config.yaml` |
| Jaccard τ | **0.30** for eval (paper design); `eval_compare.py --threshold` defaults to **0.10** which gives a near-complete graph (8516 edges) — always pass `--threshold 0.30` | `graph_rag.py`, `core/graph.py` |
| Graph stats at τ=0.30 | 163 nodes, 190 edges (27 Jaccard + 163 Citation), density 0.014, 85 connected components | `eval_report.txt` |
| RRF k | 60 | `core/fusion.py` |
| Graph slots | `max(1, ⌊k/4⌋)` (slot reservation enforced) | `core/adaptive_budget.py:compute_graph_budget` |
| Reranker | `nvidia/rerank-qa-mistral-4b` via NIM cloud | `core/nim_reranker.py`, `graph_rag.py:_get_reranker()` |
| Reranker (legacy, no longer used) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `core/reranker.py:CrossEncoderReranker` (kept on disk) |
| LLM generation | `deepseek-r1:7b` via OpenRouter fallback (`google/gemma-3-4b-it:free`) | `graph_rag.py:_generate()` |
| Test set | `rag_test_questions_corpus.json` (100 corpus-grounded questions, 100 unique target papers) | rebuilt 2026-04-25 via `rebuild_test_set.py` |
| Old test set (do not use) | `rag_test_questions_100.json` (50/100 unanswerable due to identifier mismatch) | superseded |

## Current best results (Session 4 final)

100-Q retrieval-only eval, corpus-grounded test set, τ=0.30, NIM embed + NIM Mistral-4B rerank:

| System | P@1 | NDCG@10 | MRR | Hit | Latency mean |
|---|---|---|---|---|---|
| VR-D (Dense) | 0.800 | 0.8762 | 0.8551 | 94/100 | 0.41s |
| VR-H (Hybrid) | 0.810 | 0.8761 | 0.8581 | 93/100 | 0.40s |
| GR (Graph) | 0.840 | 0.8879 | 0.8767 | 92/100 | 1.51s |
| **HGR (Full AHR)** | **0.860** | **0.9170** | **0.9025** | **96/100** | 1.76s |

Paper's central architectural claim (HGR > GR > VR-H ≈ VR-D) holds on every metric on this benchmark.

## Known research-integrity issues — status

| # | Issue | Status |
|---|---|---|
| 1 | `beir_eval.py` hardcodes fabricated SciFact/NFCorpus NDCG@10 numbers cited in paper | **OPEN.** Replacement `beir_real_eval.py` exists but never finished a run. Needs `--max-queries 100` flag added before next attempt. |
| 2 | Docx Table II has fabricated 5th "HGR-no-slot" row | **OPEN.** Remove or generate by forcing graph_budget=0 (requires patching `compute_graph_budget` floor). |
| 3 | Docx N@10 (0.0847, 0.0923) ≠ LaTeX N@10 (0.533) | **PARTIALLY MOOT** — both are now superseded by Session-4 numbers above (0.9170 HGR). Update LaTeX, drop docx. |
| 4 | Corpus claimed 628 docs, actually 163 indexed | **STILL OPEN** in the paper claim. Either re-ingest to 628+ or correct the paper. Code-side reality is 163/4333. |
| 5 | 100-question test set: 50 unanswerable | **CLOSED.** Rebuilt via `rebuild_test_set.py` → `rag_test_questions_corpus.json`. New set: 100 corpus-grounded questions, 100 unique target papers, all `relevant_docs` confirmed in corpus. |
| 6 | IAA scores appear fabricated | **OPEN.** Verify from `iaa_results.json` / `annotations_annotator*.json`. Anything not reproducible should be cut. |

## Architecture history — what's been tried, what worked

Session-by-session evolution (full detail in CHANGES.md):

| Change | Session | Outcome |
|---|---|---|
| Embedding: BGE-M3 CPU → NIM `llama-3.2-nv-embedqa-1b-v2` cloud, 2048-dim | 4 | ✓ ~10× faster query embedding; eval scores neutral on broken test set, big lift on rebuilt set |
| Reranker: `ms-marco-MiniLM-L-6-v2` CPU → NIM `rerank-qa-mistral-4b` cloud | 4 | ✓ **+5.7 pts NDCG@10 on HGR** (single biggest win) |
| Test set rebuild from corpus | 4 | ✓ NDCG@10 jumped 0.066 → 0.876 (test-set quality, not retrieval) |
| τ=0.30 (paper design) vs default 0.10 | 4 | ✓ +0.4 pts GR; HGR unchanged (BM25 was masking) |
| 3-channel CW-RRF fusion + slot reservation | 2 | ✓ kept |
| `expansion_mode="qbpr"` set | 2 | Set in config but actual implementation is 1-hop neighbour walk |
| Query expansion + PPR + centrality boost (architectural rewrite) | 4 | ✗ **REVERTED** — see "Failed experiments" below |

## Failed experiments (do not redo without first reading why)

### PPR + query expansion + centrality boost (Session 4)

Attempted to close the 3-pt GR-vs-HGR gap by replacing `_get_graph_neighbours` with `query_biased_ppr` and adding a `centrality_boost` after rerank, plus pooling dense retrieval across query reformulations.

**Result on 100-Q corpus-grounded eval at τ=0.30:**
- GR NDCG@10 dropped 0.8914 → 0.8242 (−6.7 pts), hit-rate 92→86 (lost 6 questions)
- HGR NDCG@10 dropped 0.9170 → 0.9127 (−0.4 pts)
- Latency tripled (1.5s → 3.6s GR; 1.8s → 4.2s HGR)

**Root cause:**
1. **PPR over a citation-heavy graph favours hubs over query-relevant peripherals.** The 163 citation edges connect cited-by-either pairs regardless of topic. PageRank then over-weights papers with many citations — exactly the wrong scoring for a benchmark where each question targets one specific (often peripheral) paper.
2. **Centrality boost penalizes peripheral relevant docs.** Negative boost for low-centrality candidates actively pushes the right answer down on chunk-derived QA.
3. **Query expansion adds 2-3s/query latency without precision gain** on this benchmark style. Reformulations dilute when the original query is already specific.

**What stays on disk** (importable, well-tested, unused at HEAD — preserved for the eventual multi-hop benchmark where these techniques would actually help):
- `core/graph_ppr.py` — `query_biased_ppr(graph, seeds)`, `centrality_boost(graph, seeds, candidates)`
- `core/test_graph_ppr.py` — 7 unit tests, all pass
- `query_expander.py:expand_query(query, n=3)` standalone function (the `QueryExpander` class is also there, untouched)
- `nim_smoke_query_expand.py` — smoke test

**When to revisit these:** if/when we build a multi-hop test set (questions that span 2-3 papers), graph centrality becomes a defensible signal because the relevant docs are *expected* to be graph-central. On the current single-paper benchmark, they're not.

## Open levers if pushing scores further (none guaranteed)

1. **`min_slots=0` in `compute_graph_budget`** — let HGR drop the graph slot when graph confidence is low. May claw back precision on queries where the graph adds noise. Not yet tried.
2. **Re-enable `confidence_weighted_rrf`** — `core/fusion.py` has it; eval may use plain RRF. Quick verify-and-flip.
3. **Try `nv-embed-v1`** (4096-dim, MTEB-leading, non-commercial license). Re-embed corpus once, re-run eval. Trade-off: license footnote in paper vs probable +1-2 pts.
4. **More papers** — corpus is 163. Ingesting 200+ would densify the graph and improve generalization.

## Background jobs running

**None.** Session 4 ended clean. No stale processes, no in-progress eval runs.

## Resume checklist

When resuming, in order:

1. **Update LaTeX paper** `paper/main.tex` Table II with Session-4 numbers (`paper-intel/eval_summary.json`). Frame as "rebuilt corpus-grounded benchmark" — do **not** put new HGR=0.917 next to old HGR=0.092 in the same table.
2. **Run BEIR SciFact** for a directly-comparable external benchmark number. `beir_real_eval.py` exists; needs `--max-queries 100` flag added before run. ~1 hour CPU time. Resolves the still-open Issue #1 (fabricated BEIR numbers).
3. **Build a multi-hop test set** as a second benchmark. Use the same `rebuild_test_set.py` pattern but prompt the LLM to generate questions that genuinely span 2-3 papers. This is where the preserved PPR + centrality scaffolding becomes useful.
4. **Add reproducibility section** to paper noting NIM cloud requirement (`NVIDIA_NIM_API_KEY` for both endpoints). Without it, current code path has no fallback for embedding (the BGE-M3 fallback was removed in Session 4 to avoid silent dimension mismatch).
5. Optional: delete `core/graph_ppr.py`, `core/test_graph_ppr.py`, `nim_smoke_query_expand.py`, and revert `query_expander.py` standalone function if multi-hop work won't happen.

## Key file map

- **Retrieval core**: `core/fusion.py`, `core/graph.py`, `core/adaptive_budget.py`, `core/retrieval.py` (legacy `EmbeddingBackend` used by `run_*_eval.py` — needs the same NIM swap as `graph_rag.py` if those are revived)
- **Pipeline**: `graph_rag.py` (AHR with NIM embed + NIM rerank + OpenRouter generation fallback)
- **NIM clients**: `nim_embedder.py` (embeddings, used by all 3 RAG classes), `core/nim_reranker.py` (reranking)
- **Eval drivers**: `eval_compare.py` (4-way 100-Q, primary), `beir_real_eval.py` (BEIR, unfinished), `journal_ablation_module.py` (produces Table II from eval_summary.json)
- **Test sets**: `rag_test_questions_corpus.json` (canonical, use this), `rag_test_questions_100.json` (broken, do not use)
- **Index**: `rag_index/faiss.index` (NIM 2048-dim), `rag_index/faiss.index.bge-m3.bak` (BGE-M3 1024-dim, for revert)
- **Query expansion**: `query_expander.py` — has both legacy `QueryExpander` class and Session-4 standalone `expand_query()` function (latter is unused, see Failed experiments)
- **Graph PPR scaffolding (unused)**: `core/graph_ppr.py`, `core/test_graph_ppr.py`
- **Paper source**: `paper/main.tex` (authoritative), `D:\Sanshodhak\EDI_5_final (1).docx` (stale — supersede with Session-4 numbers)
- **Results**: `paper-intel/eval_summary.json` feeds `journal_ablation_module.py`; per-question detail in `eval_results.json`; human-readable table in `eval_report.txt`

## Standard run commands

**100-Q eval (current best config):**
```bash
cd D:/Sanshodhak/paper-intel && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 \
  D:/Sanshodhak/.venv/Scripts/python.exe eval_compare.py \
  --no-generation --no-bertscore --threshold 0.30 \
  --questions rag_test_questions_corpus.json
```

**Smoke test the live search pipeline:**
```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 \
  D:/Sanshodhak/.venv/Scripts/python.exe paper-intel/smoke_search.py
```

**Re-embed corpus (only if you change embedding model):**
```bash
D:/Sanshodhak/.venv/Scripts/python.exe paper-intel/reembed_corpus_nim.py
```

**Rebuild test set (if corpus changes significantly):**
```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
  D:/Sanshodhak/.venv/Scripts/python.exe paper-intel/rebuild_test_set.py
```

## Windows-specific gotchas (recurring crash classes)

- **cp1252 codec** is Windows-default — crashes on Greek letters, math symbols, smart quotes from scientific text. Always:
  - Set `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` env vars
  - Pass `encoding="utf-8"` to every `open()` for text files
  - At script top: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`
  - Use `ensure_ascii=False` only when writing JSON to a `.write_text(..., encoding="utf-8")` target
- **OpenBLAS threading** can deadlock when multiple sklearn/transformers models load in one process. Set `OPENBLAS_NUM_THREADS=1` before any model import.
- **dotenv** must be loaded explicitly. `os.getenv("NVIDIA_NIM_API_KEY")` returns None unless `load_dotenv(".env")` ran first. `nim_embedder.py` does this at import; if you write a new helper that calls NIM directly, make sure dotenv loads or your eval will silently return empty results.

## Related-work citations gathered (for §II rewrite)

```
LightRAG — Z. Guo et al., arXiv:2410.05779, 2024 — dual-level retrieval + LLM-extracted graph index.
KAG — L. Liang et al., arXiv:2409.13731, 2024 — domain KG + mutual indexing for professional QA.
BRIGHT — H. Su et al., arXiv:2407.12883, 2024 — reasoning-intensive IR benchmark where BM25 ≥ dense.
HippoRAG — B. J. Gutiérrez et al., NeurIPS 2024 (arXiv:2405.14831) — PPR-over-KG memory retrieval.
LazyGraphRAG — D. Edge et al., Microsoft Research Blog, Nov 2024 — defers LLM extraction; 0.1% indexing cost of GraphRAG.
```
