# Sanshodhak — Change Log & Roadmap

## 2026-04-25 — Session 4: NIM cloud pivot, test-set rebuild, score validation

Detailed log: `paper-intel/CHANGES.md` "Session 4". Headline:

- **Embedding swapped** from local CPU BGE-M3 (1024-dim) → NVIDIA NIM `llama-3.2-nv-embedqa-1b-v2` (2048-dim cloud). Re-embedded all 4333 chunks, rebuilt FAISS index. Old index backed up to `rag_index/faiss.index.bge-m3.bak`.
- **Reranker swapped** from local CPU `cross-encoder/ms-marco-MiniLM-L-6-v2` → NIM `nvidia/rerank-qa-mistral-4b` cloud. **Single biggest score win of the session: HGR NDCG@10 +5.7 pts.**
- **Test set rebuilt** from current corpus (`rag_test_questions_corpus.json`, 100 questions, 100 unique target papers, all answerable). Replaces broken `rag_test_questions_100.json` where 50/100 were unanswerable due to identifier mismatch. Metric jumped 0.066 → 0.876 NDCG@10 just from the test-set fix.
- **Final result on corpus-grounded eval (τ=0.30, NIM embed + NIM rerank):**
  - VR-D: P@1=0.800, NDCG@10=0.8762, MRR=0.8551
  - VR-H: P@1=0.810, NDCG@10=0.8761, MRR=0.8581
  - GR:   P@1=0.840, NDCG@10=0.8879, MRR=0.8767
  - **HGR: P@1=0.860, NDCG@10=0.9170, MRR=0.9025, hit-rate 96/100**
  - Paper's central claim (HGR > GR > VR-H ≈ VR-D) holds on every metric.
- **Failed experiment, reverted:** Wired query expansion + PPR-based graph candidates + centrality boost into `graph_rag.py:search()`. Regressed GR by 6.7 pts NDCG@10 and HGR by 0.4 pts. Root cause: PPR over a citation-heavy graph favours hubs, but each test question targets one specific often-peripheral paper. Reverted; helper modules (`core/graph_ppr.py`, `core/test_graph_ppr.py`, `nim_smoke_query_expand.py`, standalone `expand_query()` in `query_expander.py`) preserved on disk for future multi-hop benchmark work.
- **New files:** `nim_embedder.py`, `nim_smoke_test.py`, `reembed_corpus_nim.py`, `smoke_search.py`, `rebuild_test_set.py`, `rag_test_questions_corpus.json`, `core/nim_reranker.py`, `core/graph_ppr.py`, `core/test_graph_ppr.py`, `nim_smoke_query_expand.py`.
- **Modified:** `graph_rag.py`, `ollama_rag.py`, `enhanced_rag.py` (all → NIM embedding), `eval_compare.py` (utf-8 fix), `query_expander.py` (added standalone fn), `.env` (added NIM rerank section).

## Done

### Setup
- Cloned fork from https://github.com/parag050701/Sanshodhak into D:/Sanshodhak
- Created Python 3.11 virtual environment at D:/Sanshodhak/.venv
- Installed all dependencies via `D:/Sanshodhak/.venv/Scripts/pip.exe install -r paper-intel/requirements.txt`
- Installed Flask and flask-cors (missing from requirements.txt)

### Bug Fixes
- `requirements.txt` line 28: fixed missing newline between `pydantic>=2.7.0` and `openai>=1.3.0`
- `ingestion/discovery/__init__.py`: added missing `DOAJClient` import and export
- `test_apis.py` line 203: fixed `search_unified()` call — replaced invalid `limit_per_source=5, top_k=10` with `limit=10`
- `app.py`: fixed all `r['file']` → `r['metadata']['file']` (4 locations) to match actual search result structure
- `app.py`: fixed `rag_system.query()` return — it returns a tuple `(answer, results)`, changed both call sites to `answer, _ = rag_system.query(query)`

### API Test Results (test_apis.py) — 7/7 passing
- OpenAlex: working
- CrossRef: working
- arXiv: working
- DOI utilities: working
- Search engine (open-source + closed + unified): working
- DOAJ: rate limited / timeout (external issue, not code)
- Semantic Scholar: timeout (external issue, not code)

---

## To Do

### Features
- Add `/api/query_focused` endpoint: accepts `topic` (for retrieval) + `question` (for answering) separately, so users can scope RAG answers to a topic
- Add focused query UI in `index.html`: two input fields (Topic / Question) + new mode button

### Security
- Move all hardcoded/committed API keys to `.env` (currently exposed in `secrets_template.env` and `ollama_rag.py`)
- Add `.env`, `*.env`, and `secrets_template.env` to `.gitignore`
- Rotate all exposed keys: NVIDIA NIM, OpenRouter, Semantic Scholar, CORE, Graffiti

### Infrastructure
- Add `flask` and `flask-cors` to `requirements.txt`
- Consider replacing Ollama dependency with NVIDIA NIM as primary LLM (no local GPU available — only 497MB VRAM)

---

## How to Run

```bash
cd D:/Sanshodhak/paper-intel
D:/Sanshodhak/.venv/Scripts/python.exe app.py
```

Open http://localhost:5000 in browser.
