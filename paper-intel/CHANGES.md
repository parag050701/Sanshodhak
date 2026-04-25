# Sanshodhak — Change Log

## Session 4 (2026-04-25) — embedding + reranker swap to NIM, test-set rebuild

### Goal this session
Resume from Session 3 paused state. User priority: real, defensible numbers, not the fabricated 0.0923 NDCG@10 in the docx. Complete the pivot to NVIDIA NIM cloud (no local GPU) for both embedding and reranking, then re-validate the full pipeline.

### Headline result
**HGR NDCG@10 = 0.9170, MRR = 0.9025, P@1 = 0.860** on a corpus-grounded 100-Q test set. System ordering HGR > GR > VR-H ≈ VR-D matches the paper's central architectural claim, on every metric, after the test-set rebuild.

| System | P@1 | NDCG@10 | MRR | Hit-rate | Latency |
|---|---|---|---|---|---|
| VR-D (Dense) | 0.800 | 0.8762 | 0.8551 | 94/100 | 0.41s |
| VR-H (Hybrid) | 0.810 | 0.8761 | 0.8581 | 93/100 | 0.40s |
| GR (Graph) | 0.840 | 0.8879 | 0.8767 | 92/100 | 1.51s |
| **HGR (Full AHR)** | **0.860** | **0.9170** | **0.9025** | **96/100** | 1.76s |

Caveat: not directly comparable to the prior 0.0923 number — different test set, different embedding, different reranker. Frame in the paper as "rebuilt corpus-grounded benchmark, results in Table Y" — don't blend old/new in one table.

### What was changed (in execution order)

**Embedding pipeline — swapped from CPU BGE-M3 to NVIDIA NIM cloud**
- `nim_embedder.py` (new) — centralized NIM embedding client. Loads `.env` at import (defensive — the eval driver was crashing with `NVIDIA_NIM_API_KEY not set` because it never called `load_dotenv` itself). OpenAI-compatible client pointed at `https://integrate.api.nvidia.com/v1`. Returns L2-normalised 2048-dim vectors directly (no renormalization needed).
- `nim_smoke_test.py` (new) — verifies endpoint reachability, dimension, batching, query/passage `input_type` switch.
- `reembed_corpus_nim.py` (new) — backed up old 1024-dim BGE-M3 index to `rag_index/faiss.index.bge-m3.bak`, then re-embedded all 4333 chunks via NIM (`input_type="passage"`) at batch_size=32, 195s total. Built fresh `IndexFlatIP(2048)`.
- `graph_rag.py:get_embedding()`, `ollama_rag.py:get_embedding()`, `enhanced_rag.py:get_embedding()` all delegate to `nim_embedder.embed_text(text, input_type="query")`. `enhanced_rag.py` default `rerank_model` also changed from `BAAI/bge-reranker-v2-m3` → `cross-encoder/ms-marco-MiniLM-L-6-v2` (resolves the 2GB stalled download from Session 3).
- Model: `nvidia/llama-3.2-nv-embedqa-1b-v2` — 2048-dim, QA-tuned, 8k context, commercial-OK license. Picked over `nv-embed-v1` (4096-dim, MTEB-leading but non-commercial) for license cleanliness.

**Test set rebuild — fixes the bottleneck identified in Session 3**
- Old `rag_test_questions_100.json`: 50 questions had identifier mismatches; only ~10/100 ever scored anything regardless of retrieval quality. Confirmed by audit (only 6/12 expected docs in current corpus).
- `rebuild_test_set.py` (new) — samples 100 papers from current corpus, picks a mid-document chunk per paper, prompts NIM `nvidia/llama-3.3-nemotron-super-49b-v1` to generate ONE specific factual question + answer. Single LLM call per paper, JSON parsing with regex resilience, retries with exponential backoff.
- Schema matches existing test file: `{id, question, answer, paper_id, relevant_docs:[filename.txt], type, difficulty, stratum}`.
- Output: `rag_test_questions_corpus.json` — 100 questions, 100 unique target papers (no paper hijacking multiple grounds), all `relevant_docs` confirmed in corpus.
- Topic diversity: 50% RAG/IR, 18% Graph/KG, 12% LLM-techniques, 10% NLP, 10% Adversarial-ML, 9% Bio/Med, plus Physics, Chemistry, Code, Semantic-Web. Multi-domain, RAG-heavy.
- Metric jump: NDCG@10 went from 0.066 (broken set) → 0.876 (this set, dense baseline). The 13× lift was test-set quality, not retrieval quality.

**Reranker — swapped from local cross-encoder to NIM Mistral-4B**
- `core/nim_reranker.py` (new) — drop-in replacement for `core/reranker.py:CrossEncoderReranker`. Same `.available`, `.rerank(query, candidates, top_k, text_key)`, `.rerank_ids()` interface. Calls `https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking` with model `nvidia/rerank-qa-mistral-4b`. Falls back to `NVIDIA_NIM_API_KEY` when `NIM_RERANK_API_KEY` is unset.
- `graph_rag.py:_get_reranker()` now instantiates `NIMReranker` instead of `CrossEncoderReranker`.
- Endpoint discovery note: NIM rerankers live on `ai.api.nvidia.com`, NOT `integrate.api.nvidia.com` (where embeddings live). The integrate-host model list (135 models) returns zero rerankers. Probed 4 endpoint+model combos to find the working pair.
- `.env` got a new section for `NIM_RERANK_API_KEY` / `NIM_RERANK_BASE_URL` / `NIM_RERANK_MODEL`. Code falls back to the main NIM key if the rerank-specific one is empty.

**Reranker swap impact (only GR/HGR were affected — VR-D has no reranker; VR-H runs `use_rerank=False`):**
- GR NDCG@10: 0.8456 → 0.8879 (+4.2 pts)
- HGR NDCG@10: 0.8599 → 0.9170 (+5.7 pts)
- This was the single biggest win of the session.

**τ sweep — fixed the broken default**
- `eval_compare.py --threshold` defaults to 0.10. At τ=0.10 the graph has 8516 edges over 163 nodes (density 0.645) — essentially a complete graph, no selectivity.
- Diagnostic showed Jaccard weights between papers rarely exceed 0.30. Built graphs at τ ∈ {0.10, 0.30, 0.50, 0.70} and confirmed:
  - τ=0.30: 190 edges (27 Jaccard + 163 Citation), density 0.014 — matches the paper's design target
  - τ=0.50: 180 edges, plateaus
  - τ≥0.70: graph becomes citation-only (Jaccard nearly empty)
- τ=0.30 sweep impact: GR +0.4 pts NDCG@10; HGR unchanged (BM25 was already compensating for the noisy default graph). Not the big win we hoped for, but it makes the graph stats sane and matches the paper's claimed design.

### Failed experiment — query expansion + PPR + centrality boost (REVERTED)

After the above wins, attempted three architectural changes targeting the GR-vs-HGR gap (3 pts NDCG@10). Used two parallel subagents to build helper modules, then integrated into `graph_rag.py:search()`.

**What was built (kept on disk as scaffolding, not active):**
- `core/graph_ppr.py` (new, by Agent B) — `query_biased_ppr(graph, seeds, ...)` using NetworkX's personalized PageRank, plus `centrality_boost(graph, seeds, candidates, scale)` that z-scores PPR over candidates and clamps to [-scale, +scale]. 7 unit tests in `core/test_graph_ppr.py`, all pass.
- `query_expander.py` modified (by Agent A) — added top-level `expand_query(query, n=3) -> List[str]` returning `[original, conceptual_reform, methodology_reform, synonym_reform]`. Uses NIM `meta/llama-3.1-8b-instruct`, single LLM call, JSON-array output parsed with regex. Returns `[query]` on any failure (never raises). Backward-compatible — kept existing `QueryExpander` class.
- `nim_smoke_query_expand.py` (new) — smoke test for the standalone function.

**What was integrated in `graph_rag.py:search()`:**
1. Replace single-query dense retrieval with multi-variant pooling (4 queries → dedup chunks → keep best score per chunk).
2. Replace `_get_graph_neighbours` (1-hop, dense seed) with `query_biased_ppr` for graph candidate generation.
3. After cross-encoder rerank, add `centrality_boost` as additive logit, then re-sort.

**Result on 100-Q corpus-grounded eval:**

| System | Before integration | After 3 changes | Δ |
|---|---|---|---|
| GR NDCG@10 | 0.8914 | 0.8242 | **-6.7 pts** ⚠️ |
| GR P@1 | 0.840 | 0.770 | **-7 pts** ⚠️ |
| GR hit-rate | 92/100 | 86/100 | **lost 6 questions** |
| HGR NDCG@10 | 0.9170 | 0.9127 | -0.4 pts |
| HGR latency | 1.76s | 4.19s | **+2.4s** |

**Why it regressed (root-cause analysis):**

1. **PPR over a citation-heavy graph favours hubs over query-relevant peripheral nodes.** The graph at τ=0.30 has 163 citation edges + 27 Jaccard edges. Citation edges connect papers that cite each other, regardless of topic. PageRank then over-weights papers with many citations even when they're tangential to the query. The relevant target paper for a question is often a peripheral, single-purpose paper — exactly the wrong kind of node for PPR to rank highly.

2. **Centrality boost penalizes peripheral relevant docs.** The boost is positive for graph-central candidates and negative for peripheral ones. On a single-chunk-derived test set where the relevant doc is *one specific paper*, often itself peripheral, the boost actively pushes the right answer down.

3. **Query expansion adds latency without commensurate precision gain.** Adds ~2-3s/query (one NIM LLM call ~1s + 4× embedding calls vs 1×). Hit-rate didn't improve; rank ordering didn't improve enough to justify the cost.

4. **GR is hit hardest because it lacks BM25.** HGR has BM25 as a recall safety net; PPR damage is masked. GR has no fallback — when PPR replaces 1-hop, the relevant paper drops out of top-10 entirely.

**Decision: revert all three changes in `graph_rag.py:search()`. Helper modules and smoke test left on disk** for future use (when we build a multi-hop test set, PPR + centrality boost become defensible because graph centrality really would correlate with relevance).

**What stays on disk but is unused at HEAD:**
- `core/graph_ppr.py`, `core/test_graph_ppr.py` — well-tested, importable, unused
- `query_expander.py:expand_query()` standalone function — works, not called
- `nim_smoke_query_expand.py` — smoke test, unused

### Bug fixes worth noting
- `eval_compare.py:529` — added explicit `encoding="utf-8"` to `open(questions_path)`. Default cp1252 on Windows crashed when reading the new test set (Greek phi `Φ`, math symbols).
- `nim_embedder.py` — loads `.env` at module import. Without this, eval ran 400 retrievals all returning `[]` because `NVIDIA_NIM_API_KEY` was never in the process env. Caught only because we re-greped logs after seeing all-zero scores.
- `rebuild_test_set.py` — added `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` and incremental `OUT_PATH.write_text(...)` every 10 questions. Prior crash at Q19 on `Φ` lost 18 generated questions. Same Windows cp1252 bug as commit b564d53.
- `enhanced_rag.py` default `rerank_model` changed to `cross-encoder/ms-marco-MiniLM-L-6-v2` (resolves Session 3 stalled download).

### Files created this session
- `nim_embedder.py`, `nim_smoke_test.py`, `reembed_corpus_nim.py`
- `smoke_search.py` (end-to-end search smoke test)
- `rebuild_test_set.py`, `rag_test_questions_corpus.json`
- `core/nim_reranker.py`
- `core/graph_ppr.py`, `core/test_graph_ppr.py` (kept as scaffolding)
- `nim_smoke_query_expand.py` (kept as scaffolding)
- `rag_index/faiss.index.bge-m3.bak` (backup of pre-NIM 1024-dim index)
- `eval_compare_nim_100q.log`, `eval_compare_corpus.log`, `eval_compare_corpus_nimrr.log`, `eval_compare_corpus_tau030.log`, `eval_compare_corpus_tau050.log`, `eval_compare_corpus_integrated.log`
- `rebuild_test_set.log`

### Files modified this session
- `graph_rag.py`: `get_embedding()` → NIM via `nim_embedder`; `_get_reranker()` → `NIMReranker`. (Integration changes for query expansion / PPR / centrality boost were applied AND REVERTED — file is now back to the proven state.)
- `ollama_rag.py:get_embedding()` → NIM via `nim_embedder`
- `enhanced_rag.py:get_embedding()` → NIM via `nim_embedder`; default `rerank_model` → `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `query_expander.py` — added standalone `expand_query()` (not currently called from search pipeline)
- `eval_compare.py:529` — `open(questions_path, encoding="utf-8")`
- `.env` — added `NIM_RERANK_API_KEY`, `NIM_RERANK_BASE_URL`, `NIM_RERANK_MODEL` lines
- `rag_index/faiss.index` — rebuilt at 2048-dim (old 1024-dim backed up to `.bge-m3.bak`)

### Background jobs running at end of session
**None.** All eval runs completed cleanly. No stale processes.

### Subagents dispatched
2 parallel general-purpose agents in Phase 1 of the failed PPR/expansion experiment. Both delivered to spec, smoke tests passed independently. The integration that combined their outputs is what regressed — not the agents' modules themselves.

### Resume priority order

1. **Update the LaTeX paper** (`paper/main.tex`) with the new Table II numbers from `eval_summary.json`. Frame as "rebuilt corpus-grounded benchmark" to avoid implying these are comparable to prior fabricated 0.0923 result.
2. **Run BEIR SciFact eval** via the still-unfinished `beir_real_eval.py` (need `--max-queries 100` flag added). This produces a number directly comparable to BM25 / DPR / ColBERT / BGE-M3 published baselines, addressing the fabrication issue from Session 3.
3. **Build a multi-hop / synthesis test set** as a second benchmark. Questions that genuinely span 2-3 papers. This is where PPR + centrality boost (preserved on disk) actually win architecturally — would justify the graph component for the paper.
4. **Reproducibility section in paper** — must note: NIM cloud is required to reproduce the 0.917 number. Without `NVIDIA_NIM_API_KEY`, embedding falls back nowhere (we removed the BGE-M3 fallback) — anyone replicating needs a NIM key. Reranker similarly cloud-dependent.
5. Optional cleanup: delete `core/graph_ppr.py`, `core/test_graph_ppr.py`, `nim_smoke_query_expand.py`, and revert `query_expander.py` if we decide we'll never revisit the multi-hop angle.

### Lessons for future sessions

- **Test-set bottlenecks masquerade as retrieval bottlenecks.** A 13× metric jump from purely fixing the eval set is the hint to look for. If a system change doesn't move scores, suspect the metric before suspecting the system.
- **PPR centrality is the wrong objective for chunk-derived single-relevant-doc QA.** It's a great fit for "find related work" but actively wrong for "find the one paper that introduced this method". Match the algorithm to the query distribution.
- **Subagents work well for self-contained helper modules with documented interfaces and smoke tests.** They produced clean code on first try. The risk is in the integration step (which the parent agent does) — that's where the failed assumptions surface.
- **Always swallow exceptions in the wrapper layer with a logged error**, not silent failures. The all-zeros first eval was diagnosable only because `RAGWrapper.retrieve()` logged "ERROR: NVIDIA_NIM_API_KEY not set" — without that we'd have spent hours suspecting the embedding model itself.
- **Windows cp1252 + Unicode in scientific text is a recurring crash class.** Set `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` defensively on every run; force `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at script start; always pass `encoding="utf-8"` to `open()` for any file with paper text. Multiple crashes today from this.

---

## Session 3 (2026-04-24) — paused mid-work

### Goal this session
Raise eval scores above published baselines, change architecture if needed, produce a research-worthy paper. User flagged `EDI_5_final (1).docx` for improvement.

### What was investigated (Wave 1, complete)

**Task #1 — HGR-no-slot = VR-H identity check.** Verdict: variant does NOT exist in code. `journal_ablation_module.py` runs only 4 systems; Table II 5th row was fabricated for the docx. A hypothetical HGR-no-slot *would* correctly reduce to VR-H via the `graph_budget==0` short-circuit in `run_scifact_eval.py:309`, but `compute_graph_budget` enforces `min_slots=1` so kg=0 is unreachable in practice. Remove the row or generate it by forcing graph_budget=0.

**Task #2 — Test-set audit.** Findings:
- 50/100 questions unanswerable due to filename / expected-paper-ID mismatch.
- 628-doc corpus claim vs actual 163 docs indexed (~3383 chunks / ~20 per doc).
- Rebuild proposal: sample 50-100 docs from the real index, author 1-2 factual questions per doc, encode expected doc ID as the filename hash. Eliminates the fuzzy-match cascade.

**Task #3 — Config verification.** Canonical config reconciled (see `CLAUDE.md` for table). Key corrections needed in docx:
- Table I "Embedding model": `all-MiniLM-L6-v2 (384-dim)` → `BAAI/bge-m3 (1024-dim, sentence-transformers, CPU)`
- Table I "LLM model": `llama3.2:latest` → `deepseek-r1:7b` (or the OpenRouter fallback actually used)
- §III.B "Ollama runtime" → `sentence-transformers (Ollama fallback)`
- Acknowledgments: MiniLM and BGE-M3 roles are reversed — BGE-M3 is the *primary* eval embedding; MiniLM is only the scalability proxy (Table VII)

**Task #4 — Missing citations gathered.** LightRAG, KAG, BRIGHT, HippoRAG, LazyGraphRAG — refs in `CLAUDE.md`.

**Task #5 — Formatting audit.** Full punch-list produced: run-together words throughout, missing Roman numerals on all section headings, author-block mangling, spacing issues. Ready to apply mechanically when we return to the docx.

**Task #8 — Pipeline bottleneck diagnosis.** Top-5 score-improving changes identified; see `CLAUDE.md` "Architecture improvements to land."

### Critical research-integrity findings

1. **`beir_eval.py` is fabricated** — hardcoded `{"SciFact": {"NDCG@10": 0.682, "MRR": 0.610}, ...}` and `p_value=0.034`. These are cited in the paper as measured BEIR results. Replacement file `beir_real_eval.py` was written this session (real BEIR loader + BGE-M3 + BM25 + RRF + cross-encoder rerank) but has not produced results yet.
2. **Docx Table II has a fabricated 5th "HGR-no-slot" row** not present in code or LaTeX.
3. **Docx and LaTeX report different numbers.** Docx N@10 ≈ 0.08, LaTeX N@10 ≈ 0.53. LaTeX is closer to truth; docx appears to be an earlier weaker-numbers draft. Do not feed the docx back into improvement work — work from `paper/main.tex`.
4. **`iaa_results.json` / annotator files** need verification — paper's IAA claims may be fabricated.

### What was created this session

- `paper-intel/beir_real_eval.py` — replaces `beir_eval.py` placeholders with a real SciFact evaluation. Needs a `--max-queries N` flag before next run so it finishes inside an hour on CPU.
- `paper-intel/CLAUDE.md` — persistent orientation doc.

### Background jobs left running (both should be killed on resume)

| PID | Started | Log | State |
|---|---|---|---|
| 1796 | 12:05:38 | `beir_real_eval.log` | BEIR SciFact encoding batch 73/648, ~3 hours remaining |
| 1836 | 12:10:02 | `eval_compare_100q.log` | Stalled downloading `bge-reranker-v2-m3` (2GB, `.incomplete` blob) |

Kill: `taskkill //PID 1796 //F && taskkill //PID 1836 //F` (Git Bash)

### LLM provider clarification

User referred to "NVIDIA NIM" but the actual key in `.env` is **OpenRouter** (`OPENROUTER_API_KEY=sk-or-v1-...`). `NIM_API_KEY` slot in `.env.example` is empty. All LLM calls in this session would go through OpenRouter. `graph_rag.py:_generate()` already has OpenRouter fallback wired in from Session 2.

### Agents dispatched this session

5 parallel in Wave 1 — 3 Sonnet investigators (bug check, test-set audit, config verification) + 1 Haiku formatting audit + 1 general-purpose citation lookup (blocked on web perms, Opus took over inline). All returned with findings above.

### Resume priority order

1. Kill the two stalled jobs.
2. Fix the reranker reference from `bge-reranker-v2-m3` → `cross-encoder/ms-marco-MiniLM-L-6-v2` throughout (faster, already cached).
3. Re-run 100-Q retrieval-only eval — this is Table II, the paper's primary result.
4. Re-run `beir_real_eval.py` with `--max-queries 100` for a 1-hour BEIR baseline.
5. Then Wave 3 architecture work (list in `CLAUDE.md`).
6. Then docx/LaTeX reconciliation and rewriting.

---

## Session 2 (2026-04-07)

### What was completed

**Phase 1 — Score improvements (all done)**
- `graph_rag.py`: τ changed 0.10 → 0.30, `max_graph_hops` 1 → 2, `expansion_mode` → "qbpr"
- `graph_rag.py`: Added lazy-load `_get_reranker()` (cross-encoder/ms-marco-MiniLM-L-6-v2, CPU)
- `graph_rag.py`: Added lazy-load `_get_st_model()` (BAAI/bge-m3, CPU)
- `graph_rag.py`: `get_embedding()` — sentence-transformers primary, Ollama fallback
- `graph_rag.py`: `_generate()` — Ollama primary, OpenRouter fallback (google/gemma-3-4b-it:free)
- `graph_rag.py`: `search()` fully rewritten → proper 3-channel AHR pipeline:
  - Channel 1: Dense FAISS, Channel 2: BM25, Channel 3: Graph (QBPR)
  - 3-channel Confidence-Weighted RRF fusion
  - Slot reservation (AHR): guarantees graph slots in top-k
  - Cross-encoder reranking at end of pipeline
  - Score reassignment after reranking (1/(60+rank)) so display is consistent
- `graph_rag.py`: Fixed `_best_chunk_from_paper()` display score bug (was raw BM25, now 1/61)
- `graph_rag.py`: Fixed `get_embedding()` `UnboundLocalError` (st_err variable scope)
- `graph_rag.py`: Set `OPENBLAS_NUM_THREADS=1` inside `_get_reranker()` before model load
- `core/graph.py`: τ changed 0.10 → 0.30
- `app.py`: Switched from OllamaRAG → GraphRAG
- `app.py`: Removed fake `generate_test_metrics()` function
- `app.py`: All endpoints return real `graph_stats` from `rag_system.get_graph_stats()`
- `app.py`: Reranker warm-up at startup
- `app.py`: Early env var setup (OPENBLAS_NUM_THREADS, PYTHONIOENCODING, TOKENIZERS_PARALLELISM)
  before any imports to prevent Windows threading deadlocks
- `.env`: Added NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENBLAS_NUM_THREADS, PYTHONIOENCODING
- `requirements.txt`: Added `neo4j>=5.0.0`

**Phase 2 — Ingestion fixes (all done)**
- `ingestion/discovery/pubmed_client.py`: Added `fetch_abstracts` flag + enrichment loop
- `ingestion/discovery/crossref_client.py`: Added `enrich_abstracts` flag + DOI enrichment loop

**Phase 3 — Neo4j backend (done)**
- `core/neo4j_graph.py`: New file — `Neo4jGraphIndex` mirroring `GraphIndex` interface
- `build_neo4j_graph.py`: New file — builds Neo4j graph from existing rag_index
- Neo4j Desktop installed, GDS + APOC plugins installed, graph built (36 nodes, 296 edges)
- NOTE: App still uses NetworkX graph (GraphRAG.graph: nx.Graph). Neo4j graph is built
  and visible in Neo4j Browser but not yet wired into the live search pipeline.

### Current system state (after reboot)

**Start command (PowerShell):**
```powershell
$env:OPENBLAS_NUM_THREADS="1"
cd D:\Sanshodhak\paper-intel
D:\Sanshodhak\.venv\Scripts\python.exe app.py
```

**On startup you will see:**
- OpenBLAS warning (harmless, ignore it)
- GraphRAG loaded: 3383 vectors, 3383 chunks from 36 papers
- Graph: 36 nodes, 22 edges (Jaccard=22 + Citation=6) at τ=0.30
- Cross-encoder reranker ready
- RESEARCH ASSISTANT READY at http://localhost:5000

**First query will be slow (~30-60s)** — BGE-M3 (570MB) loads on first embedding call.
Subsequent queries are fast (~0.5-1s for search, +20-30s for full pipeline with OpenRouter LLM).

**Confirmed working:**
- 3-channel AHR: Dense + BM25 + Graph → CW-RRF → cross-encoder rerank
- OpenRouter generation (google/gemma-3-4b-it:free)
- Neo4j running at bolt://127.0.0.1:7687

### What's left to do

1. **Verify score reassignment fix** — after reboot, run a search query and confirm
   scores show consistent values (~0.016 range) after cross-encoder reranking.
   If scores still look like raw RRF (0.033-0.038), the fix is working on display layer.

2. **Ingest more papers** — currently only 36 papers. The graph only has 22 edges.
   Target: 100-200 papers on RAG/hybrid retrieval topics.
   Steps:
   a. Run ingestion pipeline (OpenAlex, arXiv) with queries like "hybrid retrieval RAG"
   b. Re-embed with BGE-M3 and append to FAISS index
   c. Rebuild NetworkX knowledge graph (automatic on next app start)
   d. Rebuild Neo4j graph: `python build_neo4j_graph.py`

3. **Wire Neo4j into live pipeline** (optional) — currently the live search uses
   NetworkX (in-memory). To use Neo4j instead, replace `self.graph: nx.Graph` in
   GraphRAG with Neo4jGraphIndex calls.

4. **Commit all changes to git** — none of the session 2 changes have been committed.
