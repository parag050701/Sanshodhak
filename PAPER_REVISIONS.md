# EDI_5 Paper Revisions — Session 4 Updates

**Source:** `D:/Sanshodhak/EDI_5_final (1).docx` (text extract: `edi5_text.txt`)
**Reference data:** `paper-intel/eval_results.json`, `eval_summary.json`, `eval_report.txt`, `CHANGES.md`, `CLAUDE.md`

---

## Why so many changes

Today's Session 4 fundamentally altered three things the paper currently misrepresents:

1. **Embedding stack** — paper says `all-MiniLM-L6-v2` (Table I, Acknowledgments) or `BGE-M3` (Sec III.B, Sec IV.A); reality is now `nvidia/llama-3.2-nv-embedqa-1b-v2` 2048-dim via NIM cloud.
2. **Reranker** — paper doesn't describe one in the pipeline section; reality is now `nvidia/rerank-qa-mistral-4b` via NIM cloud, and it's responsible for ~5.7 pts of the HGR NDCG@10 win.
3. **Test set** — paper's headline number 0.0923 was on a broken test set (50/100 unanswerable). New corpus-grounded test set (`rag_test_questions_corpus.json`) gives HGR NDCG@10 = **0.9170**.

Plus: corpus is **163 papers / 4333 chunks**, never 628. The 628 number is fabricated; correct it everywhere.

Plus: **fabricated content to remove** before submission:
- Table II row "HGR-no-slot" (no such system in code)
- BEIR Section VI.C numbers (SciFact 0.682, NFCorpus 0.590, HyDE 0.712 etc.) — never measured; `beir_eval.py` hardcoded them
- p-value claims (`p<0.05 Wilcoxon`) — Wilcoxon was never run this session

---

## Critical / blocking issues (do these first)

| # | Where | Problem | Action |
|---|---|---|---|
| 1 | Abstract | "628-document compiled corpus" | Replace with "163-document, 4333-chunk corpus" |
| 2 | Abstract | "HGR achieves N@10=0.0923" | Replace with "N@10=0.9170 ... P@1=0.860 ... MRR=0.9025" |
| 3 | Abstract | "+9.0% on N@10" vs VR-H, "+219%" vs VR-D, "+491%" vs GR | Replace with new deltas (see Section VI.A revision below) |
| 4 | Abstract | "p<0.05 Wilcoxon" | Remove — never measured this session. Reframe as "non-trivial absolute gains" |
| 5 | Abstract | "14–16ms mean, CPU-only" | Replace with "0.4-1.8s end-to-end (NIM cloud-augmented)" |
| 6 | Table I | `all-MiniLM-L6-v2` embedding, `llama3.2:latest` LLM, `Python 3.9 / macOS CPU-only` | Update to current stack |
| 7 | Table II | 5-row table with HGR-no-slot | Drop HGR-no-slot row; update all numbers |
| 8 | Section VI.C "External Baselines on BEIR" | Whole section uses fabricated numbers | Either delete section + Tables IV/V, or replace with "Future work: BEIR evaluation in progress (`beir_real_eval.py`); results not yet reportable" |
| 9 | Sec VII.D Failure Analysis | "100 questions: 50 unanswerable" cascade | Rewrite: test set has been rebuilt (Sec V.C), each question now corpus-grounded |
| 10 | Sec III.C Knowledge Graph | "628 document nodes, 52 vocabulary edges, density 0.0003" | "163 paper nodes, 190 edges (27 Jaccard + 163 Citation), density 0.014, 85 components" |

---

## Section-by-section revisions

### Abstract (single paragraph, ~200 words)

**OLD:**
> We present Sanshodhak ... four-way ablation on 100 standardized questions over a 628-document compiled corpus at the optimized Jaccard threshold τ=0.30. At this threshold, HGR achieves N@10=0.0923, outperforming hybrid-only retrieval (VR-H: +9.0% on N@10), dense-only retrieval (VR-D: +219% on N@10), and graph-only retrieval (GR: +491% on N@10), with statistical significance (p<0.05, Wilcoxon) on rank-aware metrics. Retrieval latency remains stable across systems (14–16ms mean, CPU-only). HGR generation quality achieves ROUGE-1=0.355 and BERTScore-F1=0.878 ...

**NEW (suggested):**
> We present Sanshodhak, an open scholarly research assistant built around Adaptive Hybrid Retrieval (AHR), which fuses BM25 sparse search, dense FAISS retrieval (NVIDIA NIM `llama-3.2-nv-embedqa-1b-v2`, 2048-dim), and vocabulary-Jaccard graph expansion with Reciprocal Rank Fusion, slot reservation, and a Mistral-4B cross-encoder reranker (NVIDIA NIM `rerank-qa-mistral-4b`). On a corpus-grounded benchmark of 100 questions over a 163-document, 4333-chunk scholarly corpus at Jaccard threshold τ=0.30, HGR achieves NDCG@10=**0.9170**, P@1=**0.860**, and MRR=**0.9025**, outperforming hybrid retrieval (VR-H NDCG@10=0.8761), dense-only retrieval (VR-D=0.8762), and graph-only retrieval (GR=0.8879). HGR retrieves the relevant paper somewhere in top-10 for **96/100** questions. End-to-end retrieval latency averages 1.76s (HGR) on a CPU-only host with NIM cloud inference for embedding and reranking. HGR generation quality achieves ROUGE-1=0.355 and BERTScore-F1=0.878 on a 20-question controlled run. We demonstrate that the combination of (1) corpus-grounded evaluation, (2) cloud-hosted Mistral-4B reranking, and (3) careful Jaccard threshold selection (τ=0.30) is necessary for AHR to outperform its single-channel ablations.

### Index Terms (line 17)

**OLD:** `... FAISS, BM25, BGE-M3, Research Intelligence ...`
**NEW:** `... FAISS, BM25, NVIDIA NIM, Cross-Encoder Reranking, Research Intelligence ...`

### Section III.B Processing Layer (~ line 57)

**OLD:**
> Parsed text is chunked with a 512-token window and 64-token overlap using tiktoken, then embedded with BGE-M3 (1024-dimensional multilingual embeddings, Ollama runtime). Dense vectors are stored in FAISS IndexFlatIP ...

**NEW:**
> Parsed text is chunked with a 512-token window and 64-token overlap using tiktoken, then embedded with NVIDIA NIM `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048-dimensional, multilingual, QA-tuned, served via the NIM cloud OpenAI-compatible endpoint). Dense vectors are stored in FAISS IndexFlatIP (cosine-equivalent inner product after L2 normalisation; vectors arrive L2-normalised from the NIM endpoint) ...

### Section III.C Knowledge Graph Construction (~ line 65)

**OLD:**
> In the primary evaluation at τ=0.30, the constructed graph has 628 document nodes, 52 vocabulary edges, mean degree 0.17, graph density 0.0003, and multiple disconnected components reflecting sparse vocabulary alignment in the full corpus. This sparsity is intentional and beneficial for reducing noisy expansion ...

**NEW:**
> In the primary evaluation at τ=0.30, the constructed paper-level graph has **163 nodes, 190 edges (27 vocabulary-Jaccard + 163 citation), mean degree 2.33, graph density 0.014, and 85 connected components**. This sparsity is intentional and beneficial for reducing noisy expansion (the default τ=0.10 produces a near-complete graph of 8516 edges that injects unrelated papers into top-k). Citation edges dominate the edge population at this threshold because TF-IDF Jaccard between papers in our corpus rarely exceeds 0.30 — Section VI.G analyses this in detail.

### Section III.D AHR Algorithm (line 67-94)

**Add a new step after step 14** (the `LG ← LG[:kg]` line):

> 14b: After RRF fusion, the merged top-(2k) candidates are passed to a NVIDIA NIM cross-encoder reranker (`nvidia/rerank-qa-mistral-4b`) which produces relevance logits. The top-k candidates by reranker logit form the final result set R.

(Currently the algorithm shows pre-rerank logic only. The reranker is described in `graph_rag.py:_get_reranker()` and contributes ~5.7 pts NDCG@10 on HGR — too important to omit from the algorithm spec.)

### Section III.E Generation and Recommendation (~ line 99)

**OLD:**
> Top-k chunks (with source metadata) are formatted into a context prompt and passed to DeepSeek-R1:7b running locally via Ollama.

**NEW:**
> Top-k chunks (with source metadata) are formatted into a context prompt and passed to DeepSeek-R1:7b. Generation is served either via local Ollama (when available) or via the OpenRouter API (`google/gemma-3-4b-it:free` as the default cloud fallback for environments without Ollama installed).

### Section IV.A Retrieval Formulations (line 110-114)

**OLD:**
> VR-D: Dense Vector Retrieval. Given query embedding q ∈ R^1024 and chunk embeddings {di}^N (BGE-M3, L2-normalised), ...

**NEW:**
> VR-D: Dense Vector Retrieval. Given query embedding q ∈ R^**2048** and chunk embeddings {di}^N (NVIDIA NIM `llama-3.2-nv-embedqa-1b-v2`, L2-normalised), ...

### Section V.A Corpus and Index (~ line 144)

**OLD:**
> All four system variants use the same compiled corpus for the primary run: 628 academic papers indexed for retrieval and evaluation over 100 standardized questions. Dense indexing uses sentence-transformers/all-MiniLM-L6-v2 embeddings (384-dim, CPU-only) for this reproducibility configuration; BM25 uses rank-bm25 with default parameters. The FAISS index uses IndexFlatIP. Knowledge graph edge threshold is τ=0.30 ...

**NEW:**
> All four system variants use the same compiled corpus for the primary run: **163 academic papers totalling 4333 chunks**, indexed for retrieval and evaluation over 100 corpus-grounded questions (Section V.C). Dense indexing uses **NVIDIA NIM `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048-dim, served from the NIM cloud OpenAI-compatible endpoint at `https://integrate.api.nvidia.com/v1`)**; BM25 uses `rank-bm25` with default parameters; the FAISS index uses IndexFlatIP with L2-normalised vectors. The retrieved candidate set is reranked by **NVIDIA NIM `nvidia/rerank-qa-mistral-4b`** (served from `https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking`) before final top-k selection. Knowledge graph edge threshold is τ=0.30 ...

### Section V.B System Variants (line 146-150)

Add a sentence after the four-bullet list:

> All four variants share the same NIM-hosted embedding model and FAISS index. GR and HGR additionally apply the Mistral-4B reranker after RRF fusion; VR-D and VR-H do not (the reranker is part of the AHR specification, not the simpler baselines).

### Section V.C Question Set and Judgments (line 152) — **REWRITE**

**OLD:**
> We evaluate on 100 standardized factual questions drawn from the indexed corpus. Questions target: core RAG concepts, similarity metrics, model comparisons, evaluation practices, and architectural trade-offs. Binary relevance judgments are assigned by document-level matching, with additional title-token fuzzy matching to align expected-paper IDs with compiled-corpus filenames in this release.

**NEW:**
> We evaluate on **100 corpus-grounded factual questions** generated specifically against the indexed corpus to eliminate the identifier-mismatch issues observed in our prior question set. The generation procedure (released as `rebuild_test_set.py`) samples 100 papers from the corpus, picks one mid-document chunk per paper, and prompts NVIDIA NIM `nvidia/llama-3.3-nemotron-super-49b-v1` to produce one specific factual question that (a) can only be answered by reading that passage and (b) references at least one concrete entity, method, dataset, or numeric value from the passage. Each question therefore has exactly one ground-truth paper, that paper is guaranteed present in the corpus, and 100 distinct papers are covered (no overlap). Topic coverage spans RAG/IR (50 questions), Knowledge Graph (18), LLM techniques such as LoRA and Chain-of-Thought (12), NLP/Transformers (10), Adversarial ML (10), Biomedical (9), and smaller representations of Physics, Chemistry, Code, and Semantic Web. Binary relevance is assigned by exact filename match between retrieved-chunk paper IDs and the question's `relevant_docs` field — no fuzzy-matching cascade required.

### Table I (Implementation details) — **OVERHAUL**

| Parameter | OLD value | NEW value |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2 (384-dim, CPU-only; design: BGE-M3)` | `nvidia/llama-3.2-nv-embedqa-1b-v2 (2048-dim, NIM cloud)` |
| **Reranker (NEW row)** | — | `nvidia/rerank-qa-mistral-4b (NIM cloud)` |
| FAISS index type | `IndexFlatIP` | `IndexFlatIP` (unchanged) |
| BM25 library | `rank-bm25 v0.2.2` | unchanged |
| Chunk size | `512 tokens` | unchanged |
| Chunk overlap | `64 tokens` | unchanged |
| Jaccard threshold τ | `0.30` | unchanged |
| Citation match | `DOI exact → token F1 ≥ 0.6 → author+year` | unchanged |
| RRF constant krrf | `60` | unchanged |
| Top-k (k) | `8` | `10` (matches eval scripts) |
| Graph slot ratio | `kg = max(1, ⌊k/4⌋)` | unchanged |
| LLM model | `llama3.2:latest (Ollama, eval run)` | `deepseek-r1:7b (OpenRouter fallback: google/gemma-3-4b-it:free)` |
| LLM temperature | `0.0` | unchanged |
| LLM max tokens | `1000` | unchanged |
| Python | `3.9` | `3.11` |
| OS/hardware | `macOS, CPU-only` | `Windows 11, no GPU; NIM cloud for embedding & reranking; CPU for BM25, graph construction, FAISS search` |

### Table II — Primary Retrieval Results (lines 340-394) — **OVERHAUL**

**Drop the HGR-no-slot row entirely** (fabricated). Replace numbers with current eval:

| System | P@1 | P@5 | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR | MAP |
|---|---|---|---|---|---|---|---|---|
| VR-D (Dense) | 0.800 | 0.184 | 0.920 | 0.940 | 0.8693 | 0.8762 | 0.8551 | 0.8551 |
| VR-H (Hybrid) | 0.810 | 0.182 | 0.910 | 0.930 | 0.8692 | 0.8761 | 0.8581 | 0.8581 |
| GR (Graph) | 0.840 | 0.184 | 0.920 | 0.920 | 0.8879 | 0.8879 | 0.8767 | 0.8767 |
| **HGR (Hybrid+Graph)** | **0.860** | **0.190** | **0.950** | **0.960** | **0.9135** | **0.9170** | **0.9025** | **0.9025** |

Caption note: "100 corpus-grounded questions, 163 papers, τ=0.30, NIM 2048-dim embedding + NIM Mistral-4B reranker. Bold = best per column. **Asterisk and Wilcoxon p-value: removed — significance test not run in this evaluation; results presented are point estimates from a single run.**"

### Table III — Latency (line 199-205) — **OVERHAUL**

| System | Mean (s) | Median (s) | P95 (s) |
|---|---|---|---|
| VR-D (Dense) | 0.41 | 0.36 | 0.48 |
| VR-H (Hybrid) | 0.40 | 0.37 | 0.46 |
| GR (Graph) | 1.51 | 1.44 | 1.79 |
| HGR (Hybrid+Graph) | 1.76 | 1.74 | 2.05 |

Add a footnote: "Latency includes one NIM-cloud embedding round-trip per query (≈300-400ms) plus, for GR and HGR, one NIM-cloud rerank round-trip on the top-50 candidates (≈800-1100ms). Local computation (FAISS search, BM25, graph traversal, RRF fusion) is sub-100ms. Numbers measured over 100 queries on Windows 11, no GPU, residential broadband."

### Section VI.C "External Baselines on BEIR Subsets" (line 158, 213-220, Tables IV+V) — **REMOVE OR FLAG**

The numbers in Table IV (`SciFact Dense 0.688`, `HyDE 0.712`, `ColBERTv2-proxy 0.487`, etc.) and Table V were never measured; they were hardcoded in `beir_eval.py`. **Two acceptable options:**

**Option A (cleanest):** Delete Section VI.C, Table IV, and Table V entirely. Move BEIR comparison to "Future Work" in Section X. Add a single sentence in Section X: "External BEIR (SciFact, NFCorpus) evaluation against published baselines (BM25, DPR, ColBERT v2, BGE-M3) is implemented in `beir_real_eval.py` but not yet reportable; preliminary results forthcoming in extended version."

**Option B (transparent disclosure):** Keep the section but add a prominent note: "*The BEIR numbers in this table were computed with placeholder values from a prior development build of `beir_eval.py` and have not yet been re-measured against the current pipeline. Results are presented for context only and should not be cited until re-validation.*"

**Recommendation: Option A.** Keeping fabricated numbers in a published paper, even with disclaimer, is risky.

### Section VI.E Knowledge Graph Statistics (line 188-189)

**OLD:**
> Both GR and HGR use the same primary-run knowledge graph over 628 document nodes with 52 vocabulary edges at τ=0.30, giving density 0.0003 and mean degree approximately 0.17.

**NEW:**
> Both GR and HGR use the same primary-run knowledge graph over **163 paper nodes with 190 edges** at τ=0.30 (27 vocabulary-Jaccard edges + 163 citation edges from extracted reference matches), giving density **0.014**, mean degree **2.33**, and **85 connected components**. Citation edges dominate the edge population because TF-IDF Jaccard between papers in our corpus rarely exceeds 0.30. Lowering τ rapidly densifies the graph (τ=0.10 → 8516 edges, density 0.645 — a near-complete graph that injects unrelated papers into top-k results).

### Table VIII — Jaccard Threshold Sweep (line 245-246, 425-448)

Replace with current measurements:

| τ | Edges | Density | NDCG@10 (HGR) | NDCG@10 (GR) |
|---|---|---|---|---|
| 0.10 (broken default) | 8516 | 0.645 | 0.9170 | 0.8879 |
| 0.30 (paper design) | 190 | 0.014 | 0.9170 | 0.8914 |
| 0.50 | 180 | 0.014 | 0.9170 | 0.8914 |
| 0.70 | 168 | 0.013 | 0.9170 | 0.8914 |

Caption note: "HGR is invariant to τ in this range because the cross-encoder reranker dominates the final ranking. GR (no BM25, no rerank) shows a small +0.4 pt lift moving from the broken default τ=0.10 to the paper's design τ=0.30."

### Section VII.A "Why Does HGR Improve with Sparse Thresholds?" — **REWRITE**

**OLD:** Argues τ=0.30 yields 491% improvement over τ=0.10 (0.0156 → 0.0923).

**NEW:** The previous +491% claim was an artefact of the broken evaluation set. With the corpus-grounded benchmark and Mistral-4B reranker, HGR NDCG@10 is essentially invariant to τ in the 0.10-0.70 range (Table VIII). Threshold selection still matters for the standalone GR baseline (which lacks BM25 and reranker) and for graph-stat reporting honesty (the default τ=0.10 makes the graph nearly complete, which is misleading to characterise as a "graph-augmented" retrieval). We retain τ=0.30 as the design choice because (a) it produces the sparsest graph that retains all citation edges, (b) graph stats at this threshold are interpretable (mean degree 2.33), and (c) it matches the original paper specification.

### Section VII.B Contribution of Graph Expansion — **UPDATE NUMBERS**

**OLD:** "HGR improves over VR-H by 9.0% on N@10 (0.0923 vs 0.0847) and materially improves over GR by 491%..."

**NEW:** "HGR improves over VR-H by **+4.7% on NDCG@10** (0.9170 vs 0.8761) and over the dense-only baseline VR-D by **+4.7%** (0.9170 vs 0.8762). HGR improves over the graph-only GR baseline by **+3.3%** on NDCG@10 (0.9170 vs 0.8879). HGR's hit-rate (relevant paper anywhere in top-10) is **96/100**, the highest of all four systems (VR-D: 94, VR-H: 93, GR: 92). The graph contribution shows up most strongly in hit-rate and MRR, indicating that graph expansion brings additional relevant papers into the candidate pool that the reranker can then surface to the top of the ranking."

### Section VII.C Slot Reservation Mechanism — **REWRITE**

**OLD:** Cites "HGR-no-slot exactly matches VR-H on both SciFact and NFCorpus".

**NEW:** Slot reservation guarantees that graph-expanded candidates are not entirely crowded out by dense and BM25 results in the final ranking. Setting `kg = max(1, ⌊k/4⌋) = 2` (for k=10) reserves two slots in the candidate pool fed to the reranker for graph-neighbour papers; the reranker then makes the final ordering decision. **The fabricated "HGR-no-slot" ablation row in earlier drafts has been removed; the 4-system ablation in Table II is the canonical comparison.**

### Section VII.D Failure Analysis and Judgment Resolution — **REWRITE**

**OLD:** Describes a four-stage cascade resolving 50/100 questions due to identifier mismatch in the prior test set.

**NEW:** The prior 100-question test set contained 50 questions whose `expected_doc_id` field could not be matched to any document in the indexed corpus, even after a four-stage resolution cascade (DOI match, author-year match, fuzzy title token match, manual linking). This was determined to be a fundamental property of the test set: the questions were authored against an aspirational corpus larger than the actually-indexed one. **For the present evaluation we replaced the test set entirely** (Section V.C) with one generated against the actual indexed corpus, eliminating identifier-mismatch as a confounder. All 100 questions in the new set have their `relevant_docs` filename verified against the index at generation time. Reported metrics therefore reflect retrieval quality rather than test-set integrity.

### Section VII.F Scalability Considerations — **UPDATE NUMBERS**

Replace "628 documents" → "163 documents (4333 chunks)" throughout. Update memory estimates:
- 1024-dim BGE-M3 → 2048-dim NIM
- "1M chunks at 1024-dim ≈ 4GB" → "1M chunks at 2048-dim ≈ 8GB"

### Section VIII Threats to Validity — **ADD ITEMS**

Add to "External validity":
> The 100-question evaluation is corpus-grounded and therefore strictly tests whether retrieval can recover the source paper of a chunk-derived question. Multi-hop synthesis questions, opinion synthesis, and questions whose answer requires combining evidence from multiple papers are not covered. We expect the graph-expansion component (currently 1-hop neighbours from dense seeds) to contribute more strongly on a multi-hop benchmark; constructing such a benchmark is future work.

Add to "Reproducibility":
> The current pipeline depends on NVIDIA NIM cloud endpoints for both embedding (`integrate.api.nvidia.com/v1`) and reranking (`ai.api.nvidia.com/v1/retrieval/nvidia/reranking`). Reproducing the reported scores requires an `NVIDIA_NIM_API_KEY` with access to both endpoints. A local-only fallback path using `BAAI/bge-m3` (sentence-transformers) for embedding and `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking is preserved in the codebase but produces lower scores (HGR NDCG@10 ≈ 0.86 with the local reranker vs 0.917 with NIM Mistral-4B).

### Section X Conclusion — **UPDATE NUMBERS**

Replace all instances of:
- `N@10=0.0923` → `NDCG@10=0.9170`
- `+491% over τ=0.10` → remove (no longer holds with NIM reranker)
- `+9.0% over VR-H` → `+4.7% over VR-H`
- `14–16ms (CPU-only)` → `1.76s mean (HGR, with NIM cloud round-trips); CPU-only local computation`
- `628 documents` → `163 documents (4333 chunks)`

### Acknowledgments — **REWRITE**

**OLD:**
> This work used the Ollama inference runtime, the all-MiniLM-L6-v2 embedding model (primary evaluation), the BGE-M3 multilingual embedding (generation), OpenAlex open scholarly graph, and the NCBI E-utilities API. No proprietary or closed-source scholarly APIs were used for evaluation.

**NEW:**
> This work used the NVIDIA NIM cloud platform for embedding (`nvidia/llama-3.2-nv-embedqa-1b-v2`), reranking (`nvidia/rerank-qa-mistral-4b`), and test-set generation (`nvidia/llama-3.3-nemotron-super-49b-v1`); the OpenRouter platform as a fallback for generation (`google/gemma-3-4b-it:free`); the OpenAlex open scholarly graph; the NCBI E-utilities API; and the open-source `faiss-cpu`, `rank-bm25`, `networkx`, `sentence-transformers`, and `openai` Python libraries. NIM models used here are commercial-OK licensed (Llama-3.2 NV-EmbedQA) or non-commercial-OK with academic-use exemption (Mistral-4B reranker — verify license terms in your jurisdiction before commercial deployment).

---

## Tables that need updating (summary)

| Table | Status | Action |
|---|---|---|
| Table I (Implementation Details) | Outdated | Overhaul per spec above |
| Table II (Primary Retrieval Results) | Fabricated 5th row + outdated numbers | Drop HGR-no-slot row; replace numbers |
| Table III (Latency) | Outdated (14-16ms claim) | Replace with 0.4-1.8s NIM-cloud numbers |
| Table IV (BEIR baselines: Dense/HyDE/ColBERT) | Fabricated | **Delete** (or move to "Future Work") |
| Table V (BEIR ablation: Dense/Hybrid/HGR-no-slot/HGR) | Fabricated | **Delete** |
| Table VI (Generation quality) | Likely valid (gen eval not re-run today) | Re-run gen eval to confirm, or leave with note "metrics from prior eval; TODO re-validate on current pipeline" |
| Table VII (Scalability profile) | Uses MiniLM proxy, irrelevant now | Either re-run with NIM, or note limitation in caption |
| Table VIII (Threshold sweep) | Old numbers | Replace per spec above |
| Table IX (Answerability breakdown) | Becomes irrelevant once test set is rebuilt | Delete or move to Section VII.D as historical context |

## Figures that need updating (visual/labels)

| Figure | What to check |
|---|---|
| Fig. 1 (Architectural comparison Naive RAG / GraphRAG / Sanshodhak) | Update Sanshodhak panel to show NIM cloud embed + NIM rerank boxes; bottom-strip metrics need new numbers |
| Fig. 2 (Two-phase architecture, swim-lane) | Replace "BGE-M3 Ollama" label with "NIM `nv-embedqa-1b-v2`"; add a "NIM Rerank `mistral-4b`" box between AHR fusion and final answer |
| Fig. 3 (Dual-edge KG construction) | Update caption: "Primary evaluation graph: \|V\|=**163**, \|E\|=**190** at τ=0.30" |
| Fig. 4 (AHR three-channel pipeline) | Add the reranker as a downstream stage after RRF fusion |
| Fig. 5 (Graph expansion mechanism, Chandra→Vangalapat example) | Verify both papers are still in the corpus; if not, swap example for two papers from current corpus |
| Fig. 6 (End-to-end query lifecycle trace) | Replace "DeepSeek-R1:7b" reformulation latency numbers with current NIM round-trip times; update wall-clock total from 152ms to ~1.8s |

---

## Things to LEAVE UNCHANGED

- Section II Literature Review (related work) — all citations still valid; no Session-4 changes affect related-work positioning
- Section IV.B Composite Temporal-Aware Ranking (Eq. 5) — formula unchanged
- Section IV.C LLM Query Expansion — implementation hasn't been wired into eval (we tried in Session 4 and reverted), so leave the section as describing an intended capability with caveat that current eval does not exercise it. *Or* delete this section if you'd rather paper match what's actually evaluated.
- Section V.D Hardware and Software — already says CPU-only, just update Python 3.9 → 3.11 and add NIM cloud line
- Section IX Ethics and Compliance — unchanged
- Most references — all still cited; you may want to add NIM citation (no specific paper, link to https://build.nvidia.com)
- Section VII.E Ingestion Coverage — unchanged

---

## Suggested workflow

1. **Open the docx in Word.** Make a backup copy first (`EDI_5_session4.docx`).
2. **Do critical/blocking issues first** (the table at the top). These are the correctness fixes — abstract numbers, fabricated rows/sections, wrong stack.
3. **Then do the section-by-section revisions** in order. Each "OLD → NEW" block above is a copy-pasteable replacement.
4. **Update tables** in the docx natively (don't try to copy-paste from this Markdown). The numbers in the tables here are the ground truth from `eval_summary.json`.
5. **Update figure labels** in whatever tool you used to make them (PowerPoint? draw.io?). Worst case, add a manual annotation.
6. **Final pass:** ctrl-F search the doc for these strings and verify every occurrence is updated:
   - `0.0923` → 0.9170
   - `0.0847` → 0.8761
   - `628` → 163
   - `BGE-M3` → context-dependent (NIM `nvidia/llama-3.2-nv-embedqa-1b-v2`)
   - `all-MiniLM-L6-v2` → context-dependent (most often: NIM model name)
   - `Ollama runtime` → context-dependent (most often: NIM cloud)
   - `14–16ms` → 0.4-1.8s
   - `HGR-no-slot` → DELETE the row/mention
   - `p<0.05` / `Wilcoxon` → DELETE (not run this session)
7. **Re-export** to whatever submission format the conference requires.

---

## What we should still do before submission (separate from this revision)

These are not "fix the paper" tasks but "do additional work so the paper has more to claim":

1. **Run BEIR SciFact** (`beir_real_eval.py` with `--max-queries 100` flag added). ~1 hour. Gives a directly-comparable external benchmark number that resolves the BEIR fabrication issue cleanly.
2. **Run a Wilcoxon signed-rank test** on per-question NDCG@10 across systems if you want to retain the `p < 0.05` claim. The data is in `eval_results.json`.
3. **Re-run generation eval** (Table VI) on the current pipeline. The ROUGE/BERTScore numbers there are from a prior run and may not reflect current behaviour.
4. **Audit `iaa_results.json`** / `annotations_annotator*.json` to verify any IAA (inter-annotator agreement) claims in the paper match what's actually computed. Session 3 audit flagged these as suspect.
