# Sanshodhak Paper — Module-wise 7-Page Draft

Target venue style: IEEE-like 2-column, ~7 pages (excluding references if allowed by venue).

---

## Page Budget (Recommended)

- Module 1: Title + Abstract + Keywords + Intro → **1.0 page**
- Module 2: Related Work → **0.75 page**
- Module 3: System Architecture → **1.0 page**
- Module 4: Methods → **1.0 page**
- Module 5: Experimental Setup → **0.75 page**
- Module 6: Results → **1.0 page**
- Module 7: Architecture-Code Fidelity Analysis → **0.75 page**
- Module 8: Discussion + Limitations + Ethics + Conclusion → **0.75 page**
- References/Tables/Figures → venue-dependent overflow buffer

---

## Module 1 — Front Matter + Introduction (Draft)

### Title
Sanshodhak: A Scholarly Research-Intelligence Platform with Multi-Source Ingestion, Retrieval-Augmented Answering, and a Hybrid Retrieval Roadmap

### Abstract (journal-ready)
Scholarly literature workflows remain fragmented across discovery, parsing, retrieval, and synthesis tools, creating high manual overhead for researchers. We present Sanshodhak, a production-oriented research-intelligence platform that unifies multi-source paper discovery, robust PDF acquisition, fallback-aware text extraction, dense vector indexing, retrieval-augmented answering, and implementation-resource recommendation. We also incorporate the PaperWeave specification as design context for an advanced hybrid retrieval target architecture. Using implementation-grounded repository artifacts, Sanshodhak achieves Precision@1 = 0.35, Recall@5 = 0.75, MRR = 0.5204, and NDCG@5 = 0.6258 on a 20-question retrieval run; a secondary semantic generation run reports BERTScore-F1 = 0.8620. Beyond performance reporting, we perform architecture-to-code cross-verification to explicitly separate deployed functionality from partially integrated and planned components, reducing claim inflation and improving scientific transparency. The study demonstrates that Sanshodhak already provides practical utility for literature-centric workflows while establishing a clear roadmap toward full hybrid and graph-augmented retrieval.

### Keywords
Retrieval-Augmented Generation, Scholarly Search, Research Intelligence, Hybrid Retrieval, FAISS, Knowledge Graph, Literature Synthesis

### Introduction (condensed)
The growth of scientific literature has made evidence-grounded synthesis a core bottleneck in research workflows. In practice, researchers must combine multiple disconnected tools: one for search, another for PDF parsing, another for semantic indexing, and another for narrative synthesis. This fragmentation slows iteration and increases risks of missing relevant evidence.

Retrieval-augmented generation (RAG) offers a promising direction by grounding generation in retrieved context. However, deployment-grade RAG for scholarly settings requires more than retrieval and generation alone: high-recall discovery, robust parsing under noisy PDFs, transparent source attribution, and reproducible evaluation are equally important.

Sanshodhak is designed as an integrated scholarly research-intelligence platform with six deployed stages: search, download, parse, embed/index, retrieve/answer, and recommend. In parallel, the PaperWeave product specification provides a target design for advanced hybrid retrieval and confidence-weighted synthesis. This paper intentionally separates these layers to maintain scientific rigor.

Our contributions are threefold. First, we present an implementation-grounded system architecture for end-to-end literature intelligence. Second, we report repository-native retrieval and generation metrics with explicit caveats. Third, we introduce architecture-code fidelity analysis as a reporting practice that distinguishes deployed capabilities from roadmap components.

The remainder of this paper covers related work, architecture and methods, experimental setup, results, fidelity analysis, and implications for future graph-augmented retrieval research.

---

## Module 2 — Related Work (Draft)

RAG systems combine non-parametric retrieval with parametric language generation and are now widely used in knowledge-intensive NLP settings. In scholarly workflows, retrieval quality strongly influences answer quality, making ranking metrics (Precision@k, Recall@k, MRR, NDCG) critical.

Dense retrieval with vector indexes provides strong semantic matching and practical scalability. FAISS has become a widely adopted backend for efficient nearest-neighbor search over embedding spaces. In parallel, lexical methods such as BM25 remain effective for technical terminology and exact phrase matching. Hybrid retrieval methods fuse these complementary signals, often improving robustness in domain-heavy corpora.

For generation evaluation, lexical overlap metrics (ROUGE, BLEU) remain common but can understate quality when outputs are paraphrastic. Semantic metrics such as BERTScore better capture meaning preservation under lexical variation.

Existing literature often reports architecture-level concepts or benchmark performance without fully documenting deployment fidelity. Our work differs by explicitly pairing system results with architecture-to-code verification, which is particularly important for production-facing RAG claims.

---

## Module 3 — System Architecture (Draft)

### 3.1 Deployed Runtime Pipeline
Sanshodhak currently operates as:
1. Multi-source search (OpenAlex, CORE, Semantic Scholar, CrossRef, Unpaywall, arXiv),
2. PDF download with legal-source-first fallback,
3. Parsing with GROBID-first strategy and fallback extraction,
4. Chunking + embedding (BGE-M3) + FAISS indexing,
5. Top-k retrieval and answer generation,
6. Resource recommendation (repositories/models/tutorials).

### 3.2 Key Components
- **Ingestion layer:** source orchestration, deduplication, ranking.
- **Processing layer:** robust parser stack with fallback.
- **Retrieval layer:** dense FAISS retrieval in default path.
- **Generation layer:** context-grounded answer synthesis.
- **Recommendation layer:** implementation support assets.
- **API/UI layer:** request handling, progress streaming, endpoint orchestration.

### 3.3 PaperWeave Design Context
PaperWeave specifies an expanded hybrid retrieval stack: BM25 + semantic retrieval + RRF fusion + reranking + graph expansion + confidence scoring + citation-density controls. We use this as architectural direction, while maintaining strict distinction from measured deployed behavior.

### 3.4 Deployed vs Planned
- **Implemented:** ingestion, fallback parsing, dense retrieval, answering, recommendation, web APIs.
- **Partially integrated:** hybrid and reranking in alternate runtime path.
- **Planned:** online graph retrieval context assembly and dedicated observability dashboard.

---

## Module 4 — Methods (Draft)

### 4.1 Acquisition and Parsing
Documents are collected from heterogeneous scholarly APIs, deduplicated, and downloaded using a staged fallback strategy. Parsing uses a primary structured extractor (GROBID) and fallbacks (PyMuPDF/pdfplumber/PyPDF2) to improve robustness under document variability.

### 4.2 Retrieval Formulation
In the deployed dense path, chunk relevance is computed with normalized inner product:

\[ s_i = q^\top d_i \]

where \(q\) is query embedding and \(d_i\) chunk embedding.

In the target hybrid design, rankings are fused by reciprocal-rank fusion:

\[ \mathrm{RRF}(i)=\sum_j\frac{1}{k+r_j(i)} \]

where \(r_j(i)\) is rank of item \(i\) under retriever \(j\).

### 4.3 Generation and Attribution
Retrieved chunks are assembled into grounded context for generation, and source metadata is returned for answer attribution at chunk level.

### 4.4 Confidence Modeling (Design-Level)
PaperWeave’s confidence design can be represented as:

\[ C = w_eE + w_rR + w_gG + w_aA - w_uU \]

where evidence quantity \(E\), relevance \(R\), graph support \(G\), agreement \(A\), and uncertainty \(U\) are weighted factors. Calibration is future work.

### 4.5 Runtime Behavior
The platform prioritizes practical throughput and robustness over purely benchmark-optimized settings, targeting stable operation for literature review workflows.

---

## Module 5 — Experimental Setup (Draft)

### 5.1 Configuration
- Embeddings: BGE-M3 (1024 dimensions)
- Vector index: FAISS IndexFlatIP (~14 MB)
- Corpus: 36 papers, 3,383 chunks
- Typical retrieval latency: 2–4 s
- Typical generation latency: 5–8 s

### 5.2 Evaluation Runs
- **Run A:** 20-question retrieval-focused evaluation.
- **Run B:** 10-question generation-focused evaluation with semantic metrics.

### 5.3 Metrics
Retrieval: Precision@k, Recall@k, NDCG@k, MRR.  
Generation: ROUGE-1/2/L, BLEU, BERTScore-P/R/F1.

### 5.4 Reproducibility
All values are drawn from repository-native metric files and supporting scripts; no synthetic post-hoc metrics are introduced.

---

## Module 6 — Results (Draft)

### 6.1 Retrieval Results (Primary Run)
- P@1 = 0.35
- P@5 = 0.16
- R@5 = 0.75
- MRR = 0.5204
- NDCG@5 = 0.6258

These results indicate moderate precision but strong evidence coverage, suitable for retrieval-assisted synthesis where recall and ranking depth matter.

### 6.2 Generation Results (Secondary Run)
- ROUGE-1 = 0.3067
- ROUGE-2 = 0.0807
- ROUGE-L = 0.2135
- BLEU = 0.0236
- BERTScore-F1 = 0.8620

The lexical-semantic gap suggests paraphrastic generation: low n-gram overlap with preserved semantic alignment.

### 6.3 Latency Perspective
Observed latency is consistent with interactive research-assistant usage in local/developer settings.

### 6.4 Caveats
Secondary run recall values above 1.0 indicate metric-definition inconsistency or implementation issue and should not be used as headline recall claims.

---

## Module 7 — Architecture-Code Fidelity Analysis (Draft)

### 7.1 Verification Strategy
Architecture blocks were validated against runtime code paths and evaluation artifacts.

### 7.2 Layer-wise Findings
- Implemented: ingestion, parsing fallback, dense retrieval, recommendation, API serving.
- Partial: hybrid/rerank available but not default path.
- Planned: graph retrieval orchestration and advanced observability.

### 7.3 Reporting Implications
This analysis prevents claim drift by ensuring that narrative architecture and measured deployment state are explicitly aligned.

---

## Module 8 — Discussion, Limitations, Ethics, Conclusion (Draft)

### 8.1 Discussion
Sanshodhak already provides practical literature intelligence through high-coverage retrieval and grounded answer generation. PaperWeave contributes a strong blueprint for hybrid and graph-aware enhancements.

### 8.2 Limitations
- Mixed evaluation setups reduce direct comparability.
- Default runtime path is dense-first, not full hybrid.
- Graph retrieval is artifact-ready but not yet deployed online.

### 8.3 Ethics and Responsible Use
Legal-source-first retrieval, provenance visibility, and uncertainty-aware presentation are essential for responsible scholarly deployment.

### 8.4 Conclusion
Sanshodhak demonstrates a credible production baseline for research-intelligence systems, and with PaperWeave-informed roadmap alignment, it offers a clear trajectory toward hybrid, graph-augmented, confidence-calibrated scholarly assistance.

---

## Next Writing Workflow
1. Paste Module 1 into `main.tex` and finalize wording.
2. Add Module 2–3 with one architecture figure/table.
3. Add Module 4–6 with two result tables.
4. Add Module 7–8 and tighten to 7-page limit.
5. Final pass: references, formatting, and claim-fidelity audit.
