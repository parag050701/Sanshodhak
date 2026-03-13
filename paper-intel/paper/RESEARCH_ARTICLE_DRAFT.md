# Sanshodhak: A Scholarly Research-Intelligence Platform with Multi-Source Ingestion, Retrieval-Augmented Answering, and Design-Driven Hybrid Retrieval Roadmap

## Abstract
Scholarly research workflows increasingly require integrated systems that can discover literature, parse full-text papers, retrieve relevant evidence, synthesize cross-paper findings, and support grounded question answering. We present Sanshodhak, a production-oriented research-intelligence platform that combines multi-source paper discovery, robust PDF extraction, dense vector indexing, retrieval-augmented generation (RAG), and implementation-resource recommendation. In addition to reporting implemented system behavior, we incorporate the PaperWeave specification as a design-level context describing a target hybrid retrieval architecture (BM25 + semantic retrieval + reciprocal rank fusion + reranking + graph expansion). Implementation-grounded evaluation artifacts show that Sanshodhak achieves Precision@1 = 0.35, Recall@5 = 0.75, MRR = 0.5204, and NDCG@5 = 0.6258 on a 20-question retrieval run. A secondary semantic-generation run reports BERTScore-F1 = 0.8620. We further provide an architecture-to-code fidelity analysis to distinguish deployed functionality from planned components, improving scientific transparency and reducing claim inflation. The results indicate that Sanshodhak is already useful for practical literature analysis while offering a clear roadmap toward full hybrid and graph-augmented retrieval.

## 1. Introduction
The modern literature review process is bottlenecked by fragmented tooling: one system for discovery, another for parsing, another for search, and yet another for synthesis. Researchers often spend substantial time integrating heterogeneous outputs rather than reasoning about scientific content. Retrieval-augmented generation (RAG) addresses part of this challenge by grounding generation in retrieved evidence, but production use still depends on ingestion quality, retrieval robustness, and provenance fidelity.

Sanshodhak addresses this by implementing an end-to-end research-intelligence pipeline. The deployed system supports: (i) multi-source scholarly search, (ii) PDF download with fallback strategy, (iii) parsing with GROBID-first and extractor fallbacks, (iv) embedding and FAISS indexing, (v) question answering over retrieved chunks, and (vi) recommendation of implementation resources (repositories, models, tutorials). Alongside deployed behavior, we incorporate PaperWeave’s specification as a structured design target for advanced retrieval and confidence modeling.

This paper has two goals: report measurable current performance and document a scientifically honest path to the target architecture.

## 2. System Architecture
### 2.1 Deployed Sanshodhak Pipeline
The deployed runtime follows six stages:
1. Search papers from multiple APIs (OpenAlex, CORE, Semantic Scholar, CrossRef, Unpaywall, arXiv).
2. Download PDFs with legal-first fallback order.
3. Parse documents using GROBID when available; otherwise fallback extractors (PyMuPDF, pdfplumber, PyPDF2).
4. Chunk text, generate BGE-M3 embeddings, and build FAISS IndexFlatIP.
5. Retrieve top-k chunks and generate answers with source-linked evidence.
6. Recommend practical resources (code, models, tutorials) from external ecosystems.

### 2.2 PaperWeave Design Context (Target Behavior)
The PaperWeave document specifies an expanded retrieval and synthesis design:
- Hybrid retrieval combining BM25 and semantic search.
- Reciprocal Rank Fusion (RRF) for rank aggregation.
- Cross-encoder reranking for top candidates.
- Knowledge-graph expansion through citation neighbors.
- Confidence-weighted answers using evidence and agreement signals.
- Dense citation injection in generated synthesis.

In this work, these are treated as design-level context unless directly validated by current runtime metrics.

## 3. Methods
### 3.1 Retrieval Formulation
In the currently deployed dense path, chunk relevance is estimated via normalized inner product:

\[ s_i = q^\top d_i \]

where q is the query embedding and d_i is the i-th chunk embedding.

In the target hybrid path, rankings from multiple retrievers are fused using reciprocal-rank fusion:

\[ \text{RRF}(i) = \sum_j \frac{1}{k + r_j(i)} \]

where r_j(i) is the rank of item i under retriever j and k is a smoothing constant.

### 3.2 Parsing and Indexing
Parsing is robust by design: if the primary structured parser fails, fallback extraction continues processing rather than dropping documents. Text is cleaned and chunked before embedding and indexing.

### 3.3 Confidence Modeling (Design-Level)
PaperWeave’s confidence design combines evidence volume, retrieval strength, graph support, agreement, and uncertainty penalties. This is treated as a roadmap model for calibrated confidence and future validation.

## 4. Experimental Setup
Evaluation is based on repository-native artifacts:
- Primary retrieval run: 20 questions.
- Secondary generation run: 10 questions.

System-scale context:
- 36 papers indexed.
- 3,383 chunks.
- Embedding dimension: 1024 (BGE-M3).
- Vector index: FAISS IndexFlatIP (~14 MB).
- Typical retrieval latency: 2–4 seconds.
- Typical generation latency: 5–8 seconds.

## 5. Results
### 5.1 Retrieval Performance (20-question run)
- Precision@1: 0.35
- Precision@5: 0.16
- Recall@5: 0.75
- MRR: 0.5204
- NDCG@5: 0.6258

Interpretation: The system shows strong retrieval coverage (high recall at k=5) with moderate precision and good rank quality.

### 5.2 Generation Quality
Secondary semantic run reports:
- ROUGE-1: 0.3067
- ROUGE-2: 0.0807
- ROUGE-L: 0.2135
- BLEU: 0.0236
- BERTScore-F1: 0.8620

The combination of low lexical overlap (BLEU/ROUGE sensitivity) and strong semantic alignment (BERTScore) is consistent with paraphrastic answer generation.

## 6. Architecture-Code Fidelity Findings
A central outcome of this study is the explicit separation between:
- Deployed functionality (ingestion, fallback parsing, dense FAISS retrieval, answer generation, recommendation).
- Implemented alternate functionality (hybrid retrieval + reranking in enhanced modules).
- Planned functionality (production graph retrieval, end-to-end GraphRAG context construction, dedicated observability dashboard).

This distinction is critical for journal-grade reporting, as it prevents overclaiming and improves reproducibility.

## 7. Discussion
Sanshodhak demonstrates practical utility for research assistance today, especially in high-recall literature retrieval and grounded answering. The PaperWeave specification contributes high-value system direction: richer hybrid retrieval, graph-guided expansion, and calibrated confidence. Together, they define a credible pathway from a robust production baseline to a more advanced graph-augmented retrieval stack.

Key bottlenecks:
- Default runtime path is not yet full hybrid.
- Graph retrieval remains artifact-level rather than online retrieval-level.
- Evaluation scripts need harmonization (e.g., recall definition consistency across runs).

## 8. Limitations and Threats to Validity
- Mixed evaluation setups (20-question and 10-question runs) reduce strict comparability.
- Some secondary metrics indicate potential implementation inconsistencies and require metric-code audit.
- Current experiments are domain-focused and modest in size.

## 9. Conclusion
Sanshodhak provides a strong implementation-grounded foundation for scholarly research intelligence, with measurable retrieval and semantic-generation capability. Incorporating the PaperWeave specification as structured design context allows a rigorous and transparent roadmap toward hybrid, graph-augmented, confidence-calibrated research assistants. Future work should prioritize default-path hybridization, online graph retrieval integration, and unified evaluation protocol standardization.

---

## Suggested Citation (Project Software)
```bibtex
@software{sanshodhak2026,
  title={Sanshodhak: Scholarly Research-Intelligence Platform},
  author={Project Team},
  year={2026},
  url={https://github.com/your-org/sanshodhak}
}
```
