# Research Paper Outline (Sanshodhak)

## Working Title
Sanshodhak: A Scholarly Research-Intelligence Platform with Multi-Source Ingestion, Retrieval-Augmented Answering, and a Hybrid Retrieval Roadmap

---

## 0. Front Matter
### 0.1 Title
- Keep `Sanshodhak` as the primary system name.
- Mention PaperWeave only as product-spec/design context if needed.

### 0.2 Author Block
- Authors, affiliations, emails.

### 0.3 Abstract (150–250 words)
**Goal**
- Problem, approach, key results, and contribution in one compact narrative.

**Include**
- End-to-end pipeline summary.
- Core metrics: P@1, R@5, MRR, NDCG@5, BERTScore-F1.
- Architecture-code fidelity contribution.

### 0.4 Keywords
- RAG, scholarly search, hybrid retrieval, FAISS, knowledge graph, research intelligence.

---

## 1. Introduction
### 1.1 Motivation
- Literature review bottlenecks and fragmented tools.

### 1.2 Problem Statement
- Need for a practical, transparent, evidence-grounded assistant.

### 1.3 System Framing
- Sanshodhak as deployed system.
- PaperWeave as design-scope context.

### 1.4 Contributions (bullet list)
- System contribution.
- Evaluation contribution.
- Fidelity-analysis contribution.

### 1.5 Paper Structure
- One paragraph roadmap of sections.

---

## 2. Related Work
### 2.1 Retrieval-Augmented Generation
- Foundational RAG and grounded generation.

### 2.2 Retrieval Methods
- Dense retrieval, BM25, reciprocal rank fusion, reranking.

### 2.3 Evaluation of RAG Systems
- Retrieval metrics and generation metrics.
- Lexical vs semantic evaluation.

### 2.4 Positioning Against Prior Systems
- What is novel/practical in Sanshodhak.

---

## 3. System Architecture
### 3.1 End-to-End Pipeline (Deployed)
- Search → Download → Parse → Embed/Index → Retrieve/Answer → Recommend.

### 3.2 Component-Level Description
- Ingestion clients.
- Parsing/fallback stack.
- Indexing and retrieval stack.
- Recommendation engine.
- API/UI layer.

### 3.3 Design Context from PaperWeave
- Hybrid retrieval stages.
- Knowledge-graph expansion concept.
- Confidence-weighted output model.

### 3.4 Deployed vs Planned Clarification
- Explicitly mark what is currently operational vs roadmap.

---

## 4. Methods
### 4.1 Data Acquisition and Processing
- Source selection, deduplication, parsing, chunking.

### 4.2 Retrieval Formulation
- Dense scoring equation.
- Optional hybrid/RRF equation as target method.

### 4.3 Generation and Attribution
- Context construction and citation-aware answer synthesis.

### 4.4 Confidence Modeling (Design-Level)
- Multi-factor confidence equation and interpretation.

### 4.5 Complexity / Runtime Notes
- Practical time and resource behavior.

---

## 5. Experimental Setup
### 5.1 Environment and Models
- Embedding model, LLM, vector index settings.

### 5.2 Datasets and Evaluation Splits
- 20-question retrieval run.
- 10-question generation run.

### 5.3 Metrics
- P@k, R@k, NDCG@k, MRR.
- ROUGE, BLEU, BERTScore.

### 5.4 Reproducibility Artifacts
- Exact result files and scripts used.

---

## 6. Results
### 6.1 Retrieval Results
- Main table for primary run.
- Short interpretation.

### 6.2 Generation Results
- Lexical + semantic metrics table.
- Explain paraphrase effect.

### 6.3 Latency and Throughput Observations
- Practical runtime profile.

### 6.4 Error/Caveat Reporting
- Metric inconsistencies (e.g., recall > 1 in secondary run) and handling.

---

## 7. Architecture-Code Fidelity Analysis
### 7.1 Verification Method
- How architecture was cross-checked with code.

### 7.2 Layer-wise Status
- Implemented / Partial / Planned matrix.

### 7.3 Implications for Scientific Reporting
- Why fidelity reporting matters.

---

## 8. Discussion
### 8.1 What Works Well
- Retrieval coverage, grounded answers, practical utility.

### 8.2 Current Limitations
- Default dense path vs hybrid target.
- Graph retrieval integration gap.

### 8.3 Roadmap
- Hybrid as default.
- Graph runtime integration.
- Unified evaluation pipeline.

---

## 9. Threats to Validity
### 9.1 Internal Validity
- Script-level metric variability.

### 9.2 Construct Validity
- Lexical metric limitations for paraphrases.

### 9.3 External Validity
- Domain size and dataset scope.

---

## 10. Ethics, Compliance, and Responsible Use
- Legal-source-first retrieval policy.
- Citation/provenance transparency.
- Risk of over-trust and mitigation.

---

## 11. Conclusion
- One-paragraph summary.
- Main contributions and forward path.

---

## 12. References
- RAG, FAISS, BM25, ROUGE, BLEU, BERTScore, NDCG/MRR.

---

# Section Fill Order (One-by-One Workflow)
1. Abstract
2. Introduction
3. System Architecture
4. Methods
5. Experimental Setup
6. Results
7. Fidelity Analysis
8. Discussion
9. Threats to Validity
10. Ethics
11. Conclusion
12. Final reference polish and formatting pass

---

# Per-Section Writing Checklist
For each section we fill:
- Target claim(s)
- Evidence source(s)
- Numbers included
- Caveats stated
- Transition sentence to next section
