# Sanshodhak Journal Claim Audit

Date: 2026-03-16
Scope: `sanshodhak_fixed.tex`, `Sanshodhak (2).pdf`, core implementation in `paper-intel/`, evaluation artifacts in `paper-intel/*.json` and `paper-intel/*.md`

## Executive Verdict

The manuscript is based on a real system and most headline retrieval metrics are traceable to repository artifacts. However, the current paper still overclaims in several places that matter for journal review.

Current status:
- Strongly supported: nine-source ingestion capability, four-way ablation structure, most retrieval/generation metrics, graph statistics, composite ranking, and recommendation subsystem existence.
- Partially supported or contradicted: dual-edge graph contribution in evaluation, fixed slot-reservation description, and the reported `152 ms` retrieval latency.
- Publication readiness: not yet journal-ready. The main blockers are claim-method mismatches, small evaluation scale, simulated IAA artifacts, and insufficiently defensible novelty wording.

## What Is Verified

### 1. Nine-source ingestion exists in code

The repo does implement the claimed multi-source ingestion layer.

Evidence:
- `paper-intel/ingestion/discovery/search_engine.py`
- Clients present for OpenAlex, CORE, Semantic Scholar, ArcSieve, DOAJ, PubMed, CrossRef, Unpaywall, and arXiv
- Parallel search is implemented via `ThreadPoolExecutor`

Assessment:
- This is a valid system claim.
- Wording should remain `supports nine sources` rather than implying every experiment actively used all nine.

### 2. Composite temporal-aware ranking exists in code

The paper's composite ranking formula is supported by implementation.

Evidence:
- `paper-intel/query_expander.py`
- `paper-intel/ingestion/discovery/search_engine.py`

Verified behavior:
- citation normalization
- exponential temporal decay with approximately 5-year half-life
- open-access bonus

Assessment:
- This is a valid methods claim.

### 3. Four-way ablation exists and the retrieval metrics are real

The four systems `VR-D`, `VR-H`, `GR`, and `HGR` are instantiated in the evaluation code and their scores match the paper.

Evidence:
- `paper-intel/eval_compare.py`
- `paper-intel/eval_summary.json`
- `paper-intel/eval_report.txt`
- `paper-intel/PAPER_SCORE_SHEET.md`

Verified HGR retrieval metrics:
- `P@1 = 0.45`
- `R@5 = 0.85`
- `R@10 = 0.925`
- `NDCG@5 = 0.6653`
- `NDCG@10 = 0.6940`
- `MRR = 0.6267`
- `MAP = 0.6092`

Assessment:
- These are safe to report if cited exactly from the artifact files.

### 4. Generation metrics are mostly real and traceable

Evidence:
- `paper-intel/eval_summary.json`
- `paper-intel/eval_report.txt`

Verified HGR generation metrics:
- `ROUGE-1 = 0.3457`
- `ROUGE-2 = 0.1001`
- `ROUGE-L = 0.2330`
- `BLEU = 0.0341`
- `BERTScore-F1 = 0.8754`

Assessment:
- These are supported.
- The manuscript should explicitly keep the note that lexical metrics are low relative to semantic metrics.

### 5. Graph statistics match the evaluation artifacts

Evidence:
- `paper-intel/eval_summary.json`
- `paper-intel/eval_report.txt`
- `paper-intel/PAPER_SCORE_SHEET.md`

Verified graph statistics:
- `36 nodes`
- `443 edges`
- `avg_degree = 24.6111`
- `density = 0.7032`
- `connected_components = 1`

Assessment:
- These values are safe.

### 6. Recommendation subsystem exists

Evidence:
- `paper-intel/resource_recommender.py`
- `paper-intel/resource_recommender_v2.py`

Verified integrations:
- HuggingFace
- GitHub
- Papers with Code

Assessment:
- The subsystem exists and is a valid system feature.
- The paper should avoid claiming recommendation quality unless separately evaluated.

## What Is Not Safe to Claim As Written

### 1. `Dual-edge knowledge graph` is not supported by the evaluated results

This is the most important manuscript mismatch.

Evidence:
- `paper-intel/PAPER_SCORE_SHEET.md` states `Citation edges = 0`
- `paper-intel/eval_summary.json` only reports the final graph statistics, which match a vocabulary-only graph in practice
- `sanshodhak_fixed.tex` itself later admits citation edges yielded zero matched pairs

Problem:
- The abstract, title, and contribution list present the evaluated graph as dual-edge.
- The actual evaluation evidence shows the reported retrieval gains come from vocabulary-Jaccard edges only.

Safe rewrite:
- Replace `dual-edge knowledge graph` with `lightweight vocabulary-Jaccard graph; citation-edge mechanism implemented but inactive in the evaluated corpus`.
- If you want to keep `dual-edge`, it must be explicitly framed as an architectural design, not an experimentally validated contributor.

### 2. The paper describes fixed `k/4` slot reservation, but the evaluated runtime defaults to adaptive allocation

Evidence:
- `paper-intel/graph_rag.py` uses `use_adaptive_budget=True` by default
- `paper-intel/core/adaptive_budget.py` implements data-driven graph slot allocation
- `paper-intel/eval_compare.py` instantiates `GraphRAG(...)` without disabling adaptive budgeting

Problem:
- `sanshodhak_fixed.tex` describes the method as fixed slot reservation with `k_g = max(1, k/4)`.
- The actual evaluated code path appears to use adaptive graph budgeting unless manually overridden.

Safe rewrite options:
- Option A: rerun experiments with adaptive budgeting disabled and keep the fixed-slot paper narrative.
- Option B: update the paper to describe adaptive graph budgeting as the default evaluated method.

Current assessment:
- As written, the methods section does not faithfully describe the most likely evaluated runtime.

### 3. The `152 ms retrieval latency` claim is not supported by the main evaluation artifacts

Evidence:
- `paper-intel/eval_summary.json` reports HGR latency `mean_s = 14.7978`
- `paper-intel/eval_report.txt` reports HGR mean latency `14.798` seconds
- `paper-intel/eval_compare.py` aggregates total latency including generation when generation is enabled

Important nuance:
- `paper-intel/api/main.py` contains a retrieval endpoint that measures latency in milliseconds.
- That supports the possibility that a retrieval-only benchmark could be around `152 ms`.
- But there is no repository artifact currently tying the paper's `152 ms` value to a saved benchmark run.

Current assessment:
- The paper's latency table is not reproducible from the primary evaluation artifacts currently in the repo.
- For journal submission, this number must either be re-benchmarked and saved as a separate retrieval-only artifact, or replaced with the measured end-to-end latency from `eval_summary.json`.

### 4. IAA evidence is not currently journal-valid

Evidence:
- `paper-intel/iaa_results.json` explicitly says `SIMULATED annotations for pipeline demonstration`
- `paper-intel/annotations_annotator1.json` and `annotations_annotator2.json` use generic demo-style IDs, not the paper's real evaluation questions

Assessment:
- You cannot use current IAA artifacts to support evaluation validity in a journal paper.
- If the manuscript mentions Cohen's kappa or annotation reliability, that must be removed until real independent annotation is completed.

### 5. `First published` is not a safe novelty phrase

Problem:
- This is extremely hard to prove and reviewers will attack it immediately.

Current assessment:
- The repo does support a concrete four-way ablation on this corpus.
- It does not prove that no prior publication has done something similar.

Safe rewrite:
- Use `to our knowledge, this is among the first open-source four-way ablations...`
- Or more conservatively: `we provide an open-source four-way ablation...`

## Metric Cross-Check Against the Manuscript

Supported by artifacts:
- `P@1 = 0.45`
- `NDCG@5 = 0.665`
- `NDCG@10 = 0.694`
- `MRR = 0.627`
- `MAP = 0.609`
- `ROUGE-1 = 0.346`
- `BERTScore-F1 = 0.875`

Needs correction or stronger provenance:
- `152 ms retrieval latency`
- Any statement implying citation edges materially contributed to the reported gains
- Any statement implying fixed slot reservation if the run used adaptive budgeting

## Journal Novelty Assessment

### Novelty that is defensible

The strongest defensible novelty lies in the combination, not in any single component:

1. A reproducible scholarly-RAG system that combines:
- multi-source ingestion
- hybrid retrieval
- lightweight graph expansion
- local generation
- resource recommendation

2. A four-way internal ablation across:
- dense-only
- hybrid-only
- graph-only
- hybrid+graph

3. A lightweight graph design that avoids external NER and graph databases

4. A practical scholarly-system framing rather than a pure benchmark paper

### Novelty that is risky or overstated

1. `Dual-edge` as an evaluated contribution
- Not defensible with zero active citation edges in the evaluated corpus.

2. `First published`
- Too strong unless you perform a dedicated novelty search and still word it carefully.

3. `Production-oriented` for journal positioning
- Acceptable as a systems framing, but reviewers will expect stronger scale and stronger evaluation rigor than the current 36-paper corpus and 20-question setup.

## Publication Readiness

### Ready now for

- internal technical report
- project paper
- workshop/demo-style submission after claim cleanup

### Not ready yet for strong journal submission

Main reasons:
- corpus too small for journal-grade IR claims
- only 20 evaluation questions in the main ablation
- no real independent IAA artifact
- methods/claim mismatch around graph budget
- unsupported latency provenance
- overstated novelty wording

## Minimum Changes Before Journal Submission

1. Rewrite the graph claim.
Use `vocabulary-Jaccard graph` as the evaluated component, and move citation edges to future work or architecture-only language.

2. Resolve the slot-budget mismatch.
Either rerun with fixed `k/4` or rewrite the paper for adaptive graph budgeting.

3. Re-benchmark latency and save a dedicated artifact.
Store retrieval-only latency results in a versioned JSON or CSV and cite that file in the paper.

4. Remove any simulated-evaluation validity language.
Do not report IAA until it is based on real annotators.

5. Soften novelty wording.
Replace `first published` with `to our knowledge` or a narrower descriptive claim.

6. Scale the evaluation.
For journal review, move toward:
- 500+ papers
- 3+ domains
- 100+ questions
- real significance testing on the main ablation
- external benchmark comparison

## Recommended Manuscript Positioning Right Now

Best current positioning:
- `implementation-grounded scholarly RAG system paper`
- `open-source systems paper with internal ablation evidence`
- `engineering contribution with clear roadmap to larger-scale validation`

Avoid positioning as:
- definitive top-tier IR benchmark paper
- validated dual-edge graph retrieval paper
- statistically mature journal study in its current form

## Bottom Line

The system is real, the core retrieval gains are real, and there is genuine publishable value here. But the current manuscript needs claim tightening before journal submission. The biggest fixes are straightforward:
- stop claiming evaluated dual-edge gains,
- align the methods section with the adaptive-budget code or rerun fixed-budget experiments,
- replace the unsupported latency claim with artifact-backed measurements,
- remove simulated validity evidence,
- and soften the novelty language.

With those changes, the paper becomes much more credible. With larger-scale evaluation and real annotation reliability, it becomes a realistic journal candidate.