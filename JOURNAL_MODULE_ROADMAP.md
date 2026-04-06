# Sanshodhak Journal-Novelty Module Roadmap

Date: 2026-03-17
Owner: Sanshodhak team
Objective: Upgrade Sanshodhak from a promising project paper to a journal-grade, claim-safe, reproducible system while keeping runtime functionality intact.

## Principles

1. Claims must map to executable evidence files.
2. No manuscript claim without artifact provenance.
3. Every module ends with a working-system smoke check.
4. No destructive changes without a frozen baseline snapshot.

## Module Plan

### Module 1 — Baseline Freeze (Current)

Goal:
- Freeze current working state, metrics, manuscript claims, and artifact checksums.

Deliverables:
- `paper-intel/journal_baseline/baseline_manifest.json`
- `paper-intel/journal_baseline/baseline_manifest.md`
- Smoke-check result log for core assets.

Acceptance criteria:
- Core artifacts present (paper source, PDF, eval summary/report, score sheet, index files).
- Baseline metrics (HGR retrieval/generation/latency) extracted directly from `eval_summary.json`.
- Hash snapshot generated for reproducibility.

### Module 2 — Claim-Evidence Sync

Goal:
- Align manuscript wording with verified runtime behavior and saved artifacts.

Deliverables:
- Claim matrix (`claim -> file -> line/evidence`) in markdown.
- Patched paper text removing unsupported statements.

Acceptance criteria:
- No unsupported “dual-edge effective in evaluation” wording.
- No unsupported “152ms” statement unless retrieval-only benchmark artifact exists.
- Methods text matches evaluated runtime settings (adaptive budget vs fixed k/4).

### Module 3 — Citation-Edge Activation

Goal:
- Make citation-edge channel genuinely operational in evaluation corpus.

Deliverables:
- Improved citation matching pipeline (title normalization + DOI/metadata matching fallback).
- Artifact showing non-zero citation edges.

Acceptance criteria:
- Graph stats include non-zero citation-edge count in official run.
- Ablation rerun confirms whether citation edges add measurable value.

### Module 4 — Latency Benchmark Split

Goal:
- Separate retrieval-only latency from end-to-end (retrieval + generation).

Deliverables:
- `latency_retrieval_only.json`
- `latency_end_to_end.json`
- benchmark script and run command log.

Acceptance criteria:
- Both latency modes reported with mean/median/p95.
- Manuscript references the correct latency artifact explicitly.

### Module 5 — IAA Real Annotation

Goal:
- Replace simulated IAA with real independent annotation.

Deliverables:
- Real annotator files for the official question set.
- Cohen’s kappa report and adjudication protocol.

Acceptance criteria:
- No simulated labels in final evidence path.
- Kappa and annotation process reported transparently.

### Module 6 — Evaluation Scale-Up Toolkit

Goal:
- Raise study power and external credibility.

Deliverables:
- Expanded corpus build script/config.
- 100+ stratified questions.
- Updated scoring scripts for larger runs.

Acceptance criteria:
- Corpus and question-set size exceed current baseline by agreed thresholds.
- Full eval runs complete without breaking core pipeline.

### Module 7 — BEIR Full Ablation

Goal:
- Run VR-D/VR-H/GR/HGR on target BEIR subsets and report significance.

Deliverables:
- Full BEIR result table JSON/CSV.
- Correct significance tests (Wilcoxon) with effect sizes.

Acceptance criteria:
- No cherry-picked single-system reporting.
- Statistical test method and sample counts clearly documented.

### Module 8 — System Hardening

Goal:
- Ensure “fully working system” status with repeatable setup/run.

Deliverables:
- Reproducible runbook.
- Healthcheck script (imports, index load, search/query smoke).
- Dependency pin sanity pass.

Acceptance criteria:
- Fresh environment can run smoke workflow end-to-end.
- Failure modes return actionable diagnostics.

### Module 9 — Journal Package

Goal:
- Produce submission-ready manuscript package with traceable evidence.

Deliverables:
- Final TeX + PDF
- Artifact appendix linking each claim to files
- Reproducibility note and limitations section finalized

Acceptance criteria:
- All high-risk claims have artifact references.
- Known limitations are explicit and non-contradictory.
- Package compiles cleanly and is internally consistent.

## Execution Order

1. Module 1 (freeze)
2. Module 2 (claim safety)
3. Module 3 + 4 (technical evidence upgrades)
4. Module 5 + 6 + 7 (evaluation credibility)
5. Module 8 (operational reliability)
6. Module 9 (submission packaging)

## Module-by-Module Working-System Gate

At the end of each module:
1. Run artifact verification.
2. Run smoke command set.
3. Update journal evidence ledger.
4. Only then move to the next module.
