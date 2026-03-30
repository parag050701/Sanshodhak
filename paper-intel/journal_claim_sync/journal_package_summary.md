# Journal Readiness Package — Module-wise Execution Summary

Generated: 2026-03-17

This package summarizes all module outputs produced in the journal-readiness workflow.

## Module Completion Status

| Module | Objective | Status | Key Artifact(s) |
|---|---|---|---|
| 1 | Baseline freeze | ✅ Completed | `journal_baseline/baseline_manifest.json`, `journal_baseline/baseline_manifest.md` |
| 2 | Claim-evidence synchronization | ✅ Completed | `claim_sync_report.json`, `claim_matrix.md` |
| 3 | Citation-edge activation validation | ✅ Completed | `citation_activation_report.json`, `citation_activation_report.md` |
| 4 | Retrieval vs end-to-end latency split | ✅ Completed | `latency_split_report.json`, `latency_split_report.md` |
| 5 | Real IAA analysis (live retrieval grounded) | ✅ Completed | `iaa_report.json`, `iaa_report.md` |
| 6 | Scale-up experiments | ✅ Completed | `../scalability_results/scalability_results.json`, `../scalability_results/plots/*` |
| 7 | Full ablation report + LaTeX table | ✅ Completed | `ablation_report.json`, `ablation_report.md`, `ablation_table.tex` |
| 8 | Hardening / regression checks | ✅ Completed | py_compile pass + claim sync pass |
| 9 | Final journal package assembly | ✅ Completed | `journal_package_summary.md` (this file) |

## Key Verified Outcomes

- **Claim safety:** `claim_sync_report.json` reports overall `pass`.
- **Citation-edge evidence:** citation edges are present (`citation_edges = 6`), validating dual-edge graph capability.
- **Latency split available:** retrieval-only and end-to-end latency are now separated in dedicated artifacts.
- **IAA threshold exceeded:** Cohen's kappa = **0.9467** (threshold ≥ 0.70).
- **Ablation contribution quantified:** HGR improves over Dense baseline by:
  - ΔP@1 = **+0.100**
  - ΔNDCG@5 = **+0.060**
  - ΔMRR = **+0.081**
- **Scale-up evidence generated:** scaling traces available for 1k/5k/10k corpus sizes with plots.

## Artifacts Index

- Baseline:
  - `../journal_baseline/baseline_manifest.json`
  - `../journal_baseline/baseline_manifest.md`
- Claim Sync & Safety:
  - `claim_sync_report.json`
  - `claim_matrix.md`
- Citation Activation:
  - `citation_activation_report.json`
  - `citation_activation_report.md`
- Latency Split:
  - `latency_split_report.json`
  - `latency_split_report.md`
- IAA:
  - `iaa_report.json`
  - `iaa_report.md`
- Ablation:
  - `ablation_report.json`
  - `ablation_report.md`
  - `ablation_table.tex`
- Scale-up:
  - `../scalability_results/scalability_results.json`
  - `../scalability_results/plots/build_time.png`
  - `../scalability_results/plots/memory.png`
  - `../scalability_results/plots/latency.png`

## Re-run Commands

```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel

python3 journal_baseline_freeze.py
python3 claim_evidence_sync.py
python3 journal_citation_activation_check.py
python3 latency_split_benchmark.py --sample 20
python3 journal_iaa_module.py
python3 scalability_analysis.py --sizes 1000 5000 10000 --no-graph --output-dir scalability_results
python3 journal_ablation_module.py
python3 -m py_compile journal_baseline_freeze.py journal_citation_activation_check.py latency_split_benchmark.py journal_iaa_module.py journal_ablation_module.py
```
