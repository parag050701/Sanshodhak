# Module 2 — Claim-Evidence Sync

This folder stores manuscript claim-safety checks against repository artifacts.

## Files

- `claim_sync_report.json`: machine-readable status
- `claim_matrix.md`: publication-ready claim checklist

## Run

```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel
python3 claim_evidence_sync.py
```

## Purpose

Prevent journal-risk regressions by automatically checking:

1. novelty overclaims (`first published`)
2. graph claim consistency (dual-edge wording vs citation-edge evidence)
3. unsupported latency claim (`152 ms`)
4. methods/runtime consistency (adaptive graph budgeting)
5. core metric consistency with `eval_summary.json`
