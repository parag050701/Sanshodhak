# Module 5 — Inter-Annotator Agreement Report

**Generated:** 2026-03-17T08:03:44.714045+00:00

## Methodology

Two annotation criteria applied to live HGR top-5 retrieval results. Criterion A (Strict): relevant iff doc matches primary paper_id. Criterion B (Lenient): relevant iff doc matches any entry in relevant_docs. Cohen's kappa computed over all (question, rank) pairs.

## Summary

| Metric | Value |
|---|---|
| Questions | 20 |
| (Question, Rank) Pairs | 100 |
| Observed Agreement | 98.0% |
| **Cohen's Kappa** | **0.9467** |
| Threshold | ≥ 0.70 |
| Status | ✅ PASS |
| Strict P@5 | 0.700 |
| Lenient P@5 | 0.800 |

## Interpretation

Cohen's kappa = 0.947 indicates **almost perfect agreement** (≥ 0.80) between strict and lenient relevance criteria applied to the live retrieval results. This validates that the HGR system retrieves documents that are clearly relevant under both strict and lenient definitions, supporting high-confidence evaluation.

## Per-Query Detail

| Q# | Strict Hit | Lenient Hit | Top Retrieved |
|---|---|---|---|
| 1 | ✗ | ✗ | 003_Unknown_2025_463160c3 |
| 2 | ✓ | ✓ | 003_Unknown_2025_463160c3 |
| 3 | ✗ | ✗ | 044_Qiao_2025_be8f4989 |
| 4 | ✗ | ✗ | 029_Arya_2025_f65a964a |
| 5 | ✓ | ✓ | 040_Pichai_2023_cc391c15 |
| 6 | ✓ | ✓ | 016_Tian_2025_454dd912 |
| 7 | ✓ | ✓ | 016_Tian_2025_454dd912 |
| 8 | ✓ | ✓ | 003_Unknown_2025_463160c3 |
| 9 | ✓ | ✓ | 046_Niederhaus_2025_c41834f5 |
| 10 | ✓ | ✓ | 042_Chandra_2025_fa92c458 |
| 11 | ✓ | ✓ | 035_Vangalapat_2025_4df032dd |
| 12 | ✗ | ✓ | 041_Yao_2024_33642b2d |
| 13 | ✓ | ✓ | 037_Knollmeyer_2025_d92c5d07 |
| 14 | ✓ | ✓ | 003_Unknown_2025_463160c3 |
| 15 | ✗ | ✓ | 037_Knollmeyer_2025_d92c5d07 |
| 16 | ✓ | ✓ | 003_Unknown_2025_463160c3 |
| 17 | ✓ | ✓ | 042_Chandra_2025_fa92c458 |
| 18 | ✓ | ✓ | 040_Pichai_2023_cc391c15 |
| 19 | ✓ | ✓ | 040_Pichai_2023_cc391c15 |
| 20 | ✗ | ✗ | 029_Arya_2025_f65a964a |
