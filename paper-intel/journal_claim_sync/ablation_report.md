# Module 7 — Ablation Study Report

**Generated:** 2026-03-17T08:17:13.605322+00:00
**Questions:** 20 (domain-specific RAG evaluation set)

## Component Contribution Analysis

Four system configurations evaluated by incrementally adding components:

| System | Description | P@1 | R@5 | NDCG@5 | MRR | ΔP@1 | ΔNDCG@5 | ΔMRR |
|---|---|---|---|---|---|---|---|---|
| **VR-D** | Dense only (FAISS) | 0.350 | 0.825 | 0.605 | 0.546 | — | — | — |
| **VR-H** | +BM25 (Hybrid) | 0.300 | 0.725 | 0.533 | 0.479 | -0.050 | -0.072 | -0.067 |
| **GR** | +Graph expansion (no BM25) | 0.350 | 0.800 | 0.595 | 0.533 | — | -0.010 | -0.013 |
| **HGR** | +BM25 +Graph (full system) | 0.450 | 0.850 | 0.665 | 0.627 | +0.100 | +0.060 | +0.081 |

## Generation Quality

| System | ROUGE-1 | ROUGE-L | BERTScore-F1 | Latency Mean (s) |
|---|---|---|---|---|
| **VR-D** | 0.000 | 0.000 | 0.000 | 14.23 |
| **VR-H** | 0.000 | 0.000 | 0.000 | 12.88 |
| **GR** | 0.000 | 0.000 | 0.000 | 15.42 |
| **HGR** | 0.000 | 0.000 | 0.000 | 14.80 |

## Total HGR Improvement over Dense Baseline

| Metric | Baseline (VR-D) | HGR | Δ |
|---|---|---|---|
| P@1 | 0.350 | 0.450 | **+0.100** |
| R@5 | 0.825 | 0.850 | **+0.025** |
| NDCG@5 | 0.605 | 0.665 | **+0.060** |
| MRR | 0.546 | 0.627 | **+0.081** |

## LaTeX Table (for manuscript)

```latex
\begin{table}[t]
\centering
\caption{Ablation study: contribution of each retrieval component.
$\Delta$ columns show improvement over the Dense-only baseline (VR-D).}
\label{tab:ablation}
\begin{tabular}{llccccccc}
\toprule
System & Description & P@1 & R@5 & NDCG@5 & MRR & $\Delta$P@1 & $\Delta$NDCG@5 & $\Delta$MRR \\
\midrule
VR-D & Dense only (FAISS) & 0.350 & 0.825 & 0.605 & 0.546 & — & — & — \\
VR-H & +BM25 (Hybrid) & 0.300 & 0.725 & 0.533 & 0.479 & -0.050 & -0.072 & -0.067 \\
GR & +Graph expansion (no BM25) & 0.350 & 0.800 & 0.595 & 0.533 & — & -0.010 & -0.013 \\
\textbf{HGR} & +BM25 +Graph (full system) & \textbf{0.450} & \textbf{0.850} & \textbf{0.665} & \textbf{0.627} & \textbf{+0.100} & \textbf{+0.060} & \textbf{+0.081} \\
\bottomrule
\end{tabular}
\end{table}
```

