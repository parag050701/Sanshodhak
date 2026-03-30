# Latency Split Benchmark

- Generated UTC: 2026-03-17T07:55:37.707561+00:00
- Questions benchmarked: 20
- Index: `paper-intel/rag_index`
- Questions file: `paper-intel/rag_test_questions.json`

| System | Retrieval Mean (s) | E2E Mean (s) | Gen Est Mean (s) | Retrieval P95 (s) | E2E P95 (s) |
|---|---:|---:|---:|---:|---:|
| VR-D (Dense) | 0.285 | 0.134 | 0.005 | 0.322 | 0.157 |
| VR-H (Hybrid) | 0.138 | 0.143 | 0.009 | 0.152 | 0.168 |
| GR (Graph) | 0.135 | 0.140 | 0.011 | 0.161 | 0.161 |
| HGR (Hybrid+Graph) | 0.181 | 0.181 | 0.010 | 0.213 | 0.209 |

## Notes
- End-to-end latency is measured directly from full `query()` execution.
- Generation-only latency is estimated from separate runs and should be treated as approximate.
