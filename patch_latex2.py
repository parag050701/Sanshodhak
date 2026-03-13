import re

with open("/home/admin-/Desktop/Sanshodhak/paper-intel/paper/main.tex", "r") as f:
    content = f.read()

# Update Related Work
old_rw = r"""\subsection{Retrieval-Augmented Generation}
Lewis et al.~\cite{lewis2020rag} established RAG as a viable
non-parametric memory mechanism for knowledge-intensive NLP.
Subsequent work introduced modular RAG variants~\cite{gao2023modular},
self-RAG with adaptive retrieval decisions~\cite{asai2023selfrag},
and iterative retrieval for multi-hop reasoning~\cite{trivedi2022interleaving}.
For scholarly domains, retrieval precision is particularly critical
because hallucinated citations or misattributed claims carry
downstream scientific risk~\cite{khalifa2023sourcenli}."""

new_rw = r"""\subsection{Retrieval-Augmented Generation}
Pre-trained models augmented with retrieval, such as REALM~\cite{guu2020realm} and Fusion-in-Decoder (FiD)~\cite{izacard2021fid}, established RAG as a viable non-parametric memory mechanism for knowledge-intensive NLP~\cite{lewis2020rag}. Subsequent work introduced modular RAG variants~\cite{gao2023modular}, self-RAG with adaptive retrieval decisions~\cite{asai2023selfrag}, and iterative retrieval for multi-hop reasoning~\cite{trivedi2022interleaving}. For scholarly domains, retrieval precision is particularly critical because hallucinated citations or misattributed claims carry downstream scientific risk~\cite{khalifa2023sourcenli}."""

content = content.replace(old_rw, new_rw)

old_dense = r"""\subsection{Dense and Hybrid Retrieval}
Dense retrieval via bi-encoders~\cite{karpukhin2020dpr} enables
semantic matching at scale through FAISS-based approximate nearest
neighbour search~\cite{johnson2019faiss}. BM25 term-frequency
retrieval remains competitive for keyword-rich queries, especially
those containing rare terms~\cite{robertson2009probabilistic}.
Hybrid approaches combining lexical and semantic signals outperform
either alone on BEIR-style benchmarks~\cite{thakur2021beir};
Reciprocal Rank Fusion (RRF)~\cite{cormack2009rrf} is a parameter-free
fusion method that provides robust cross-modal rank aggregation."""

new_dense = r"""\subsection{Dense and Hybrid Retrieval}
Dense retrieval via unsupervised methods like Contriever~\cite{izacard2022contriever} and supervised bi-encoders~\cite{karpukhin2020dpr, muennighoff2023mteb} enables semantic matching at scale using FAISS approximate nearest neighbour search~\cite{johnson2019faiss}. While learned sparse expansions like SPLADE~\cite{formal2021splade} improve upon classical term-matching, BM25 term-frequency retrieval remains highly competitive for keyword-rich scientific queries containing rare terms~\cite{robertson2009probabilistic}. Hybrid approaches combining lexical and semantic signals regularly outperform either independently on reasoning-intensive benchmarks like BEIR and BRIGHT~\cite{thakur2021beir, su2024bright}. To fuse these ranking signals, Reciprocal Rank Fusion (RRF)~\cite{cormack2009rrf} provides robust, parameter-free cross-modal rank aggregation."""

content = content.replace(old_dense, new_dense)

# Update the Citation Edge Claim to reflect truth
old_kg = r"""\textbf{Citation edges (Type~2).}
Reference sections are extracted via regex and matched to chunk
titles by a $\geq 3$-token overlap criterion. A citation edge
$(v_i, v_j)$ is inserted when chunk $i$'s reference section
mentions a paper whose title overlaps with chunk $j$'s source.

The resulting graph has 36 nodes, 443 edges,
mean degree 24.6, graph density 0.703, and a single
connected component—indicating high bibliographic cohesion in the
RAG corpus."""

new_kg = r"""\textbf{Citation edges (Type~2) [Experimental Limitation].}
Reference sections are extracted via regex and matched to chunk titles by a $\geq 3$-token overlap criterion. However, in our evaluated setup, format variations caused citation edges to yield 0 matched pairs. 

Consequently, the knowledge graph utilized in the AHR ablations relies exclusively on Vocabulary edges (Type 1). The resulting semantics-only knowledge graph possesses 36 nodes, 443 edges, mean degree 24.6, graph density 0.703, and a single connected component—indicating high TF-IDF structural cohesion in the scholarly corpus."""

content = content.replace(old_kg, new_kg)
content = content.replace("dual-edge", "single-edge \emph{vocabulary-only}")
content = content.replace("Dual-Edge", "Vocabulary-Jaccard")

with open("/home/admin-/Desktop/Sanshodhak/paper-intel/paper/main.tex", "w") as f:
    f.write(content)
