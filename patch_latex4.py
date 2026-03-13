import re

with open("/home/admin-/Desktop/Sanshodhak/paper-intel/paper/main.tex", "r") as f:
    content = f.read()

# Update Figure 1
old_fig = r"""\begin{figure*}[!t]
  \centering
  % ── pipeline figure ──────────────────────────────────────────────────
  % Replace with actual figure when available
  \fbox{\parbox{0.96\textwidth}{\centering
    \textbf{Figure 1: Sanshodhak End-to-End Pipeline}\\[4pt]
    \small
    \textbf{Ingestion Layer:}
    9 APIs (OpenAlex, CORE, S2, ArcSieve, DOAJ, PubMed, CrossRef, Unpaywall, arXiv)
    $\xrightarrow{\text{parallel}}$
    Dedup + Composite Rank
    $\xrightarrow{\text{OA-first}}$
    PDF Download\\[2pt]
    $\Downarrow$\\[2pt]
    \textbf{Processing Layer:}
    GROBID / PyMuPDF / pdfplumber / PyPDF2 (fallback cascade)
    $\rightarrow$ Clean + Chunk (512 tok / 64 tok overlap)
    $\rightarrow$ BGE-M3 Embed $\rightarrow$ FAISS + BM25\\[2pt]
    $\Downarrow$\\[2pt]
    \textbf{Retrieval Layer (AHR):}
    Query $\rightarrow$ [BM25 ranks $\oplus$ FAISS ranks $\oplus$ Graph expansion ranks]
    $\xrightarrow{\text{RRF}}$ Top-$k$ with slot reservation\\[2pt]
    $\Downarrow$\\[2pt]
    \textbf{Generation Layer:}
    DeepSeek-R1:7b (Ollama) $\rightarrow$ Cited answer + resource recommendations
  }}
  \caption{End-to-end Sanshodhak pipeline. AHR fuses three retrieval signals;
           the slot-reservation mechanism ensures graph-expanded chunks are
           always represented in the final context.}
  \label{fig:pipeline}
\end{figure*}"""

new_fig = r"""\begin{figure*}[!t]
\centering
\resizebox{0.9\textwidth}{!}{%
\begin{tikzpicture}[
  node distance=1.5cm and 1.2cm,
  >=Stealth,
  font=\sffamily\small,
  box/.style={draw, rounded corners, inner sep=6pt, align=center, fill=gray!10, text width=2.8cm},
  layer/.style={draw, dashed, inner sep=12pt, rounded corners=8pt, fill=blue!5},
  layerlabel/.style={font=\sffamily\bfseries, above right, anchor=south west}
]

% INGESTION LAYER
\node[box, fill=green!10] (apis) {9 Academic APIs\\(Parallel Fetch)};
\node[box, right=of apis] (dedup) {Deduplicate \&\\Composite Rank};
\node[box, right=of dedup] (pdf) {OA-First\\PDF Download};
\draw[->] (apis) -- (dedup);
\draw[->] (dedup) -- (pdf);

\begin{pgfonlayer}{background}
\node[layer, fit=(apis) (pdf)] (ingest_bg) {};
\node[layerlabel] at (ingest_bg.north west) {Ingestion Layer};
\end{pgfonlayer}

% PROCESSING LAYER
\node[box, below=1.8cm of apis, fill=yellow!10] (parse) {Parsing Cascade\\(GROBID/PyMuPDF)};
\node[box, right=of parse] (chunk) {Clean \& Chunk\\(512 tok)};
\node[box, right=of chunk, fill=orange!10] (index) {BM25 + FAISS\\+ Jaccard Graph};
\draw[->] (pdf.south) |- ($ (pdf.south) - (0,0.8) $) -| (parse.north);
\draw[->] (parse) -- (chunk);
\draw[->] (chunk) -- (index);

\begin{pgfonlayer}{background}
\node[layer, fill=yellow!5, fit=(parse) (index)] (process_bg) {};
\node[layerlabel] at (process_bg.north west) {Processing Layer};
\end{pgfonlayer}

% RETRIEVAL LAYER (AHR)
\node[box, fill=blue!10, below=1.8cm of parse, text width=2.2cm] (q) {User Query};
\node[box, right=0.8cm of q, text width=1.5cm, fill=red!10] (bm25) {BM25 Rank};
\node[box, above=0.3cm of bm25, text width=1.5cm, fill=red!10] (faiss) {FAISS Dense};
\node[box, below=0.3cm of bm25, text width=1.5cm, fill=red!10] (graph) {Graph Expand};

\node[box, right=1.2cm of bm25, fill=purple!10] (rrf) {Reciprocal Rank\\Fusion (RRF)};
\node[box, right=of rrf] (topk) {Top-$k$ Context\\(Slot Reserved)};

\draw[->] (index.south) |- ($ (index.south) - (0,0.4) $) -| (faiss.north);

\draw[->] (q) -- (bm25);
\draw[->] (q) |- (faiss);
\draw[->] (q) |- (graph);

\draw[->] (faiss) -| (rrf);
\draw[->] (bm25) -- (rrf);
\draw[->] (graph) -| (rrf);
\draw[->] (rrf) -- (topk);

\begin{pgfonlayer}{background}
\node[layer, fill=red!5, fit=(q) (topk) (faiss) (graph)] (retrieval_bg) {};
\node[layerlabel] at (retrieval_bg.north west) {Adaptive Hybrid Retrieval (AHR) Layer};
\end{pgfonlayer}

% GENERATION LAYER
\node[box, fill=teal!10, below=1.8cm of rrf] (llm) {DeepSeek-R1\\Generation};
\node[box, right=of llm] (output) {Cited Answer +\\Recommendations};

\draw[->] (topk.south) |- ($ (topk.south) - (0,0.6) $) -| (llm.north);
\draw[->] (llm) -- (output);

\begin{pgfonlayer}{background}
\node[layer, fill=teal!5, fit=(llm) (output)] (gen_bg) {};
\node[layerlabel] at (gen_bg.north west) {Generation Layer};
\end{pgfonlayer}

\end{tikzpicture}
}
\caption{End-to-end Sanshodhak pipeline featuring the Adaptive Hybrid Retrieval (AHR) mechanism. The slot-reservation module forces graph-expanded chunks into the final RRF $k$-context.}
\label{fig:pipeline}
\end{figure*}"""

content = content.replace(old_fig, new_fig)

with open("/home/admin-/Desktop/Sanshodhak/paper-intel/paper/main.tex", "w") as f:
    f.write(content)
