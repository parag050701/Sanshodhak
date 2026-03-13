```mermaid
flowchart TD
    A[User Query: Research Topic] --> B[Phase 0: Paper Collection]
    
    B --> C[max_orchestrator.py]
    C --> D[CrossRef 10X Search]
    D --> E[Semantic Filtering]
    E --> F[Download with Backpropagation]
    
    F --> G{Download Sources}
    G -->|Priority| H[Unpaywall]
    G -->|Shadow| I[Sci-Hub]
    G -->|Shadow| J[Anna's Archive]
    G -->|Backup| K[OpenAlex/CORE/S2]
    
    H --> L[PDF Saved]
    I --> L
    J --> L
    K --> L
    
    L --> M[🆕 Auto-Processor Trigger]
    M --> N[Phase 1: PDF Processing]
    
    N --> O{GROBID Available?}
    O -->|Yes| P[GROBID Processing]
    O -->|No| Q[Fallback Extraction]
    
    P --> R{Quality Check}
    R -->|Good| S[Parse TEI XML]
    R -->|Poor/Failed| Q
    
    Q --> T{Try PyMuPDF}
    T -->|Success| U[Extract Text + Metadata]
    T -->|Fail| V{Try pdfplumber}
    V -->|Success| U
    V -->|Fail| W{Try PyPDF2}
    W -->|Success| U
    W -->|Fail| X[Error Logged]
    
    S --> Y[Clean Text]
    U --> Y
    
    Y --> Z[Split Sections]
    Z --> AA[Create Structured JSON]
    
    AA --> AB{Save Outputs}
    AB --> AC[JSON File]
    AB --> AD[TXT File]
    
    AC --> AE[Ready for Phase 2: Knowledge Graph]
    AD --> AE
    X --> AE
    
    style M fill:#90EE90
    style N fill:#90EE90
    style Q fill:#FFD700
    style T fill:#FFD700
    style V fill:#FFD700
    style W fill:#FFD700
    style AE fill:#87CEEB

classDef new fill:#90EE90,stroke:#228B22,stroke-width:3px
classDef fallback fill:#FFD700,stroke:#FF8C00,stroke-width:2px
classDef ready fill:#87CEEB,stroke:#4682B4,stroke-width:3px

class M,N new
class Q,T,V,W fallback
class AE ready
```

# System Architecture Diagram

## Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT: Research Topic                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 0: PAPER COLLECTION                      │
│                    (max_orchestrator.py)                         │
├─────────────────────────────────────────────────────────────────┤
│  1. CrossRef 10X Search  →  Massive paper pool                  │
│  2. Semantic Filtering   →  Best papers                         │
│  3. Download Priority:                                           │
│     ├─ Unpaywall (DOIs)                                          │
│     ├─ Sci-Hub (shadow)                                          │
│     ├─ Anna's Archive (shadow)                                   │
│     └─ OpenAlex/CORE/S2/arXiv                                    │
│  4. Backpropagation      →  If <100%, expand & retry            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   PDF SAVED    │
                    └────────┬───────┘
                             │
                  ┌──────────▼───────────┐
                  │  🆕 AUTO-PROCESSOR   │  ← NEW INTEGRATION
                  │  (immediate trigger) │
                  └──────────┬───────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1: PDF PROCESSING                        │
│                     (pdf_to_text.py)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │          PRIMARY: GROBID Processing                 │       │
│  │  ┌───────────────────────────────────────────────┐  │       │
│  │  │ 1. Send PDF → GROBID server                   │  │       │
│  │  │ 2. Receive TEI XML                            │  │       │
│  │  │ 3. Parse structure (title, authors, sections) │  │       │
│  │  │ 4. Extract body text + references             │  │       │
│  │  └───────────────────────────────────────────────┘  │       │
│  └─────────────────┬───────────────────────────────────┘       │
│                    │                                             │
│                    ▼                                             │
│         ┌──────────────────────┐                                │
│         │   Quality Check      │                                │
│         │  (text length > 500) │                                │
│         └──────┬───────────────┘                                │
│                │                                                 │
│       ┌────────┴────────┐                                       │
│       │                 │                                       │
│  ✅ GOOD           ❌ POOR/FAIL                                 │
│       │                 │                                       │
│       │                 ▼                                       │
│       │    ┌────────────────────────────────┐                  │
│       │    │  🆕 FALLBACK EXTRACTION       │                   │
│       │    │  (PyMuPDF → pdfplumber → PyPDF2) │                │
│       │    ├────────────────────────────────┤                  │
│       │    │ 1. Try PyMuPDF (fitz)         │                   │
│       │    │    - Fastest, best quality     │                   │
│       │    │    - Extracts metadata         │                   │
│       │    │                                │                   │
│       │    │ 2. If fail → pdfplumber       │                   │
│       │    │    - Good for tables           │                   │
│       │    │                                │                   │
│       │    │ 3. If fail → PyPDF2           │                   │
│       │    │    - Lightweight fallback      │                   │
│       │    └────────────┬───────────────────┘                  │
│       │                 │                                       │
│       └─────────────────┴───────────────────┐                  │
│                                              │                  │
│                        ┌─────────────────────▼────────────┐    │
│                        │  Common Processing               │    │
│                        ├──────────────────────────────────┤    │
│                        │ 1. Clean text (unicode, hyphens) │    │
│                        │ 2. Split into sections           │    │
│                        │ 3. Create structured JSON        │    │
│                        │ 4. Save outputs                  │    │
│                        └─────────────────┬────────────────┘    │
│                                          │                      │
└──────────────────────────────────────────┼──────────────────────┘
                                           │
                                           ▼
                    ┌───────────────────────────────────┐
                    │       Save Outputs                │
                    ├───────────────────────────────────┤
                    │  JSON: processed_json/{id}.json   │
                    │  TXT:  processed_text/{id}.txt    │
                    │  Metadata: extraction_method      │
                    └───────────────────┬───────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │   READY FOR PHASE 2               │
                    │   (Knowledge Graph Construction)  │
                    └───────────────────────────────────┘
```

## Success Rate Flow

```
Test Results (29 PDFs):
├─ GROBID Success: 100%
├─ Fallback Triggered: 0%
├─ Total Success: 29/29 (100%)
└─ Avg Processing: 1.1 PDFs/sec

Expected Production:
├─ GROBID Success: ~90-95%
├─ Fallback Success: ~95-99%
└─ Combined Success: ~99.5%
```
