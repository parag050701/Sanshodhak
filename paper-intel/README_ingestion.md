# Sanshodhak Ingestion Engine - Phase 1

Complete production-grade academic paper discovery and ingestion system with two-layer search architecture.

## 🎯 Architecture Overview

### Two-Layer Search Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION ENGINE                          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         LAYER 1: Open-Source Discovery             │    │
│  │  (Parallel Search Across 5 APIs)                   │    │
│  │                                                     │    │
│  │  • OpenAlex      (250M+ OA works)                  │    │
│  │  • CORE          (200M+ OA papers)                 │    │
│  │  • Semantic S2   (200M+ papers)                    │    │
│  │  • DOAJ          (Verified OA journals)            │    │
│  │  • ARC-SIEVE     (Retraction checking)             │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│              Deduplication (DOI + Fuzzy Title)              │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │         LAYER 2: Closed-Access Enrichment          │    │
│  │  (Sequential Metadata + OA Enrichment)             │    │
│  │                                                     │    │
│  │  • CrossRef      (130M+ DOIs)                      │    │
│  │  • Unpaywall     (OA PDF enrichment)               │    │
│  │  • arXiv         (Preprints)                       │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│              Merge + Rank by Quality Metrics                │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │         PDF Download (Multi-Source Fallback)        │    │
│  │                                                     │    │
│  │  1. Direct OA URL  (OpenAlex/Unpaywall)            │    │
│  │  2. arXiv Mirror   (if arxiv_id present)           │    │
│  │  3. Sci-Hub        (if enabled + DOI)              │    │
│  │  4. LibGen         (if enabled + MD5)              │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│              Validation + Success Rate Check                │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │    Iterative Expansion (if needed)                  │    │
│  │                                                     │    │
│  │  • Broaden year range                              │    │
│  │  • Get recommendations (S2)                        │    │
│  │  • Try related queries                             │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📋 API Coverage

| API | Papers | Rate Limit | Search | OA Filter | PDF URLs | Citations |
|-----|--------|-----------|---------|-----------|----------|-----------|
| **OpenAlex** | 250M+ | 10 req/s | ✅ | ✅ | ✅ | ✅ |
| **CORE** | 200M+ | 100/min | ✅ | ✅ | ✅ | ❌ |
| **Semantic Scholar** | 200M+ | 1 req/s | ✅ | ❌ | ✅ | ✅ |
| **DOAJ** | Journals | 100/min | ✅ | ✅ (all) | ✅ | ❌ |
| **CrossRef** | 130M+ | 50 req/s | ✅ | ❌ | ❌ | ✅ |
| **Unpaywall** | DOI lookup | 100k/day | ❌ | ✅ | ✅ | ❌ |
| **arXiv** | Preprints | 3s delay | ✅ | ✅ (all) | ✅ | ❌ |
| **Sci-Hub** | Fallback | 3s delay | ❌ (DOI) | N/A | ✅ | ❌ |
| **LibGen** | Fallback | No limit | ❌ (MD5) | N/A | ✅ | ❌ |

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy config template
cp config/secrets_template.env .env

# Edit with your API keys
nano .env
```

### Basic Usage

```python
import asyncio
from paper-intel.ingestion import IngestionEngine

async def main():
    # Initialize engine
    engine = IngestionEngine(
        query="quantum computing applications",
        required_count=20,
        config_path="config/settings.yaml"
    )
    
    # Run complete pipeline
    results = await engine.run()
    
    # Check results
    print(f"Found: {len(results['papers'])} papers")
    print(f"Downloaded: {len(results['pdf_paths'])} PDFs")
    print(f"Sources used: {results['sources_used']}")

asyncio.run(main())
```

### Step-by-Step Example

```python
import asyncio
from paper-intel.ingestion import IngestionEngine

async def detailed_example():
    engine = IngestionEngine(
        query="machine learning healthcare",
        required_count=15,
        min_year=2020
    )
    
    # Step 1: Open-source discovery
    print("🔍 Searching open-source APIs...")
    open_papers = await engine.search_open_source()
    print(f"   Found {len(open_papers)} OA papers")
    
    # Step 2: Closed-access enrichment
    print("📚 Enriching from closed-access sources...")
    closed_papers = await engine.search_closed_access()
    print(f"   Found {len(closed_papers)} additional papers")
    
    # Step 3: Merge and deduplicate
    print("🔗 Merging and deduplicating...")
    all_papers = engine.merge_and_deduplicate(open_papers, closed_papers)
    print(f"   Total unique: {len(all_papers)} papers")
    
    # Step 4: Rank by quality
    print("📊 Ranking by citations/recency/OA...")
    ranked = engine.rank_and_filter(all_papers)
    print(f"   Top {len(ranked)} selected")
    
    # Step 5: Download PDFs
    print("⬇️  Downloading PDFs...")
    results = await engine.download_pdfs(ranked)
    print(f"   Success: {sum(1 for r in results if r.success)}/{len(results)}")
    
    return results

asyncio.run(detailed_example())
```

## ⚙️ Configuration

### settings.yaml

```yaml
apis:
  # API Keys (or use .env)
  s2_api_key: null
  core_api_key: null
  unpaywall_email: "your@email.com"
  crossref_email: "your@email.com"
  openalex_email: "your@email.com"
  
  # CORE can be slow/rate-limited
  enable_core: true

download:
  # Legal gray area - disabled by default
  enable_scihub: false
  enable_libgen: false
  
  # Output directory
  output_dir: "papers/"
  
  # Download settings
  max_retries: 3
  timeout: 60
  parallel_downloads: 5

search:
  # Year filtering
  min_year: 2015
  max_year: null  # Current year
  
  # Preferences
  prefer_open_access: true
  limit_per_source: 20
  
  # Iteration settings
  max_iterations: 3
  min_success_rate: 0.7  # 70% PDFs must download

grobid:
  enabled: false  # Requires GROBID server
  server_url: "http://localhost:8070"
  timeout: 300
  consolidate_citations: true

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/ingestion.log"
```

### .env File

```bash
# Semantic Scholar (optional but recommended)
SEMANTIC_SCHOLAR_API_KEY=your_s2_key_here

# CORE API (optional, can disable in settings.yaml)
CORE_API_KEY=your_core_key_here

# Email addresses (required for polite pool rate limits)
UNPAYWALL_EMAIL=your@email.com
CROSSREF_EMAIL=your@email.com
OPENALEX_EMAIL=your@email.com

# Sci-Hub/LibGen (disabled by default)
ENABLE_SCIHUB=false
ENABLE_LIBGEN=false

# Optional: GROBID server URL
GROBID_URL=http://localhost:8070

# Output directory
OUTPUT_DIR=papers/
```

## 📥 PDF Download Workflow

```
Paper Metadata (DOI, arXiv ID, PDF URLs)
    ↓
┌───────────────────────────────────────────┐
│  1. Try Direct OA URL                     │
│     - OpenAlex best_oa_location           │
│     - Unpaywall oa_locations              │
│     - DOAJ PDF links                      │
└───────────────────────────────────────────┘
    ↓ (if failed)
┌───────────────────────────────────────────┐
│  2. Try arXiv Mirror                      │
│     - arxiv.org/pdf/{id}.pdf              │
└───────────────────────────────────────────┘
    ↓ (if failed)
┌───────────────────────────────────────────┐
│  3. Try Sci-Hub (if enabled)              │
│     - Rotate 6 mirrors                    │
│     - Parse HTML for PDF link             │
└───────────────────────────────────────────┘
    ↓ (if failed)
┌───────────────────────────────────────────┐
│  4. Try LibGen (if enabled + MD5)         │
│     - Anna's Archive lookup               │
│     - 4 LibGen mirrors                    │
└───────────────────────────────────────────┘
    ↓
Validate PDF (magic bytes, size > 1KB)
    ↓
Save to: papers/{doi_hash}_{authors}_{year}.pdf
```

## 🔍 Quality Ranking Algorithm

Papers are scored using multiple factors:

```python
score = (
    log10(citations + 1) * 2.0    # Log-scale to prevent outliers
    + recency_bonus                # 1.0 for 2024, decays 0.1/year
    + oa_bonus                     # 0.5 if open access
    + source_score * 0.5           # OpenAlex=1.0, CORE=0.8, etc.
)
```

**Why log-scale?** A paper with 10,000 citations shouldn't dominate 100 good papers with 100 citations each.

## 🔄 Iterative Expansion

If initial search yields < 70% success rate, the engine tries:

1. **Broaden Year Range**: `min_year - 5` years
2. **Get Recommendations**: Semantic Scholar related papers from top 3 results
3. **Try Related Queries**: Add "review", "survey", "applications", "methods"

Maximum 3 iterations before giving up.

## 🛡️ DOI Validation

Rejects papers from known predatory publishers:

- `10.21275` (IJSR - fake journals)
- `10.2139` (SSRN - working papers, not peer-reviewed)
- `10.32388` (IJARIIE)
- `10.65215` (IJEAST)
- `10.37074` (IJCRT)
- `10.5281` (Zenodo - unreliable DOIs)

## 📊 Deduplication

Two-pass deduplication:

1. **Exact DOI match**: Same normalized DOI = duplicate
2. **Fuzzy title match**: 90%+ similarity (token_set_ratio) = duplicate

When merging duplicates:
- Prefer papers with OA PDF URLs
- Combine all identifiers (DOI, arXiv, PMID, etc.)
- Keep highest citation count

## 🚦 Rate Limiting

All clients implement polite rate limiting:

| Client | Default Delay | Adjustable |
|--------|--------------|-----------|
| OpenAlex | 0.1s (10/s) | ✅ |
| CORE | 0.6s (100/min) | ✅ |
| SemanticScholar | 1.0s | ✅ |
| DOAJ | 0.6s | ✅ |
| CrossRef | 0.02s (50/s) | ✅ |
| Unpaywall | 0.001s | ✅ |
| arXiv | 3.0s (mandatory) | ❌ |
| Sci-Hub | 3.0s | ✅ |

429 (rate limit) responses trigger exponential backoff: 2^attempt seconds.

## 📁 Folder Structure

```
paper-intel/
├── ingestion/
│   ├── models.py                    # Data models
│   ├── ingestion_engine.py          # Main orchestrator
│   ├── discovery/
│   │   ├── base_client.py           # Abstract API client
│   │   ├── doi_utils.py             # DOI utilities
│   │   ├── openalex_client.py       # OpenAlex
│   │   ├── core_client.py           # CORE
│   │   ├── semanticscholar_client.py # S2
│   │   ├── doaj_client.py           # DOAJ
│   │   ├── arcsieve_client.py       # ARC-SIEVE
│   │   ├── crossref_client.py       # CrossRef
│   │   ├── unpaywall_client.py      # Unpaywall
│   │   ├── arxiv_client.py          # arXiv
│   │   ├── scihub_client.py         # Sci-Hub
│   │   ├── annasarchive_client.py   # LibGen
│   │   ├── search_engine.py         # Multi-source orchestrator
│   │   └── pdf_downloader.py        # Download manager
│   └── preprocessing/
│       ├── grobid_client.py         # GROBID integration
│       ├── pdf_to_text.py           # Text extraction
│       └── text_cleaner.py          # Text cleaning
├── config/
│   ├── settings.yaml                # Main config
│   └── secrets_template.env         # Environment template
└── papers/                          # Downloaded PDFs
```

## 🧪 Testing

### Test Individual Components

```python
# Test OpenAlex client
from paper-intel.ingestion.discovery import OpenAlexClient

client = OpenAlexClient(email="your@email.com")
results = client.search("deep learning", limit=5)
print(f"Found {len(results)} papers")

# Test PDF downloader
from paper-intel.ingestion.discovery import PDFDownloader
from paper-intel.ingestion.models import PaperMetadata

downloader = PDFDownloader(output_dir="test_papers/")
paper = PaperMetadata(
    doi="10.1038/nature14539",
    title="Deep learning",
    pdf_urls=["https://..."]
)
result = downloader.download_paper(paper)
print(f"Success: {result.success}, Path: {result.pdf_path}")
```

### Test Search Engine

```python
from paper-intel.ingestion.discovery import SearchEngine

engine = SearchEngine(config_path="config/settings.yaml")

# Test open-source layer
papers = engine.search_open_source("reinforcement learning", limit=10)
print(f"OA papers: {len(papers)}")

# Test closed-access layer
papers = engine.search_closed_access("neural networks", limit=10)
print(f"Closed papers: {len(papers)}")

# Test unified search
papers = engine.search_unified("computer vision", limit=20)
print(f"Total papers: {len(papers)}")
```

## 🐛 Troubleshooting

### CORE API 429 Rate Limit

**Solution 1**: Disable CORE
```yaml
apis:
  enable_core: false
```

**Solution 2**: Get API key from [core.ac.uk](https://core.ac.uk/services/api)
```bash
CORE_API_KEY=your_key_here
```

### DOAJ Search Not Working

The system now uses proper Lucene query syntax:
```python
# ✅ Correct
query = "machine learning AND bibjson.year:>=2020"

# ❌ Wrong (old implementation)
query = "?q=machine learning"
```

### Sci-Hub/LibGen Not Working

These are **disabled by default** for legal reasons. Enable only if necessary:

```bash
ENABLE_SCIHUB=true
ENABLE_LIBGEN=true
```

**Warning**: Sci-Hub violates copyright in many jurisdictions. Use only after legal sources exhausted.

### PDF Download Failures

Check logs for details:
```python
logging.basicConfig(level=logging.DEBUG)
```

Common issues:
- **403 Forbidden**: Publisher blocks automated downloads
- **404 Not Found**: Broken PDF URL
- **Timeout**: Large file or slow connection (increase `timeout` in settings)

## 📈 Performance Metrics

Typical performance (20 papers, 2020-2024):

| Metric | Value |
|--------|-------|
| **Open-source search** | 2-5 seconds |
| **Closed-access search** | 3-8 seconds |
| **Total papers found** | 50-100 candidates |
| **After deduplication** | 30-60 unique |
| **PDF download time** | 15-30 seconds (parallel) |
| **Success rate** | 75-90% (OA preferred) |

## 🔐 API Key Setup

### Semantic Scholar (Recommended)

1. Visit [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
2. Sign up for free API key (1 req/s, no limit)
3. Add to `.env`: `SEMANTIC_SCHOLAR_API_KEY=your_key`

### CORE (Optional)

1. Visit [core.ac.uk/services/api](https://core.ac.uk/services/api)
2. Register for API key (100 req/min free tier)
3. Add to `.env`: `CORE_API_KEY=your_key`

### Polite Pool (Required)

For OpenAlex, CrossRef, Unpaywall:
- Add your email to `.env`
- No registration needed
- Gets higher rate limits (10x faster)

## 📄 Output Format

Results are saved as JSON:

```json
{
  "query": "quantum computing",
  "papers": [
    {
      "doi": "10.1038/s41586-019-1666-5",
      "title": "Quantum supremacy using a programmable superconducting processor",
      "authors": ["Arute, Frank", "..."],
      "year": 2019,
      "citations": 5234,
      "is_open_access": true,
      "pdf_url": "https://...",
      "source": "openalex"
    }
  ],
  "pdf_paths": [
    "papers/10_1038_s41586-019-1666-5_Arute_2019.pdf"
  ],
  "sources_used": ["openalex", "core", "semanticscholar", "crossref"],
  "total_found": 47,
  "downloaded": 18,
  "success_rate": 0.90,
  "elapsed_time": 42.3
}
```

## 🚀 Advanced Usage

### Custom Ranking Function

```python
from paper-intel.ingestion import IngestionEngine
from paper-intel.ingestion.models import PaperMetadata

def my_ranker(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    # Custom logic: prefer recent papers with keywords
    keywords = {"transformer", "attention", "bert"}
    
    def score(paper):
        base = paper.citations or 0
        recency = 2024 - paper.year if paper.year else 0
        keyword_match = sum(1 for kw in keywords if kw in paper.title.lower())
        return base * 0.5 + recency * 10 + keyword_match * 50
    
    return sorted(papers, key=score, reverse=True)

engine = IngestionEngine(
    query="natural language processing",
    required_count=20
)

# Override default ranker
engine.rank_and_filter = lambda papers: my_ranker(papers)[:20]

results = await engine.run()
```

### Batch Processing

```python
import asyncio
from paper-intel.ingestion import IngestionEngine

async def batch_ingest(queries: list[str], papers_per_query: int = 10):
    results = {}
    
    for query in queries:
        print(f"\n🔍 Processing: {query}")
        engine = IngestionEngine(query, papers_per_query)
        results[query] = await engine.run()
    
    return results

queries = [
    "deep learning healthcare",
    "quantum computing algorithms",
    "renewable energy optimization"
]

results = asyncio.run(batch_ingest(queries))
```

## 🤝 Contributing

Areas for improvement:

1. **More APIs**: PubMed, Europe PMC, Microsoft Academic Graph
2. **Better deduplication**: Use embeddings for semantic similarity
3. **Citation graphs**: Build relationships between papers
4. **Full-text search**: Index extracted text with Elasticsearch
5. **GROBID integration**: Extract structured data from PDFs

## 📚 References

- OpenAlex API: [openalex.org/api](https://openalex.org/api)
- CORE API: [core.ac.uk/docs](https://core.ac.uk/docs)
- Semantic Scholar: [semanticscholar.org/api](https://www.semanticscholar.org/product/api)
- DOAJ: [doaj.org/api](https://doaj.org/api/v3)
- CrossRef: [crossref.org/documentation](https://www.crossref.org/documentation/retrieve-metadata/)
- Unpaywall: [unpaywall.org/products/api](https://unpaywall.org/products/api)
- arXiv: [arxiv.org/help/api](https://arxiv.org/help/api)

## ⚖️ Legal Notice

This tool is designed for **academic research** and respects publisher rights:

- ✅ Prioritizes open-access sources (OpenAlex, DOAJ, arXiv)
- ✅ Uses Unpaywall for legal OA discovery
- ✅ Respects rate limits and robots.txt
- ⚠️ Sci-Hub/LibGen disabled by default (enable at your own risk)

**User Responsibility**: Ensure your usage complies with local laws and institutional policies.

## 📞 Support

Issues? Check:
1. Logs: `logs/ingestion.log`
2. Debug mode: `logging.level: DEBUG` in settings.yaml
3. Test individual clients (see Testing section)

---

**Built with ❤️ for Sanshodhak** | Phase 1: Paper Discovery & Ingestion
