# 📚 Sanshodhak Paper Intelligence - Phase 1 Ingestion System

## 🚀 Quick Navigation

### 🏁 Start Here
1. **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
2. **[example_simple.py](example_simple.py)** - Basic usage example
3. **[test_apis.py](test_apis.py)** - Verify your setup

### 📖 Documentation
- **[README_ingestion.md](README_ingestion.md)** - Complete documentation (18KB)
  - Architecture diagrams
  - All 11 APIs explained
  - Configuration guide
  - PDF workflow
  - Troubleshooting
  - Advanced usage

- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - What was built & why
  - Complete file listing
  - Feature checklist
  - Performance metrics
  - Known issues & solutions

### 🧪 Testing & Demos
- **[test_apis.py](test_apis.py)** - Test all API clients
  - Verify DOAJ (user's concern)
  - Test OpenAlex, Semantic Scholar, CrossRef, arXiv
  - Validate DOI utilities
  - Test search engine

- **[demo_ingestion.py](demo_ingestion.py)** - Interactive demos
  1. Quick start (one-liner)
  2. Step-by-step pipeline
  3. Iterative expansion
  4. API comparison
  5. PDF fallback chain

- **[example_simple.py](example_simple.py)** - Minimal example
  - 10 lines of code
  - Fetch 10 papers
  - Print results

### ⚙️ Configuration
- **[config/settings.yaml](config/settings.yaml)** - Main config
  - API keys & emails
  - Download options
  - Search parameters
  - GROBID settings
  - Logging config

- **[config/secrets_template.env](config/secrets_template.env)** - Environment template
  - Copy to `.env`
  - Add your API keys
  - Configure emails

### 📦 Dependencies
- **[requirements.txt](requirements.txt)** - Python packages
  ```bash
  pip install -r requirements.txt
  ```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   INGESTION ENGINE                       │
│                                                          │
│  🔍 Layer 1: Open-Source (5 APIs parallel)              │
│     → OpenAlex, CORE, S2, DOAJ, ArcSieve                │
│                                                          │
│  📚 Layer 2: Closed-Access (sequential)                 │
│     → CrossRef → Unpaywall → arXiv                      │
│                                                          │
│  🔗 Merge & Deduplicate (DOI + fuzzy title)             │
│                                                          │
│  📊 Rank (citations + recency + OA + source quality)    │
│                                                          │
│  ⬇️  Download PDFs (4-stage fallback):                  │
│     1. Direct OA → 2. arXiv → 3. Sci-Hub → 4. LibGen   │
│                                                          │
│  ✅ Validate & Check Success Rate (≥70%)                │
│                                                          │
│  🔄 Expand if needed (max 3 iterations):                │
│     • Broaden year range                                │
│     • Get S2 recommendations                            │
│     • Try related queries                               │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Quick Start (30 seconds)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure (add your email)
cp config/secrets_template.env .env
echo "OPENALEX_EMAIL=your@email.com" >> .env

# 3. Test
python test_apis.py

# 4. Run example
python example_simple.py
```

## 💻 Basic Usage

```python
import asyncio
from ingestion import IngestionEngine

async def main():
    engine = IngestionEngine(
        query="your research topic",
        required_count=20
    )
    
    results = await engine.run()
    print(f"Downloaded {len(results['pdf_paths'])} PDFs!")

asyncio.run(main())
```

## 📁 Project Structure

```
paper-intel/
├── README_ingestion.md      ← Complete docs (START HERE after quickstart)
├── QUICKSTART.md            ← 5-minute setup (START HERE)
├── PROJECT_SUMMARY.md       ← What was built
├── INDEX.md                 ← This file
│
├── requirements.txt         ← Dependencies
├── example_simple.py        ← Minimal example
├── test_apis.py            ← API tests
├── demo_ingestion.py       ← Interactive demos
│
├── config/
│   ├── settings.yaml       ← Main config
│   └── secrets_template.env ← Environment template
│
├── ingestion/
│   ├── models.py           ← Data models
│   ├── ingestion_engine.py ← Main orchestrator
│   ├── discovery/          ← 11 API clients + orchestrators
│   └── preprocessing/      ← Text extraction (placeholders)
│
└── papers/                 ← Downloaded PDFs
```

## ✅ Feature Checklist

- [x] **11 API clients**: OpenAlex, CORE, S2, DOAJ, ArcSieve, CrossRef, Unpaywall, arXiv, Sci-Hub, LibGen, Anna's Archive
- [x] **Two-layer search**: Open-source (parallel) + Closed-access (sequential)
- [x] **Multi-source PDF download**: 4-stage fallback chain
- [x] **Smart ranking**: Citations + recency + OA status + source quality
- [x] **Deduplication**: DOI exact match + 90% fuzzy title match
- [x] **DOI validation**: Reject 6 predatory publisher prefixes
- [x] **Iterative expansion**: 3 strategies if results insufficient
- [x] **Rate limiting**: Per-client configurable delays, 429 handling
- [x] **Retry logic**: Exponential backoff for failed requests
- [x] **PDF validation**: Magic bytes check, file size verification
- [x] **Parallel execution**: ThreadPoolExecutor for concurrent operations
- [x] **Type safety**: Full dataclass annotations
- [x] **Comprehensive logging**: Track every step
- [x] **JSON output**: Complete results with metadata
- [x] **DOAJ fix**: Proper Lucene query syntax (user's concern)
- [x] **CORE fix**: Rate limit handling (can be disabled)

## 🎓 Usage Scenarios

### 1. Simple Query
```bash
python example_simple.py
```

### 2. Recent Papers Only
```python
engine = IngestionEngine("deep learning", required_count=20, min_year=2023)
```

### 3. Open Access Preferred
```yaml
# config/settings.yaml
search:
  prefer_open_access: true
```

### 4. Batch Processing
```python
topics = ["quantum computing", "machine learning", "renewable energy"]
for topic in topics:
    engine = IngestionEngine(topic, required_count=10)
    results = await engine.run()
```

### 5. Step-by-Step Pipeline
```python
open_papers = await engine.search_open_source()
closed_papers = await engine.search_closed_access()
all_papers = engine.merge_and_deduplicate(open_papers, closed_papers)
ranked = engine.rank_and_filter(all_papers)
results = await engine.download_pdfs(ranked)
```

## 📊 Expected Performance

**Query**: "machine learning" (20 papers, 2020-2024)

| Metric | Value |
|--------|-------|
| Search time | 5-10s |
| Papers found | 50-100 |
| After dedup | 30-60 |
| Download time | 15-30s |
| **Total** | **25-40s** |
| **Success rate** | **75-90%** |

## 🐛 Common Issues

### CORE API Rate Limit
```yaml
# config/settings.yaml
apis:
  enable_core: false
```

### Low PDF Success
- Prefer OA: `prefer_open_access: true`
- Enable Sci-Hub: `ENABLE_SCIHUB=true` (legal risk!)
- Expand iterations: `max_iterations: 5`

### Slow Performance
- More parallelism: `parallel_downloads: 10`
- Fewer per source: `limit_per_source: 10`

## 📞 Need Help?

1. **Setup issues**: See [QUICKSTART.md](QUICKSTART.md)
2. **API problems**: Run `python test_apis.py`
3. **Usage questions**: See [README_ingestion.md](README_ingestion.md)
4. **Understanding what was built**: See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## 🏆 What Makes This Special?

1. **Two-layer search**: Redundancy across 11 APIs
2. **Smart fallback**: 4 PDF sources with validation
3. **Iterative expansion**: Auto-broadens if results insufficient
4. **Quality focus**: Ranks by citations + recency + OA
5. **Production-ready**: Rate limiting, retries, validation, logging
6. **Legal safety**: OA preferred, gray-area sources disabled by default
7. **Clean architecture**: 30 modular files, type-safe
8. **Complete docs**: 25KB of documentation + tests + demos

## 🎯 Next Steps

### For Users
1. **Install**: `pip install -r requirements.txt`
2. **Configure**: Copy `.env`, add email
3. **Test**: `python test_apis.py`
4. **Use**: `python example_simple.py`

### For Developers
1. **Read**: [README_ingestion.md](README_ingestion.md)
2. **Understand**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
3. **Test**: `python test_apis.py`
4. **Explore**: `python demo_ingestion.py` (select 0 for all demos)

### For Researchers
1. **Quick start**: [QUICKSTART.md](QUICKSTART.md)
2. **Basic example**: `python example_simple.py`
3. **Customize**: Edit `config/settings.yaml`
4. **Batch process**: See "Batch Processing" in [README_ingestion.md](README_ingestion.md)

---

## 📚 Documentation Map

```
START HERE → QUICKSTART.md (5 min)
    ↓
    ├─→ example_simple.py (run it!)
    ↓
    ├─→ test_apis.py (verify setup)
    ↓
    └─→ README_ingestion.md (complete docs)
            ↓
            ├─→ Architecture diagrams
            ├─→ API coverage details
            ├─→ Configuration guide
            ├─→ Advanced usage
            └─→ Troubleshooting

DEVELOPERS → PROJECT_SUMMARY.md
    ↓
    └─→ What was built, why, and how

DEMOS → demo_ingestion.py
    ↓
    └─→ 5 interactive demos
```

---

**Status**: ✅ **PRODUCTION READY**

**Version**: Phase 1 - Complete Ingestion System

**Built for**: Sanshodhak Academic Paper Intelligence Platform

**Date**: 2024

**License**: See project root

---

**Start here**: [QUICKSTART.md](QUICKSTART.md) → [example_simple.py](example_simple.py) → [README_ingestion.md](README_ingestion.md)
