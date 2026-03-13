# Quick Start Guide - Sanshodhak Ingestion Engine

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd paper-intel
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

```bash
# Copy template
cp config/secrets_template.env .env

# Edit with your keys
nano .env
```

**Minimum required**: Your email addresses for polite API pools
```bash
UNPAYWALL_EMAIL=your@email.com
CROSSREF_EMAIL=your@email.com
OPENALEX_EMAIL=your@email.com
```

**Recommended**: Semantic Scholar API key (free, 1 req/s)
```bash
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

### 3. Test Your Setup

```bash
# Run API tests
python test_apis.py
```

Expected output:
```
✅ DOAJ working! Found 3 papers
✅ OpenAlex working! Found 3 papers
✅ Semantic Scholar working! Found 3 papers
...
```

### 4. Run First Demo

```bash
# Interactive demo
python demo_ingestion.py
# Select: 1 (Quick Start)
```

### 5. Use in Your Code

```python
import asyncio
from ingestion import IngestionEngine

async def main():
    engine = IngestionEngine(
        query="your research topic here",
        required_count=20
    )
    
    results = await engine.run()
    print(f"Downloaded {len(results['pdf_paths'])} PDFs!")

asyncio.run(main())
```

## 🎯 Common Use Cases

### Find Recent Papers (2023+)

```python
engine = IngestionEngine(
    query="large language models",
    required_count=30,
    min_year=2023
)
```

### Prefer Open Access Only

```python
# Edit config/settings.yaml
search:
  prefer_open_access: true
  
# Or filter after search
results = await engine.run()
oa_papers = [p for p in results['papers'] if p['is_open_access']]
```

### Batch Download for Multiple Topics

```python
async def batch_ingest(topics):
    for topic in topics:
        print(f"Processing: {topic}")
        engine = IngestionEngine(topic, required_count=10)
        await engine.run()

topics = ["quantum computing", "deep learning", "renewable energy"]
asyncio.run(batch_ingest(topics))
```

## 🔧 Troubleshooting

### CORE API Rate Limit (429)

```yaml
# config/settings.yaml
apis:
  enable_core: false  # Disable CORE
```

### Slow Downloads

```yaml
# config/settings.yaml
download:
  parallel_downloads: 10  # Increase from 5
  timeout: 120           # Increase timeout
```

### No PDFs Downloaded

1. Check if papers are open access:
   ```python
   oa_count = sum(1 for p in results['papers'] if p['is_open_access'])
   print(f"OA papers: {oa_count}/{len(results['papers'])}")
   ```

2. Enable Sci-Hub (legal gray area):
   ```bash
   # .env
   ENABLE_SCIHUB=true
   ```

## 📊 Understanding Results

```python
results = await engine.run()

# What's included:
results['papers']          # List of paper metadata
results['pdf_paths']       # List of downloaded PDF paths
results['sources_used']    # Which APIs were queried
results['success_rate']    # % of PDFs downloaded
results['elapsed_time']    # Total time in seconds
results['iterations']      # How many search expansions
```

## 📚 Next Steps

1. **Read full docs**: `README_ingestion.md`
2. **Try all demos**: `python demo_ingestion.py` → select 0
3. **Customize ranking**: See "Advanced Usage" in README
4. **Add preprocessing**: Enable GROBID for text extraction

## 💡 Tips

- **Start small**: Try `required_count=5` first to test quickly
- **Check logs**: `logs/ingestion.log` has detailed info
- **Test APIs individually**: Use `test_apis.py` to debug
- **Use polite pools**: Add your email to get 10x faster rate limits
- **Prefer OA sources**: 75-90% success rate vs 30-40% for closed access

## 🆘 Getting Help

1. Run tests: `python test_apis.py`
2. Enable debug logging: `logging.level: DEBUG` in `config/settings.yaml`
3. Check specific client:
   ```python
   from ingestion.discovery import OpenAlexClient
   client = OpenAlexClient(email="your@email.com")
   papers = client.search("test query", limit=1)
   print(papers)
   ```

---

**Ready to ingest papers? Start with `test_apis.py` to verify your setup!**
