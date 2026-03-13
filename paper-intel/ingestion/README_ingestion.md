# Phase 1: PDF → TEXT → STRUCTURED JSON

Complete pipeline to convert research PDFs into canonical JSON format for knowledge graph construction.

## Architecture

```
PDF → GROBID → TEI XML → Parser → Cleaner → Splitter → JSON + TXT
```

### Components

1. **GrobidClient** (`grobid_client.py`)
   - Async/sync GROBID REST API client
   - Connection pooling with httpx
   - Exponential backoff retry logic
   - Batch processing support

2. **TEIParser** (`tei_parser.py`)
   - Parse GROBID TEI XML output
   - Extract: title, authors, abstract, body, references, metadata
   - Namespace-aware XML handling

3. **TextCleaner** (`text_cleaner.py`)
   - Unicode normalization (ligatures, smart quotes)
   - Hyphenation fixing (line-break removal)
   - Header/footer removal (page numbers, copyright)
   - Whitespace normalization

4. **SectionSplitter** (`section_splitter.py`)
   - Detect paper sections with regex patterns
   - Map to 7 canonical sections:
     - introduction
     - related_work
     - methods
     - experiments
     - results
     - discussion
     - conclusion

5. **PDFToTextPipeline** (`pdf_to_text.py`)
   - Main orchestrator
   - Parallel processing with semaphores
   - Error handling and logging

## Setup

### 1. Start GROBID Server

```bash
docker pull lfoppiano/grobid:0.8.0
docker run -p 8070:8070 lfoppiano/grobid:0.8.0
```

Wait ~30 seconds for startup. Check health:
```bash
curl http://localhost:8070/api/isalive
```

### 2. Install Dependencies

```bash
cd paper-intel/ingestion
pip install httpx tenacity pyyaml
```

### 3. Configure

Edit `settings.yaml`:
```yaml
grobid_url: "http://localhost:8070"
workers: 4
```

## Usage

### CLI

Process a directory of PDFs:

```bash
python -m ingestion.pdf_to_text /path/to/pdfs/ \
  --output-dir ingestion \
  --grobid-url http://localhost:8070 \
  --workers 4 \
  --log-level INFO
```

### Python API

```python
import asyncio
from pathlib import Path
from ingestion.pdf_to_text import PDFToTextPipeline

async def process():
    pipeline = PDFToTextPipeline(
        grobid_url="http://localhost:8070",
        output_dir=Path("ingestion"),
        workers=4
    )
    
    pdf_paths = list(Path("downloads/").glob("*.pdf"))
    stats = await pipeline.process_batch(pdf_paths)
    
    print(f"Success: {stats['success']}/{stats['total']}")
    
    await pipeline.close()

asyncio.run(process())
```

## Output Format

### JSON Schema

```json
{
  "paper_id": "deep_learning_med_12abc456",
  "source_pdf": "paper.pdf",
  "title": "Deep Learning for Medical Imaging",
  "authors": ["John Smith", "Jane Doe"],
  "abstract": "This paper presents...",
  "sections": {
    "introduction": "Deep learning has transformed...",
    "related_work": "Previous work includes...",
    "methods": "We propose a novel architecture...",
    "experiments": "We evaluated on three datasets...",
    "results": "Our method achieved 95% accuracy...",
    "discussion": "The results demonstrate...",
    "conclusion": "We presented a novel approach..."
  },
  "references": [
    {
      "title": "Attention Is All You Need",
      "authors": ["Vaswani et al."],
      "year": "2017",
      "venue": "NeurIPS"
    }
  ],
  "metadata": {
    "year": "2023",
    "doi": "10.1234/example",
    "venue": "Nature Medicine",
    "processed_at": "2024-01-15T10:30:00"
  }
}
```

### Output Directories

```
ingestion/
├── output_json/          # Structured JSON files
│   ├── paper1_hash.json
│   └── paper2_hash.json
├── raw_text/            # Clean full text
│   ├── paper1_hash.txt
│   └── paper2_hash.txt
└── errors/              # Error logs
    └── paper3_hash.log
```

## Pipeline Steps (Per PDF)

1. **GROBID Processing**
   - Upload PDF to GROBID `/api/processFulltextDocument`
   - Receive TEI XML output
   - Retry on failure (3 attempts with exponential backoff)

2. **TEI Parsing**
   - Parse XML with namespace awareness
   - Extract metadata (title, authors, abstract, year, DOI, venue)
   - Extract full body text
   - Extract references

3. **Text Cleaning**
   - Unicode normalization (NFC)
   - Convert ligatures (ﬁ → fi, ﬂ → fl)
   - Fix hyphenated line breaks (compu-\nter → computer)
   - Remove headers/footers (page numbers, copyright)
   - Normalize whitespace

4. **Section Splitting**
   - Detect section headings with regex
   - Map to canonical sections (30+ variations supported)
   - Confidence scoring based on position, numbering, formatting
   - Fallback: Full text → introduction if no sections found

5. **JSON Creation**
   - Assemble structured output
   - Generate unique paper_id (filename + hash)
   - Add metadata timestamp

6. **Save Outputs**
   - Write JSON to `output_json/<paper_id>.json`
   - Write raw text to `raw_text/<paper_id>.txt`
   - Write errors to `errors/<paper_id>.log` (if failed)

## Performance

- **Throughput**: ~1-2 PDFs/second (GROBID bottleneck)
- **Concurrency**: 4 workers recommended
- **Memory**: ~200 MB per worker
- **GROBID**: Docker container uses ~2 GB RAM

## Error Handling

Common errors:

1. **GROBID not available**
   ```
   ❌ GROBID server not available at http://localhost:8070
   ```
   → Start GROBID Docker container

2. **Timeout**
   ```
   GROBID failed: Timeout after 120s
   ```
   → Increase `grobid_timeout` in settings.yaml

3. **No title extracted**
   ```
   ⚠️  No title extracted for paper_hash
   ```
   → PDF may be scanned/image-based (GROBID cannot OCR)

4. **Empty sections**
   ```
   sections: {"introduction": "", "methods": "", ...}
   ```
   → Section detection failed, full text in `raw_text/`

## Debugging

### Enable debug logging

```bash
python -m ingestion.pdf_to_text /path/to/pdfs/ --log-level DEBUG
```

### Check GROBID directly

```bash
curl -X POST \
  -F "input=@paper.pdf" \
  http://localhost:8070/api/processFulltextDocument
```

### Inspect intermediate outputs

Set `save_tei_xml: true` in settings.yaml to keep TEI XML files.

## Integration with Phase 0

Chain with paper collection:

```bash
# Phase 0: Collect papers
python max_orchestrator.py "deep learning medical imaging" --target 15

# Phase 1: Process PDFs
python -m ingestion.pdf_to_text downloads/deep_learning_medical_imaging/
```

## Next Steps: Phase 2+

Phase 1 outputs feed into:

- **Phase 2**: Entity extraction (authors, institutions, methods, datasets)
- **Phase 3**: Knowledge graph construction
- **Phase 4**: GraphRAG deployment

Each phase consumes the structured JSON from `output_json/`.

## Troubleshooting

### Docker on macOS/Windows

```bash
# Ensure Docker Desktop is running
docker ps

# If port conflicts:
docker run -p 8071:8070 lfoppiano/grobid:0.8.0
# Update grobid_url to http://localhost:8071
```

### Memory Issues

Reduce workers:
```bash
python -m ingestion.pdf_to_text /path/to/pdfs/ --workers 2
```

### Slow Processing

GROBID is CPU-bound. Options:
- Use faster machine
- Reduce `grobid_timeout` to fail faster on stuck PDFs
- Skip large/complex PDFs

## Examples

### Process 100 PDFs

```bash
python -m ingestion.pdf_to_text downloads/my_papers/ --workers 4
```

Expected output:
```
2024-01-15 10:30:00 - INFO - Found 100 PDFs
2024-01-15 10:30:01 - INFO - ✅ GROBID server is healthy
2024-01-15 10:30:05 - INFO - ✅ Success: paper1_abc
2024-01-15 10:30:08 - INFO - ✅ Success: paper2_def
...
================================================================================
PHASE 1 COMPLETE
================================================================================
Total PDFs:    100
✅ Success:    97
❌ Failed:     3
Duration:      180.5s
Rate:          0.55 PDFs/sec

Outputs:
  JSON:        ingestion/output_json
  Raw text:    ingestion/raw_text
  Errors:      ingestion/errors
================================================================================
```

### Single PDF (Python)

```python
import asyncio
from pathlib import Path
from ingestion.pdf_to_text import PDFToTextPipeline

async def process_one():
    pipeline = PDFToTextPipeline()
    
    success, paper_id, error = await pipeline.process_single_pdf(
        Path("paper.pdf")
    )
    
    if success:
        print(f"✅ {paper_id}")
        json_path = pipeline.json_dir / f"{paper_id}.json"
        print(f"Output: {json_path}")
    else:
        print(f"❌ {error}")
    
    await pipeline.close()

asyncio.run(process_one())
```

## License

Part of Sanshodhak Research Intelligence System.
