# Phase 2: Quick Reference

## Installation

```bash
pip install httpx pyyaml
cp config/secrets_template.env config/.env
nano config/.env  # Add GRAFFITI_API_KEY and OPENROUTER_API_KEY
```

## Basic Usage

```bash
# Extract all papers
python -m extraction.extraction_engine

# Extract single paper
python -m extraction.extraction_engine --paper-id paper123

# Test on sample
python test_phase2.py
```

## Configuration

**config/settings.yaml:**
```yaml
graffiti:
  base_url: "https://api.graffiti.ai/extract"
  max_chunk_size: 3000
  max_retries: 3

llm:
  model: "qwen/qwen-2.5-32b-instruct"
  temperature: 0.1

extraction:
  use_graffiti: true
  use_llm_fallback: true
  workers: 4
```

## Entity Types

- **Model**: BERT, GPT, ResNet
- **Method**: Attention, dropout, SGD
- **Task**: Classification, translation
- **Dataset**: ImageNet, GLUE
- **Metric**: Accuracy, F1-score
- **Result**: 92.4% accuracy
- **Paper**: Referenced works
- **Author**: Researchers

## Relation Types

- **USES_MODEL**: Uses a model
- **USES_METHOD**: Uses a technique
- **USES_DATASET**: Uses dataset
- **EVALUATES_ON**: Tests on benchmark
- **REPORTS_METRIC**: Reports metric
- **ACHIEVES_RESULT**: Gets performance
- **COMPARES_WITH**: Compares baseline
- **CITES**: References paper
- **AUTHORED_BY**: Written by
- **STUDIES_TASK**: Addresses problem

## Outputs

```
extraction/
  output_entities/<paper_id>.json    # Entities with IDs
  output_relations/<paper_id>.json   # Relations with IDs
  output_triples/<paper_id>.json     # Triples + metadata
```

## CLI Options

```bash
--json-dir PATH          # Phase 1 JSON directory
--text-dir PATH          # Phase 1 text directory
--output-dir PATH        # Output directory
--workers N              # Parallel workers (default: 4)
--paper-id ID            # Extract single paper
--no-graffiti            # Disable Graffiti
--no-llm                 # Disable LLM fallback
--log-level LEVEL        # DEBUG, INFO, WARNING, ERROR
```

## Example Output

**Entities:**
```json
[
  {"id": "E_paper123_1", "type": "Model", "name": "BERT"},
  {"id": "E_paper123_2", "type": "Dataset", "name": "GLUE"}
]
```

**Relations:**
```json
[
  {
    "source": "E_paper123_1",
    "type": "EVALUATES_ON",
    "target": "E_paper123_2"
  }
]
```

**Triples:**
```json
{
  "triples": [
    ["BERT", "EVALUATES_ON", "GLUE"],
    ["BERT", "REPORTS_METRIC", "Accuracy"]
  ]
}
```

## Programmatic Usage

```python
from extraction.extraction_engine import ExtractionEngine

async with ExtractionEngine() as engine:
    # Extract single paper
    success, error = await engine.extract_single('paper123')
    
    # Extract all papers
    stats = await engine.extract_all()
    print(f"Success: {stats['success']}/{stats['total']}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No API key | Add to config/.env |
| No papers found | Run Phase 1 first |
| Rate limited | Reduce workers |
| Memory issues | Process in batches |
| Low quality | Check text quality |

## Performance

- **Speed**: ~4 papers/minute (4 workers)
- **Success**: ~99% with fallback
- **Entities**: 15-20 per paper
- **Relations**: 20-30 per paper

## Next: Neo4j Integration

```cypher
// Load entities
LOAD CSV WITH HEADERS FROM 'file:///entities.csv' AS row
CREATE (e:Entity {id: row.id, type: row.type, name: row.name})

// Load relations
MATCH (s:Entity {id: row.source}), (t:Entity {id: row.target})
CREATE (s)-[:REL {type: row.type}]->(t)
```
