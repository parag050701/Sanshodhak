# 🔬 Sanshodhak: Agentic Research Assistant

**Full-stack AI research assistant with real-time progress tracking**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Overview

Sanshodhak (संशोधक) - Sanskrit for "Researcher" - is an agentic AI system that orchestrates a complete research pipeline:

**Pipeline Stages:**
1. **🔍 SEARCH** - Fetch papers from multiple sources (OpenAlex, CORE, Semantic Scholar, arXiv)
2. **📄 PARSE** - Extract text from PDFs using GROBID/fallback extractors
3. **🧬 EMBED** - Create vector embeddings and build FAISS index
4. **💡 RETRIEVE** - Query the Sanshodhak Retrieval Engine (RAG)
5. **🎯 RECOMMEND** - Find HuggingFace models, GitHub repos, web resources

**Key Features:**
- ✅ Real-time progress tracking via WebSocket
- ✅ Minimalistic black/orange research-focused UI
- ✅ Live citation tracking with paper sources
- ✅ Multi-source paper discovery (open + closed access)
- ✅ 3,383 research papers indexed (expandable)
- ✅ LLM-powered keyword generation for recommendations

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
python3 --version

# Ollama (for embeddings + LLM)
ollama pull bge-m3
ollama pull deepseek-r1:7b
```

### Installation
```bash
# Clone repository
git clone <repo-url>
cd paper-intel

# Install dependencies
pip install -r requirements.txt
pip install flask-socketio python-socketio

# Verify RAG index exists
ls -lh rag_index/faiss.index  # Should be ~14MB
```

### Launch Sanshodhak
```bash
# Option 1: Startup script
./start_sanshodhak.sh

# Option 2: Direct Python
python3 sanshodhak_app.py

# Option 3: Background process
nohup python3 sanshodhak_app.py > sanshodhak.log 2>&1 &
```

**Access the interface:**
- Local: http://localhost:5000
- Network: http://192.168.0.115:5000

---

## 🏗️ Architecture

### Component Structure
```
paper-intel/
├── sanshodhak_agent.py       # Agentic orchestrator
├── sanshodhak_app.py          # Flask + WebSocket server
├── ollama_rag.py              # Retrieval engine
├── resource_recommender_v2.py # Multi-source recommendations
├── ingestion/
│   ├── discovery/
│   │   └── search_engine.py   # Paper search (OpenAlex, CORE, etc.)
│   ├── tei_parser.py          # GROBID XML parser
│   └── preprocessing/
│       └── pdf_to_text.py     # PDF text extraction
├── templates/
│   └── sanshodhak.html        # Modern research UI
└── rag_index/                 # FAISS vector store
    ├── faiss.index            # 3,383 chunks (14MB)
    └── metadata.pkl           # Chunk metadata (1.8MB)
```

### Technology Stack
- **Backend**: Flask 2.3.0 + Flask-SocketIO
- **Frontend**: Vanilla JS + Socket.IO client
- **Embeddings**: bge-m3 (1024-dim, multilingual)
- **LLM**: DeepSeek-R1:7b (reasoning model)
- **Vector Store**: FAISS (cosine similarity)
- **Search APIs**: OpenAlex, CORE, Semantic Scholar, arXiv, CrossRef, Unpaywall

---

## 💻 Usage

### 1. Retrieval Engine Only (Fast)
**Use Case:** Answer questions using existing indexed papers

```bash
# Via Web UI
Mode: "💡 Retrieval Engine"
Query: "What is hybrid retrieval?"
```

**API Endpoint:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid retrieval?", "top_k": 5}'
```

**Response:**
```json
{
  "answer": "Hybrid retrieval combines BM25 and DPR...",
  "citations": [
    {
      "file": "048_Agnihotram_2025.txt",
      "score": 0.727,
      "snippet": "Hybrid retrieval ensures..."
    }
  ]
}
```

### 2. Resource Recommendations Only
**Use Case:** Find related tools/repos/tutorials

```bash
# Via Web UI
Mode: "🎯 Resources Only"
Query: "transformer models"
```

**API Endpoint:**
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer models"}'
```

**Returns:**
- HuggingFace models (top-10 by downloads)
- GitHub repos (top-10 by stars)
- Web tutorials (curated + DuckDuckGo)

### 3. Full Pipeline (Complete)
**Use Case:** Research assistant workflow with all stages

```bash
# Via Web UI
Mode: "🔄 Full Pipeline"
Query: "How does Graph RAG work?"
Options:
  - search_new_papers: false (use existing index)
  - limit: 10
```

**API Endpoint:**
```bash
curl -X POST http://localhost:5000/api/full_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Graph RAG work?",
    "search_new_papers": false,
    "limit": 10
  }'
```

**Pipeline Flow:**
1. **SEARCH** *(skipped if search_new_papers=false)*
2. **PARSE** *(only if new papers found)*
3. **EMBED** *(only if new papers parsed)*
4. **RETRIEVE** *(always executed)*
5. **RECOMMEND** *(always executed)*

---

## 📊 Real-Time Progress Tracking

### WebSocket Updates
The frontend receives live progress updates for each stage:

```javascript
socket.on('progress', (update) => {
  {
    "stage": "retrieve",
    "status": "running",
    "message": "Querying Sanshodhak Retrieval Engine...",
    "timestamp": 1733461234.567,
    "data": {
      "top_k": 5,
      "model": "deepseek-r1:7b"
    }
  }
});
```

**Stage Statuses:**
- `pending` - Waiting to start
- `running` - Currently executing
- `completed` - Finished successfully
- `failed` - Error occurred
- `skipped` - Not needed for this mode

### UI Features
- **Live stage indicators** - Color-coded (orange=running, green=done, red=error)
- **Progress messages** - Real-time status updates
- **Citation tracking** - See which papers contributed to answers
- **Resource cards** - Interactive links to HF models, GitHub repos
- **Minimalistic design** - Black/orange research aesthetic

---

## 🔧 Configuration

### Environment Variables
```bash
# .env file
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3
LLM_MODEL=deepseek-r1:7b

# API Keys (optional but recommended)
SEMANTIC_SCHOLAR_API_KEY=your_key
CORE_API_KEY=your_key
UNPAYWALL_EMAIL=your@email.com
```

### Customization
```python
# sanshodhak_agent.py
agent = SanshodhakAgent(
    rag_system=rag_system,
    enable_search=True,        # Enable paper search
    progress_callback=callback # WebSocket updates
)

# Modify search sources
search_engine = SearchEngine(
    unpaywall_email="research@sanshodhak.ai",
    enable_core=True,          # CORE API
    # Add more sources...
)
```

---

## 📈 Performance

### System Stats
- **Index Size**: 3,383 chunks (36 papers)
- **Embedding Dimension**: 1024 (bge-m3)
- **Vector Store**: FAISS IndexFlatIP (~14MB)
- **Retrieval Time**: ~2-4 seconds (top-5)
- **Generation Time**: ~5-8 seconds (DeepSeek-R1)

### Evaluation Metrics (Existing Index)
```
Retrieval:
  - Precision@1: 0.35
  - Recall@5: 0.75
  - MRR: 0.52
  - NDCG@5: 0.626

Generation:
  - ROUGE-L: 0.0 (paraphrasing expected)
  - BLEU: 0.0 (not direct match)
```

---

## 🛠️ Development

### Adding New Papers
```bash
# 1. Add PDFs to papers/ directory
cp research_paper.pdf papers/

# 2. Rebuild index
python3 build_rag_index.py

# 3. Restart server
./start_sanshodhak.sh
```

### Extending Search Sources
```python
# ingestion/discovery/search_engine.py
def search_open_source(self, query):
    # Add new client
    self.new_source = NewSourceClient(api_key=...)
    
    # Add to parallel execution
    futures[executor.submit(
        self.new_source.search, query, limit
    )] = 'NewSource'
```

### Custom Recommendation Logic
```python
# resource_recommender_v2.py
def recommend(self, query):
    # Modify keyword generation
    keywords = self._generate_keywords_with_llm(query)
    
    # Add new resource type
    custom_resources = self._search_custom_api(keywords)
    
    return {
        'huggingface_models': [...],
        'github_repos': [...],
        'custom_resources': custom_resources  # NEW
    }
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask_socketio'"
```bash
pip install --user flask-socketio python-socketio
```

### Issue: "RuntimeError: Error in faiss::FileIOReader"
```bash
# RAG index not found - rebuild it
python3 build_rag_index.py
```

### Issue: "Ollama connection refused"
```bash
# Start Ollama
ollama serve

# Pull models
ollama pull bge-m3
ollama pull deepseek-r1:7b
```

### Issue: WebSocket not connecting
```bash
# Check firewall
sudo ufw allow 5000

# Check CORS settings in sanshodhak_app.py
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
```

---

## 📚 API Reference

### Health Check
```bash
GET /api/health

Response:
{
  "status": "online",
  "system": "Sanshodhak Research Assistant",
  "components": {
    "agent": true,
    "rag_engine": true,
    "search_engine": true,
    "index_size": 3383
  }
}
```

### Query Retrieval Engine
```bash
POST /api/query
Body: {
  "query": "Your question",
  "top_k": 5
}

Response:
{
  "answer": "Generated answer",
  "sources": [...],
  "citations": [...]
}
```

### Get Recommendations
```bash
POST /api/recommend
Body: {
  "query": "Your topic"
}

Response:
{
  "recommendations": {
    "huggingface_models": [...],
    "github_repos": [...],
    "web_resources": [...]
  }
}
```

### Full Pipeline
```bash
POST /api/full_pipeline
Body: {
  "query": "Your question",
  "search_new_papers": false,
  "limit": 10
}

Response:
{
  "result": {
    "stages": {
      "retrieve": {...},
      "recommend": {...}
    },
    "duration_seconds": 12.5
  }
}
```

### Pipeline Status
```bash
GET /api/status

Response:
{
  "status": {
    "session_id": "20251206_072847",
    "stages": {
      "search": {"status": "pending"},
      "parse": {"status": "pending"},
      "embed": {"status": "pending"},
      "retrieve": {"status": "completed"},
      "recommend": {"status": "completed"}
    }
  }
}
```

---

## 🎨 UI Customization

### Color Theme
Edit `templates/sanshodhak.html`:
```css
:root {
    --bg-primary: #0a0a0a;        /* Main background */
    --accent-orange: #ff6b35;     /* Primary accent */
    --text-primary: #ffffff;      /* Main text */
    --success: #00ff88;           /* Success states */
    --error: #ff4444;             /* Error states */
}
```

### Layout
```html
<!-- Change grid layout -->
<div class="results-grid">
  <!-- Adjust columns -->
  grid-template-columns: 3fr 1fr;  /* More space for answer */
</div>
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📧 Contact

- **Project**: Sanshodhak Research Assistant
- **Documentation**: See `ARCHITECTURE.md` for technical details
- **Issues**: Open GitHub issue for bugs/features

---

## 🙏 Acknowledgments

- **Ollama** - Local LLM runtime
- **FAISS** - Vector similarity search
- **OpenAlex** - Open research metadata
- **CORE** - Open access papers
- **Semantic Scholar** - Citation graphs
- **HuggingFace** - Model/dataset search
- **GitHub** - Code repository search

---

**Built with ❤️ for researchers, by researchers**
