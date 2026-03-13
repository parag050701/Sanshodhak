# 🚀 COMPLETE SEARCH PIPELINE - READY TO USE!

## ✨ What This Does

This is a **COMPLETE** research paper search and analysis pipeline that:

1. **🔍 SEMANTIC SEARCH** - Uses the SAME paper-intel SearchEngine
   - 10X strategy (search 10× more papers, select best)
   - Multi-source: CrossRef, Unpaywell, OpenAlex, CORE, Semantic Scholar, arXiv
   - Intelligent ranking by citations × recency × OA availability

2. **📥 DOWNLOAD PDFs** - Saves to `research_papers/` folder
   - Fallback chain: Unpaywall → arXiv → Publisher
   - Parallel downloads
   - Automatic retry logic

3. **📄 PARSE PDFs** - Extract text with fallback extractors
   - PyMuPDF (fastest) → pdfplumber (accurate) → PyPDF2 (compatible)
   - Saves to `research_papers/extracted_text/`

4. **🧬 EMBED & RAG** - Build NEW vector index
   - Ollama embeddings (bge-m3, 1024-dim)
   - FAISS IndexFlatIP (cosine similarity)
   - Saves to `research_papers/rag_index/`

5. **🎯 EVALUATE** - Test RAG quality
   - Retrieval accuracy
   - Answer relevance
   - Response time
   - Coverage (% papers used)

6. **💡 RETRIEVE** - Answer research questions
   - DeepSeek-R1:7b for answer generation
   - Top-5 source citations
   - Relevance scores

7. **🎯 RECOMMEND** - Find resources
   - HuggingFace models & datasets
   - GitHub repositories
   - Web tutorials & documentation

---

## 🎯 Quick Start

### Option 1: Using the launcher script (RECOMMENDED)
```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel
./run_complete_pipeline.sh
```

### Option 2: Direct execution
```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel
python3 complete_search_pipeline.py
```

---

## 📝 User Inputs

The pipeline will ask you for **3 things**:

1. **Research Topic** - e.g., "Graph Neural Networks", "Transfer Learning", "Quantum Computing"
2. **Number of Papers** - e.g., 5, 10, 20 (default: 10)
3. **Minimum Year** - e.g., 2020, 2018 (default: current year - 5)

### Example:
```
1️⃣  Research Topic: Graph Neural Networks
2️⃣  Number of Papers (default: 10): 5
3️⃣  Minimum Publication Year (default: 2020): 2020
```

---

## 📁 Output Structure

After running, you'll have:

```
research_papers/
├── *.pdf                      # Downloaded PDFs
├── extracted_text/            # Parsed text files
│   └── *.txt
└── rag_index/                 # Vector index
    ├── faiss.index
    └── metadata.pkl
```

---

## ⏱️ Estimated Time

| Papers | Time    |
|--------|---------|
| 5      | 5-10 min |
| 10     | 10-15 min|
| 20     | 15-25 min|
| 50     | 30-60 min|

*Time varies based on PDF availability and network speed*

---

## 🎓 What Makes This Different?

### ✅ COMPLETE IMPLEMENTATION
- **NOT a placeholder** - Uses ACTUAL paper-intel SearchEngine
- **NOT simplified** - Full 10X search strategy with 7 APIs
- **NOT mock data** - Real PDFs downloaded and parsed
- **NOT pre-built index** - NEW embeddings for YOUR topic

### ✅ SAME AS PAPER-INTEL
- Uses `ingestion/discovery/search_engine.py` - EXACT SAME CODE
- Uses `ingestion/ingestion_engine.py` - EXACT SAME ORCHESTRATION
- Uses `ingestion/preprocessing/fallback_extractor.py` - EXACT SAME PARSERS

### ✅ PRODUCTION-READY
- Error handling at every stage
- Progress reporting in real-time
- Fallback strategies for failures
- Quality metrics and evaluation

---

## 🔧 Requirements

### 1. Ollama (REQUIRED)
```bash
# Start Ollama
ollama serve

# Pull models
ollama pull bge-m3          # For embeddings
ollama pull deepseek-r1:7b  # For answer generation
```

### 2. Python Dependencies
Already installed in your environment:
- `numpy`, `faiss-cpu`, `requests`
- All paper-intel ingestion modules

---

## 📊 Example Output

```
================================================================================
  ✅ PIPELINE COMPLETE in 487.3s!
================================================================================

📁 Output Locations:
  • PDFs: research_papers/
  • Texts: research_papers/extracted_text/
  • Index: research_papers/rag_index/

✨ Summary:
  • Papers found: 10
  • PDFs downloaded: 7
  • Texts parsed: 7
  • Chunks indexed: 1,247

🎉 All stages completed successfully!
```

---

## 🎯 Use Cases

### For Presentations
Perfect for quickly building a research knowledge base:
1. Enter your presentation topic
2. Get top papers automatically
3. RAG system answers questions about the topic
4. Get recommended code/models to demo

### For Research
Complete literature review pipeline:
1. Search latest papers in your area
2. Build topic-specific vector index
3. Query for specific information
4. Find implementation resources

### For Learning
Understand a new research area:
1. Download foundational papers
2. Ask questions to RAG system
3. Get tutorials and code examples
4. Explore recommended resources

---

## 🐛 Troubleshooting

### "Ollama is not running"
```bash
ollama serve
```

### "Model not found"
```bash
ollama pull bge-m3
ollama pull deepseek-r1:7b
```

### "No PDFs downloaded"
- Try increasing paper count
- Check internet connection
- Papers might not have OA PDFs (this is normal)

### "Import error"
```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel
export PYTHONPATH=$PWD:$PYTHONPATH
python3 complete_search_pipeline.py
```

---

## 🎯 Next Steps After Pipeline Runs

1. **Check PDFs**: `ls research_papers/*.pdf`
2. **Read texts**: `cat research_papers/extracted_text/*.txt | head -100`
3. **Verify index**: `ls -lh research_papers/rag_index/`
4. **Query RAG**: Use the interactive RAG system
5. **Explore resources**: Check recommended HuggingFace/GitHub links

---

## 📈 Performance Notes

- **10X Search**: Searches 10× more papers than requested for best quality
- **Parallel Downloads**: 5 concurrent PDF downloads
- **Smart Caching**: Reuses existing PDFs if already downloaded
- **Incremental Indexing**: Only indexes new papers

---

## ✅ READY TO USE!

Everything is implemented and working. Just run:

```bash
./run_complete_pipeline.sh
```

Or for presentation mode (auto-answers with defaults):
```bash
echo -e "Graph Neural Networks\n5\n2020" | python3 complete_search_pipeline.py
```

---

**Made for your presentation. Good luck! 🚀**
