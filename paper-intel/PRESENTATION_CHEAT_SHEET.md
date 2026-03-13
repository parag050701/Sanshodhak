# 🎤 PRESENTATION CHEAT SHEET

## 🚀 Quick Start
```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel
./PRESENT.sh
```

---

## 💬 Demo Flow (5 minutes)

### 1. Introduction (30 sec)
"We built Sanshodhak - an AI research assistant that searches papers, downloads PDFs, and provides intelligent answers using RAG."

### 2. Show What's Indexed (15 sec)
```
✅ 11 PDFs downloaded
✅ 2,376 chunks indexed
✅ Topics: Deep Learning, Transformers, Neural Networks
```

### 3. Ask Questions (2 min)
**Example 1**: Basic question
```
💬 Your input: What are transformers?
→ Shows AI-generated answer + 3 source citations
```

**Example 2**: Technical question
```
💬 Your input: How does self-attention mechanism work?
→ Shows detailed explanation from papers
```

**Example 3**: Comparison
```
💬 Your input: What are the main contributions of deep learning?
→ Shows synthesized answer from multiple papers
```

### 4. Resource Recommendations (1.5 min)
**Example 1**: Deep Learning
```
💬 Your input: resources Deep Learning
→ Shows:
   🤗 HuggingFace datasets
   💻 GitHub repos with stars
   🌐 Web resources
```

**Example 2**: Specific topic
```
💬 Your input: resources Transformers
→ Shows relevant GitHub repos and HuggingFace models
```

**Example 3**: Another domain
```
💬 Your input: resources Computer Vision
→ Shows CV-specific resources
```

### 5. Explain Architecture (1 min)
"The system has 7 stages:
1. **Search**: Multi-source (CrossRef, OpenAlex, arXiv, S2)
2. **Download**: PDFs with fallback strategies
3. **Parse**: Text extraction (3 methods)
4. **Embed**: Ollama bge-m3 embeddings
5. **Evaluate**: RAG quality metrics
6. **Retrieve**: DeepSeek-R1 for answers
7. **Recommend**: GitHub, HuggingFace, web crawling"

---

## 🎯 Key Points to Mention

### ✅ What Works
- Multi-source paper discovery (52 papers found, selected top 10)
- Intelligent PDF download (7/10 success rate)
- Robust text extraction (handled 7 PDFs perfectly)
- Fast vector search (2,376 chunks in FAISS)
- High-quality answers (using DeepSeek-R1)
- Resource recommendations (GitHub, HuggingFace, Web)

### 🔧 Technical Stack
- **Search**: CrossRef, Unpaywall, OpenAlex, Semantic Scholar, arXiv
- **Storage**: FAISS vector database
- **Embeddings**: Ollama bge-m3 (1024-dim)
- **LLM**: DeepSeek-R1:7b (local)
- **Parsing**: PyMuPDF → pdfplumber → PyPDF2 (cascading fallbacks)
- **Web**: BeautifulSoup, Requests

### 🚀 Speed
- Full pipeline: ~5-10 minutes (depending on downloads)
- Query response: ~11 seconds
- Resource search: ~30 seconds

---

## 🎬 Sample Questions (Copy-Paste Ready)

### For RAG Demo:
```
What are transformers?
How does self-attention work?
What are the main contributions of deep learning?
Explain the attention mechanism
What are the challenges in deep learning?
```

### For Resource Recommendations:
```
resources Deep Learning
resources Transformers
resources Computer Vision
resources Natural Language Processing
resources Generative AI
```

---

## 📊 Statistics to Quote

- **Papers searched**: 52 candidates
- **PDFs downloaded**: 11 total
- **Chunks indexed**: 2,376
- **Index size**: 9.3 MB
- **Average query time**: 11 seconds
- **Embedding dimension**: 1024
- **Top-k retrieval**: 5 chunks

---

## 🛡️ Backup Plan

If demo fails or network issues:
1. Show the RUN.sh output (already completed)
2. Show files in research_papers/ directory
3. Show FAISS index size
4. Explain architecture from slides
5. Show code in final_pipeline.py

---

## ❓ Expected Questions

**Q: Why only 7/10 PDFs downloaded?**  
A: Some papers are paywalled. Our system tries multiple sources (CrossRef, Unpaywall, OpenAlex) but respects copyright.

**Q: How accurate are the answers?**  
A: Answers are grounded in actual papers - we show source citations with confidence scores. Average retrieval score: 0.6+

**Q: Can it handle other topics?**  
A: Yes! We tested "Transformers" (4 PDFs), "Deep Learning" (11 PDFs). Works for any academic topic.

**Q: How does it compare to ChatGPT?**  
A: ChatGPT has general knowledge. Sanshodhak grounds answers in YOUR downloaded papers - better for research.

**Q: Is it scalable?**  
A: FAISS handles millions of vectors. Currently 2,376 chunks, but can scale to 100K+ papers.

---

## 🎉 Closing Statement

"Sanshodhak demonstrates end-to-end AI research assistance: from discovery to intelligent question answering. The system is modular, scalable, and provides verifiable answers grounded in academic papers."

---

## 🚨 Emergency Commands

If demo freezes:
```bash
# Kill and restart
pkill -f demo_existing_data
python3 demo_existing_data.py
```

If Ollama not responding:
```bash
# Check Ollama
curl http://localhost:11434/api/tags
```

Show existing data:
```bash
ls -lh research_papers/*.pdf
du -sh research_papers/rag_index/
```
