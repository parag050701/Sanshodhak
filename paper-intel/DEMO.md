# 🎯 SANSHODHAK DEMO GUIDE

## Quick Start (30 seconds)

```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel
python3 demo_existing_data.py
```

## Features

### 1️⃣ Ask Questions
Simply type your question and get AI-powered answers from your research papers:

```
💬 Your input: What are transformers?
💬 Your input: How does attention mechanism work?
💬 Your input: What are the main contributions?
```

### 2️⃣ Get Resource Recommendations
Type `resources <topic>` to get:
- 🤗 HuggingFace models & datasets
- 💻 GitHub repositories  
- 🌐 Web resources

Examples:
```
💬 Your input: resources Deep Learning
💬 Your input: resources Transformers
💬 Your input: resources Computer Vision
💬 Your input: resources Natural Language Processing
```

### 3️⃣ Quick Sample Questions
Type a number (1-4) for pre-defined questions:
```
💬 Your input: 1
   → What are transformers in machine learning?
```

## Commands

| Command | Action |
|---------|--------|
| `Type your question` | Ask anything about your papers |
| `resources <topic>` | Get GitHub/HuggingFace/Web resources |
| `1`, `2`, `3`, `4` | Use sample questions |
| `quit` | Exit demo |

## Example Session

```
💬 Your input: What are transformers?
🔍 Searching...
📝 ANSWER: [AI-generated answer from your papers]
📚 Sources: 5 chunks

💬 Your input: resources Transformers
🔍 Finding resources for: Transformers
🎯 RESOURCE RECOMMENDATIONS: Transformers
🤗 HuggingFace: 
   • transformers (library)
   • huggingface/datasets
💻 GitHub:
   • huggingface/transformers ⭐ 120,000
   • lucidrains/vit-pytorch ⭐ 15,000
🌐 Web:
   • Attention Is All You Need (paper)
   • The Illustrated Transformer (blog)

💬 Your input: quit
✅ DEMO COMPLETE!
```

## Your Current Data

✅ **11 PDFs indexed**  
✅ **2,376 chunks** in FAISS index  
✅ **Topics**: Deep Learning, Transformers, Neural Networks

## For Presentations

1. **Start demo**: `python3 demo_existing_data.py`
2. **Ask a question**: Show RAG working
3. **Get resources**: `resources Deep Learning`
4. **Show the code**: Mention Ollama + FAISS + DeepSeek
5. **Explain pipeline**: Search → Download → Parse → Embed → RAG

🎉 **You're ready to present!**
