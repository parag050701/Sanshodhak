#!/bin/bash
# COMPLETE SEARCH PIPELINE - QUICK START
# =======================================

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "█                                                              █"
echo "█        SANSHODHAK COMPLETE SEARCH PIPELINE                  █"
echo "█                                                              █"
echo "████████████████████████████████████████████████████████████████"
echo ""
echo "This script runs the COMPLETE pipeline:"
echo ""
echo "  1. 🔍 SEMANTIC SEARCH - Uses paper-intel SearchEngine"
echo "  2. 📥 DOWNLOAD PDFs - To research_papers/ folder"
echo "  3. 📄 PARSE PDFs - Extract text with fallback extractors"
echo "  4. 🧬 EMBED & RAG - Build FAISS index with Ollama"
echo "  5. 🎯 EVALUATE - Test RAG quality metrics"
echo "  6. 💡 RETRIEVE - Answer research questions"
echo "  7. 🎯 RECOMMEND - Find HuggingFace, GitHub, web resources"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ ERROR: Ollama is not running!"
    echo ""
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    echo "Then pull required models:"
    echo "  ollama pull bge-m3"
    echo "  ollama pull deepseek-r1:7b"
    echo ""
    exit 1
fi

echo "✅ Ollama is running"
echo ""

# Check if models are available
echo "Checking required models..."
if ! ollama list | grep -q "bge-m3"; then
    echo "⚠️  Model bge-m3 not found. Pulling..."
    ollama pull bge-m3
fi

if ! ollama list | grep -q "deepseek-r1"; then
    echo "⚠️  Model deepseek-r1:7b not found. Pulling..."
    ollama pull deepseek-r1:7b
fi

echo "✅ All models ready"
echo ""

# Create output directory
mkdir -p research_papers
echo "✅ Output directory: research_papers/"
echo ""

# Run the pipeline
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting complete pipeline..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$(dirname "$0")"
python3 complete_search_pipeline.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Pipeline complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
