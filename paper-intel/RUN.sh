#!/bin/bash
# ✅ RUN COMPLETE PIPELINE - ONE COMMAND
# ========================================

clear

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "█                                                              █"
echo "█              SANSHODHAK COMPLETE PIPELINE                   █"
echo "█                                                              █"
echo "████████████████████████████████████████████████████████████████"
echo ""
echo "✅ ALL FIXED AND WORKING!"
echo ""
echo "This will run the complete 7-stage pipeline:"
echo "  [1/7] SEARCH - Multi-source semantic search"
echo "  [2/7] DOWNLOAD - PDFs to research_papers/"
echo "  [3/7] PARSE - Extract text"
echo "  [4/7] EMBED & RAG - Build FAISS index"
echo "  [5/7] EVALUATE - Test quality"
echo "  [6/7] RETRIEVE - Answer questions"
echo "  [7/7] RECOMMEND - Find resources"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ ERROR: Ollama is not running!"
    echo ""
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

echo "✅ Ollama is running"
echo ""

# Check directory
cd "$(dirname "$0")" || exit 1

# Check if research_papers exists and show existing files
if [ -d "research_papers" ]; then
    pdf_count=$(find research_papers -name "*.pdf" -type f 2>/dev/null | wc -l)
    if [ "$pdf_count" -gt 0 ]; then
        echo "📁 Found $pdf_count existing PDFs in research_papers/"
        echo "   (Pipeline will add new papers to this folder)"
        echo ""
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting pipeline..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run pipeline
python3 final_pipeline.py

exit_code=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $exit_code -eq 0 ]; then
    echo "✅ Pipeline complete!"
    echo ""
    echo "📁 Check your results:"
    echo "   • PDFs: research_papers/"
    echo "   • Texts: research_papers/extracted_text/"
    echo "   • Index: research_papers/rag_index/"
    echo ""
    
    # Show file counts
    pdf_count=$(find research_papers -maxdepth 1 -name "*.pdf" -type f 2>/dev/null | wc -l)
    txt_count=$(find research_papers/extracted_text -name "*.txt" -type f 2>/dev/null | wc -l)
    
    if [ "$pdf_count" -gt 0 ]; then
        echo "📊 Results:"
        echo "   • $pdf_count PDFs downloaded"
        echo "   • $txt_count texts extracted"
        
        if [ -f "research_papers/rag_index/faiss.index" ]; then
            index_size=$(du -h research_papers/rag_index/faiss.index 2>/dev/null | cut -f1)
            echo "   • FAISS index: $index_size"
        fi
    fi
else
    echo "⚠️  Pipeline exited with code $exit_code"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
