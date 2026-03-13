#!/bin/bash
# SANSHODHAK DEMO LAUNCHER
# Choose between quick demo or full pipeline execution

clear

cat << 'EOF'
════════════════════════════════════════════════════════════════════════════════
🎯 SANSHODHAK RESEARCH PIPELINE DEMO
════════════════════════════════════════════════════════════════════════════════

Choose your demo mode:

  1️⃣  QUICK DEMO (30 seconds) ⚡
      • Uses existing 11 PDFs (already downloaded)
      • 2,376 chunks already indexed
      • Interactive Q&A + Resource recommendations
      • Perfect for: Time-constrained presentations
      
  2️⃣  FULL PIPELINE EXECUTION (5-15 minutes) 🚀
      • Complete 7-stage pipeline LIVE
      • Search → Download → Parse → Embed → RAG → Recommend
      • Watch real-time progress
      • Perfect for: Showing the complete system

════════════════════════════════════════════════════════════════════════════════
EOF

echo ""
read -p "Enter your choice (1 or 2): " choice
echo ""

case $choice in
  1)
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "⚡ QUICK DEMO MODE - Using Existing Data"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "📊 Current data:"
    echo "   • 11 PDFs downloaded"
    echo "   • 2,376 chunks indexed"
    echo "   • Topics: Deep Learning, Transformers, Neural Networks"
    echo ""
    echo "💬 Commands you can use:"
    echo "   • Type any question about your papers"
    echo "   • Type 'resources <topic>' for GitHub/HuggingFace recommendations"
    echo "   • Type 1-4 for sample questions"
    echo "   • Type 'quit' to exit"
    echo ""
    echo "Press Enter to start..."
    read
    python3 demo_existing_data.py
    ;;
    
  2)
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "🚀 FULL PIPELINE EXECUTION - Live Demo"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "You will see ALL 7 stages execute in real-time:"
    echo ""
    echo "  [1/7] SEARCH - Multi-source paper discovery"
    echo "  [2/7] DOWNLOAD - PDF retrieval with fallbacks"
    echo "  [3/7] PARSE - Text extraction from PDFs"
    echo "  [4/7] EMBED - Build FAISS vector index"
    echo "  [5/7] EVALUATE - Test RAG quality"
    echo "  [6/7] RETRIEVE - Answer research question"
    echo "  [7/7] RECOMMEND - Find GitHub/HuggingFace resources"
    echo ""
    echo "📝 You'll be asked for:"
    echo "   1. Research topic (e.g., 'Graph Neural Networks')"
    echo "   2. Number of papers (e.g., 5)"
    echo "   3. Minimum year (e.g., 2020)"
    echo ""
    echo "⏱️  Estimated time: 5-15 minutes (depending on downloads)"
    echo ""
    echo "💡 Recommended topics for best results:"
    echo "   • Graph Neural Networks"
    echo "   • BERT Language Models"
    echo "   • Vision Transformers"
    echo "   • Generative AI"
    echo "   • Reinforcement Learning"
    echo ""
    echo "⚠️  Avoid: Medical/Healthcare topics (often paywalled)"
    echo ""
    read -p "Press Enter to start full pipeline..." 
    python3 final_pipeline.py
    ;;
    
  *)
    echo "❌ Invalid choice. Please run again and choose 1 or 2."
    exit 1
    ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ Demo Complete!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Your data is in:"
echo "   • PDFs: research_papers/"
echo "   • Texts: research_papers/extracted_text/"
echo "   • Index: research_papers/rag_index/"
echo ""
echo "🔄 Run again: ./DEMO_LAUNCHER.sh"
echo ""
