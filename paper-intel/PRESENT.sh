#!/bin/bash
# PRESENTATION DEMO SCRIPT
# Run this during your presentation

cd /home/admin-/Desktop/Sanshodhak/paper-intel

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎯 SANSHODHAK - AI Research Assistant"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ What we built:"
echo "   • Multi-source paper discovery (CrossRef, OpenAlex, arXiv, S2)"
echo "   • Automatic PDF download with fallbacks"
echo "   • Text extraction (PyMuPDF + pdfplumber + PyPDF2)"
echo "   • Vector embeddings (Ollama bge-m3)"
echo "   • RAG system (FAISS + DeepSeek-R1)"
echo "   • Resource recommendations (GitHub, HuggingFace, Web)"
echo ""
echo "📊 Current data:"
echo "   • 11 PDFs downloaded"
echo "   • 2,376 chunks indexed"
echo "   • Topics: Deep Learning, Transformers, Neural Networks"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 DEMO - Interactive RAG + Resource Recommendations"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Commands you can use:"
echo "  • Type any question about deep learning"
echo "  • Type 'resources <topic>' for GitHub/HuggingFace links"
echo "  • Type '1' for sample question"
echo "  • Type 'quit' to exit"
echo ""
echo "Press Enter to start demo..."
read

python3 demo_existing_data.py

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ Demo Complete!"
echo "════════════════════════════════════════════════════════════════════════════════"
