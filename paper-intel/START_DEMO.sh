#!/bin/bash

clear

cat << "EOF"
████████████████████████████████████████████████████████████████████████████████
█                         SANSHODHAK                                            █
█                    RESEARCH PIPELINE                                          █
████████████████████████████████████████████████████████████████████████████████

Choose your demo mode:

  1️⃣  QUICK INTERACTIVE DEMO (2-5 minutes)
      • Uses existing 4,382 indexed chunks
      • YOU ask the questions
      • Get GitHub & HuggingFace resources
      • No downloading needed
      ⚡ FAST & INTERACTIVE

  2️⃣  FULL PIPELINE EXECUTION (10-15 minutes)
      • Search 100+ papers from multiple sources
      • Download new PDFs
      • Parse, embed, and index
      • RAG Q&A + Resource recommendations
      🚀 COMPLETE END-TO-END

  3️⃣  Exit

════════════════════════════════════════════════════════════════════════════════
EOF

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Quick Interactive Demo..."
        echo ""
        export OPENROUTER_API_KEY="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"
        python quick_demo.py
        ;;
    2)
        echo ""
        echo "🚀 Starting Full Pipeline Execution..."
        echo ""
        export OPENROUTER_API_KEY="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"
        python final_pipeline.py
        ;;
    3)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run again and select 1, 2, or 3."
        exit 1
        ;;
esac
