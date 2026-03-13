#!/bin/bash
# Sanshodhak: Agentic Research Assistant - Startup Script

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🔬 SANSHODHAK: AGENTIC RESEARCH ASSISTANT                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import flask_socketio" 2>/dev/null || {
    echo "⚠️  Installing WebSocket dependencies..."
    pip install flask-socketio python-socketio
}

# Check RAG index
if [ ! -f "rag_index/faiss.index" ]; then
    echo "⚠️  RAG index not found!"
    echo "   Please run: python3 build_rag_index.py"
    exit 1
fi

echo "✅ All checks passed"
echo ""
echo "Starting Sanshodhak server..."
echo ""

# Start server
python3 sanshodhak_app.py
