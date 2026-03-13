#!/bin/bash
# Quick Reference for RAG System Usage

echo "=============================================="
echo "  RAG SYSTEM - QUICK REFERENCE"
echo "=============================================="
echo ""

# Set working directory
cd /home/admin-/Desktop/Sanshodhak/paper-intel || exit 1

show_menu() {
    echo "Choose an option:"
    echo ""
    echo "1) Run comprehensive evaluation"
    echo "2) Interactive resource recommender"
    echo "3) Test RAG with single query"
    echo "4) View evaluation results"
    echo "5) View demo answers"
    echo "6) Batch resource recommendations"
    echo "7) Generate new test questions"
    echo "8) Quick system status"
    echo "9) Exit"
    echo ""
    read -p "Enter choice [1-9]: " choice
    
    case $choice in
        1)
            echo ""
            echo "Running comprehensive evaluation..."
            python3 run_comprehensive_eval.py
            ;;
        2)
            echo ""
            echo "Starting interactive resource recommender..."
            python3 interactive.py
            ;;
        3)
            echo ""
            read -p "Enter your query: " query
            python3 -c "
from ollama_rag import OllamaRAG
rag = OllamaRAG()
rag.load('rag_index')
print('\n=== Search Results ===')
results = rag.search('$query', top_k=3)
for i, r in enumerate(results, 1):
    print(f'[{i}] {r[\"file\"]} (score: {r[\"score\"]:.3f})')
    print(f'    {r[\"text\"][:150]}...\n')
print('=== Generated Answer ===')
answer = rag.query('$query')
print(answer)
"
            ;;
        4)
            echo ""
            if [ -f evaluation_results_OllamaRAG.json ]; then
                echo "=== Evaluation Results ==="
                python3 -c "
import json
with open('evaluation_results_OllamaRAG.json') as f:
    results = json.load(f)
print(f\"System: {results['system']}\")
print(f\"Timestamp: {results['timestamp']}\")
print(f\"Evaluation Time: {results['evaluation_time_seconds']:.1f}s\")
print(f\"\nRetrieval Metrics:\")
for k, v in results['retrieval']['metrics'].items():
    print(f\"  {k}: {v:.4f}\")
print(f\"\nGeneration Metrics:\")
for k, v in results['generation']['metrics'].items():
    print(f\"  {k}: {v:.4f}\")
"
            else
                echo "❌ No evaluation results found. Run option 1 first."
            fi
            ;;
        5)
            echo ""
            if [ -f demo_answers.json ]; then
                echo "=== Demo Answers (First 2) ==="
                python3 -c "
import json
with open('demo_answers.json') as f:
    demos = json.load(f)
for i, demo in enumerate(demos[:2], 1):
    print(f\"\n[Demo {i}]\")
    print(f\"Q: {demo['question']}\")
    print(f\"A: {demo.get('generated_answer', 'N/A')[:200]}...\")
    print(f\"Docs: {len(demo.get('retrieved_docs', []))} retrieved\")
"
            else
                echo "❌ No demo answers found. Run option 1 first."
            fi
            ;;
        6)
            echo ""
            echo "Running batch resource recommendations..."
            if [ -f demo_queries.json ]; then
                python3 interactive.py batch demo_queries.json
            else
                echo "❌ demo_queries.json not found"
            fi
            ;;
        7)
            echo ""
            echo "Generating new test questions..."
            python3 -c "
import json
questions = []
print('Enter questions (empty line to finish):')
while True:
    q = input('Question: ').strip()
    if not q:
        break
    a = input('Answer: ').strip()
    questions.append({
        'question': q,
        'answer': a,
        'type': 'factual',
        'difficulty': 'medium',
        'relevant_docs': []
    })
    
if questions:
    with open('custom_test_questions.json', 'w') as f:
        json.dump(questions, f, indent=2)
    print(f'\n✅ Saved {len(questions)} questions to custom_test_questions.json')
"
            ;;
        8)
            echo ""
            echo "=== System Status ==="
            echo ""
            echo "📦 RAG Index:"
            if [ -d rag_index ]; then
                echo "  ✅ rag_index/ exists"
                echo "  Files: $(ls rag_index | wc -l)"
            else
                echo "  ❌ rag_index/ not found"
            fi
            echo ""
            echo "📊 Evaluation Files:"
            [ -f evaluation_results_OllamaRAG.json ] && echo "  ✅ evaluation_results_OllamaRAG.json" || echo "  ❌ evaluation_results_OllamaRAG.json"
            [ -f demo_answers.json ] && echo "  ✅ demo_answers.json" || echo "  ❌ demo_answers.json"
            [ -f rag_test_questions.json ] && echo "  ✅ rag_test_questions.json ($(python3 -c "import json; print(len(json.load(open('rag_test_questions.json'))))" 2>/dev/null || echo "?") questions)" || echo "  ❌ rag_test_questions.json"
            echo ""
            echo "🤖 Ollama Status:"
            if command -v ollama &> /dev/null; then
                echo "  ✅ Ollama installed"
                ollama list 2>/dev/null | grep -E "bge-m3|deepseek-r1" || echo "  ⚠️  Models not found"
            else
                echo "  ❌ Ollama not installed"
            fi
            echo ""
            echo "📚 Documentation:"
            [ -f EVALUATION_SUMMARY.md ] && echo "  ✅ EVALUATION_SUMMARY.md" || echo "  ❌ EVALUATION_SUMMARY.md"
            [ -f README.md ] && echo "  ✅ README.md" || echo "  ❌ README.md"
            ;;
        9)
            echo ""
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo ""
            echo "❌ Invalid choice. Please try again."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
    show_menu
}

# Main
clear
show_menu
