"""
Sanshodhak: Agentic Research Assistant - Web Application
========================================================
Real-time research assistant with progress tracking
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any
import json

# Set working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Import Sanshodhak components
from sanshodhak_agent import SanshodhakAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sanshodhak_secret_key_2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global agent
agent: SanshodhakAgent = None


def progress_callback(update: Dict[str, Any]):
    """Send progress updates to frontend via WebSocket"""
    try:
        socketio.emit('progress', update)
    except Exception as e:
        logger.error(f"WebSocket emit error: {e}")


@app.route('/')
def index():
    """Main research assistant interface"""
    return render_template('sanshodhak.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    return jsonify({
        'status': 'online',
        'system': 'Sanshodhak Research Assistant',
        'version': '1.0.0',
        'components': {
            'agent': agent is not None,
            'rag_engine': agent.rag_system is not None if agent else False,
            'ingestion_engine': agent.ingestion_engine is not None if agent else False,
            'index_size': agent.rag_system.index.ntotal if agent and agent.rag_system and agent.rag_system.index else 0
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/query', methods=['POST'])
def query_rag_only():
    """
    Query only the Sanshodhak Retrieval Engine (RAG)
    Skip search/parse/embed stages
    """
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not agent:
            return jsonify({'error': 'Agent not initialized'}), 500
        
        # Run retrieval only
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(agent.retrieve_answer(query, top_k=top_k))
        loop.close()
        
        return jsonify({
            'success': True,
            'query': query,
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'citations': result.get('citations', []),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend_only():
    """Get resource recommendations only"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not agent:
            return jsonify({'error': 'Agent not initialized'}), 500
        
        # Run recommendations only
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(agent.recommend_resources(query))
        loop.close()
        
        return jsonify({
            'success': True,
            'query': query,
            'recommendations': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Recommend error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/full_pipeline', methods=['POST'])
def full_pipeline():
    """
    Execute complete Sanshodhak pipeline:
    Search → Download → Parse → Embed → Retrieve → Recommend
    """
    try:
        data = request.json
        query = data.get('query', '')
        required_papers = data.get('required_papers', 30)
        min_year = data.get('min_year', 2020)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not agent:
            return jsonify({'error': 'Agent not initialized'}), 500
        
        # Execute full pipeline with progress tracking
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            agent.execute_full_pipeline(
                query=query,
                required_papers=required_papers,
                min_year=min_year
            )
        )
        loop.close()
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def pipeline_status():
    """Get current pipeline status"""
    try:
        if not agent:
            return jsonify({'error': 'Agent not initialized'}), 500
        
        status = agent.get_pipeline_status()
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Status error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'message': 'Connected to Sanshodhak',
        'session_id': request.sid
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


def initialize_agent():
    """Initialize Sanshodhak Agent"""
    global agent
    
    print("\n" + "="*70)
    print("  🔬 SANSHODHAK: AGENTIC RESEARCH ASSISTANT")
    print("="*70)
    
    print("\n🤖 Initializing Sanshodhak Agent...")
    agent = SanshodhakAgent(
        progress_callback=progress_callback
    )
    print("✅ Agent initialized - Ready for paper discovery!")
    
    print("\n" + "="*70)
    print("  🌐 SERVER READY")
    print("="*70)
    print(f"  Local:   http://localhost:5000")
    print(f"  Network: http://0.0.0.0:5000")
    print(f"  Mode:    Full Pipeline (Search → Download → Parse → Embed → RAG)")
    print("="*70 + "\n")


if __name__ == '__main__':
    initialize_agent()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
