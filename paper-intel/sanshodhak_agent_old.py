"""
Sanshodhak: Agentic Research Assistant - FULLY INTEGRATED
==========================================================
Complete pipeline with REAL paper-intel integration:
Search → Download → Parse → Embed → Retrieve → Recommend
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Import PAPER-INTEL components (the real ones!)
from ingestion.ingestion_engine import IngestionEngine
from ingestion.models import PaperMetadata, DownloadResult
from ingestion.fallback_extractor import FallbackExtractor
from ingestion.tei_parser import TEIParser

# Import our RAG and recommendation components
from ollama_rag import OllamaRAG
from resource_recommender_v2 import EnhancedResourceRecommender

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of each pipeline stage"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProgressUpdate:
    """Progress update message"""
    stage: str
    status: str
    message: str
    timestamp: float
    data: Optional[Dict] = None


class SanshodhakAgent:
    """
    FULLY INTEGRATED Agentic Research Assistant
    
    Pipeline Stages:
    1. SEARCH: Multi-source paper discovery (OpenAlex, CORE, Semantic Scholar, arXiv, CrossRef, Unpaywall)
    2. DOWNLOAD: Intelligent PDF downloading with fallbacks
    3. PARSE: Extract structured text from PDFs (GROBID TEI XML + fallbacks)
    4. EMBED: Create vector embeddings and build/update FAISS index
    5. RETRIEVE: Query Sanshodhak Retrieval Engine (RAG)
    6. RECOMMEND: Find HuggingFace models, GitHub repos, web resources
    """
    
    def __init__(
        self,
        rag_system: Optional[OllamaRAG] = None,
        enable_search: bool = True,
        papers_dir: str = "research_papers",
        progress_callback: Optional[Callable] = None,
        # API keys
        s2_api_key: Optional[str] = None,
        core_api_key: Optional[str] = None,
        unpaywall_email: Optional[str] = None
    ):
        """
        Initialize Sanshodhak Agent with FULL paper-intel integration.
        
        Args:
            rag_system: Pre-initialized RAG system (or will create new)
            enable_search: Enable paper search functionality
            papers_dir: Directory to save downloaded PDFs
            progress_callback: Function to call with progress updates
            s2_api_key: Semantic Scholar API key
            core_api_key: CORE API key
            unpaywall_email: Email for Unpaywall
        """
        self.rag_system = rag_system
        self.enable_search = enable_search
        self.progress_callback = progress_callback
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        
        # API credentials
        self.s2_api_key = s2_api_key
        self.core_api_key = core_api_key
        self.unpaywall_email = unpaywall_email or "research@sanshodhak.ai"
        
        # Initialize extractors
        self.pdf_extractor = FallbackExtractor()
        self.tei_parser = TEIParser()
        
        # Initialize recommender
        self.recommender = None
        
        # Pipeline state
        self.stages = {
            'search': {'status': StageStatus.PENDING, 'data': {}},
            'download': {'status': StageStatus.PENDING, 'data': {}},
            'parse': {'status': StageStatus.PENDING, 'data': {}},
            'embed': {'status': StageStatus.PENDING, 'data': {}},
            'retrieve': {'status': StageStatus.PENDING, 'data': {}},
            'recommend': {'status': StageStatus.PENDING, 'data': {}}
        }
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_cache = {}
    
    def _emit_progress(self, stage: str, status: str, message: str, data: Optional[Dict] = None):
        """Emit progress update"""
        update = ProgressUpdate(
            stage=stage,
            status=status,
            message=message,
            timestamp=datetime.now().timestamp(),
            data=data
        )
        
        # Update internal state
        if stage in self.stages:
            self.stages[stage]['status'] = StageStatus(status)
            if data:
                self.stages[stage]['data'].update(data)
        
        # Log
        logger.info(f"[{stage.upper()}] {message}")
        
        # Callback
        if self.progress_callback:
            try:
                self.progress_callback(asdict(update))
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    async def search_papers(
        self,
        query: str,
        required_count: int = 10,
        min_year: Optional[int] = None
    ) -> List[PaperMetadata]:
        """
        Stage 1: SEARCH - Use REAL paper-intel multi-source search
        
        Sources (in order):
        - OpenAlex (open metadata)
        - CORE (open access full-text)
        - Semantic Scholar (citations + abstracts)
        - arXiv (preprints)
        - CrossRef (DOI resolution)
        - Unpaywall (OA finder)
        
        This is THE REAL paper-intel search functionality!
        """
        if not self.enable_search:
            self._emit_progress('search', 'skipped', 'Search disabled')
            return []
        
        self._emit_progress('search', 'running', 
            f'🔍 Searching papers: "{query}"', {
                'query': query,
                'target_count': required_count,
                'sources': ['OpenAlex', 'CORE', 'Semantic Scholar', 'arXiv', 'CrossRef', 'Unpaywall']
            })
        
        try:
            # Initialize the REAL ingestion engine
            engine = IngestionEngine(
                query=query,
                required_count=required_count,
                output_dir=str(self.papers_dir),
                min_year=min_year,
                s2_api_key=self.s2_api_key,
                core_api_key=self.core_api_key,
                unpaywall_email=self.unpaywall_email,
                enable_core=True,
                prefer_open_access=True
            )
            
            # Search open-source layer
            self._emit_progress('search', 'running', 
                '📚 Searching open-source APIs (OpenAlex, CORE, S2)...')
            
            open_papers = await engine.search_open_source()
            
            self._emit_progress('search', 'running',
                f'✓ Open-source: {len(open_papers)} papers found', {
                    'open_source_count': len(open_papers)
                })
            
            # Search closed-access layer
            self._emit_progress('search', 'running',
                '🔒 Searching closed-access APIs (CrossRef + Unpaywall)...')
            
            closed_papers = await engine.search_closed_access()
            
            self._emit_progress('search', 'running',
                f'✓ Closed-access: {len(closed_papers)} papers found', {
                    'closed_access_count': len(closed_papers)
                })
            
            # Merge and deduplicate
            self._emit_progress('search', 'running',
                '🔄 Merging and deduplicating results...')
            
            all_papers = await engine.merge_and_deduplicate(open_papers, closed_papers)
            
            # Rank papers
            ranked_papers = await engine.rank_papers(all_papers)
            
            # Get top papers
            final_papers = ranked_papers[:required_count]
            
            # Prepare data for frontend
            papers_data = [
                {
                    'title': p.title,
                    'doi': p.doi,
                    'year': p.year,
                    'authors': p.authors[:3] if p.authors else [],
                    'source': p.source,
                    'citations': p.citation_count,
                    'is_open_access': p.is_open_access,
                    'pdf_url': p.pdf_url
                } for p in final_papers
            ]
            
            self._emit_progress('search', 'completed',
                f'✅ Found {len(final_papers)} papers (deduplicated from {len(open_papers) + len(closed_papers)})', {
                    'paper_count': len(final_papers),
                    'papers': papers_data
                })
            
            return final_papers
            
        except Exception as e:
            self._emit_progress('search', 'failed', f'❌ Search error: {str(e)}')
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    async def download_papers(self, papers: List[PaperMetadata]) -> Dict[str, str]:
        """
        Stage 2: DOWNLOAD - Use REAL paper-intel PDF downloader
        
        Download strategy:
        1. Try PDF URL directly
        2. Try arXiv if available
        3. Try DOI resolution
        4. (Optional) Sci-Hub fallback
        
        Returns mapping: paper_id -> pdf_path
        """
        if not papers:
            self._emit_progress('download', 'skipped', 'No papers to download')
            return {}
        
        self._emit_progress('download', 'running',
            f'📥 Downloading {len(papers)} PDFs...', {
                'total': len(papers)
            })
        
        pdf_paths = {}
        
        # Initialize the REAL ingestion engine for downloading
        engine = IngestionEngine(
            query="",  # Not needed for download
            required_count=len(papers),
            output_dir=str(self.papers_dir),
            unpaywall_email=self.unpaywall_email
        )
        
        try:
            # Download PDFs using paper-intel downloader
            download_results = await engine.download_pdfs(papers)
            
            # Process results
            success_count = 0
            for paper_id, result in download_results.items():
                if result['success']:
                    pdf_paths[paper_id] = result['pdf_path']
                    success_count += 1
                    
                    self._emit_progress('download', 'running',
                        f'✓ Downloaded: {result["filename"]}', {
                            'progress': success_count,
                            'total': len(papers),
                            'percentage': (success_count / len(papers)) * 100
                        })
            
            self._emit_progress('download', 'completed',
                f'✅ Downloaded {success_count}/{len(papers)} PDFs', {
                    'success_count': success_count,
                    'total': len(papers),
                    'success_rate': (success_count / len(papers)) * 100
                })
            
            return pdf_paths
            
        except Exception as e:
            self._emit_progress('download', 'failed', f'❌ Download error: {str(e)}')
            logger.error(f"Download failed: {e}", exc_info=True)
            return {}
    
    async def parse_pdfs(self, pdf_paths: Dict[str, str]) -> List[Dict]:
        """
        Stage 3: PARSE - Extract text from PDFs using REAL paper-intel extractors
        
        Methods (in order):
        1. GROBID TEI XML (structured, best quality)
        2. PyMuPDF (fast fallback)
        3. pdfplumber (accurate fallback)
        4. PyPDF2 (last resort)
        
        Returns list of parsed documents with text and metadata
        """
        if not pdf_paths:
            self._emit_progress('parse', 'skipped', 'No PDFs to parse')
            return []
        
        self._emit_progress('parse', 'running',
            f'📄 Parsing {len(pdf_paths)} PDFs...', {
                'total': len(pdf_paths)
            })
        
        parsed_docs = []
        
        for i, (paper_id, pdf_path) in enumerate(pdf_paths.items()):
            try:
                # Use REAL paper-intel fallback extractor
                success, text = self.pdf_extractor.extract(Path(pdf_path))
                
                if success and text:
                    parsed_docs.append({
                        'paper_id': paper_id,
                        'pdf_path': pdf_path,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
                    
                    self._emit_progress('parse', 'running',
                        f'✓ Parsed: {Path(pdf_path).name} ({len(text.split())} words)', {
                            'progress': i + 1,
                            'total': len(pdf_paths),
                            'percentage': ((i + 1) / len(pdf_paths)) * 100
                        })
                else:
                    logger.warning(f"Failed to extract text from {pdf_path}")
                    
            except Exception as e:
                logger.error(f"Parse error for {pdf_path}: {e}")
        
        self._emit_progress('parse', 'completed',
            f'✅ Parsed {len(parsed_docs)}/{len(pdf_paths)} PDFs', {
                'parsed_count': len(parsed_docs),
                'total': len(pdf_paths),
                'total_words': sum(d['word_count'] for d in parsed_docs)
            })
        
        return parsed_docs
    
    async def embed_and_index(self, parsed_docs: List[Dict]) -> bool:
        """
        Stage 4: EMBED - Create embeddings and update FAISS index
        
        Process:
        1. Chunk documents (512 tokens with 50 overlap)
        2. Generate embeddings (bge-m3, 1024-dim)
        3. Add to FAISS index
        4. Save updated index
        
        Model: bge-m3 (multilingual, optimized for retrieval)
        Index: FAISS IndexFlatIP (cosine similarity)
        """
        if not parsed_docs:
            self._emit_progress('embed', 'skipped', 'No documents to embed')
            return False
        
        self._emit_progress('embed', 'running',
            f'🧬 Creating embeddings for {len(parsed_docs)} documents...', {
                'model': 'bge-m3',
                'dimension': 1024,
                'doc_count': len(parsed_docs)
            })
        
        try:
            # Initialize or load RAG system
            if not self.rag_system:
                self.rag_system = OllamaRAG()
            
            # Build new index with all documents
            all_texts = []
            all_metadata = []
            
            for doc in parsed_docs:
                # Chunk the text (simple chunking - 512 tokens)
                text = doc['text']
                chunks = [text[i:i+2000] for i in range(0, len(text), 1900)]  # ~500 tokens per chunk
                
                for chunk_idx, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_metadata.append({
                        'file': Path(doc['pdf_path']).name,
                        'paper_id': doc['paper_id'],
                        'chunk_id': chunk_idx,
                        'chunk_count': len(chunks)
                    })
            
            self._emit_progress('embed', 'running',
                f'📊 Created {len(all_texts)} chunks from {len(parsed_docs)} documents', {
                    'chunk_count': len(all_texts),
                    'doc_count': len(parsed_docs)
                })
            
            # Build index
            self._emit_progress('embed', 'running',
                f'🔨 Building FAISS index with {len(all_texts)} chunks...')
            
            # Use RAG system to build index
            self.rag_system.build_index(all_texts, all_metadata)
            
            # Save index
            self.rag_system.save('rag_index')
            
            self._emit_progress('embed', 'completed',
                f'✅ Indexed {len(all_texts)} chunks from {len(parsed_docs)} papers', {
                    'chunk_count': len(all_texts),
                    'doc_count': len(parsed_docs),
                    'index_size': self.rag_system.index.ntotal
                })
            
            return True
            
        except Exception as e:
            self._emit_progress('embed', 'failed', f'❌ Embedding error: {str(e)}')
            logger.error(f"Embed failed: {e}", exc_info=True)
            return False
    
    async def retrieve_answer(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Stage 5: RETRIEVE - Query Sanshodhak Retrieval Engine
        
        Method: Hybrid retrieval (semantic + keyword)
        LLM: DeepSeek-R1:7b (reasoning model)
        Context: Top-K retrieved chunks
        """
        if not self.rag_system or not self.rag_system.index:
            self._emit_progress('retrieve', 'failed', 'RAG system not initialized or no index')
            return {}
        
        self._emit_progress('retrieve', 'running',
            f'💡 Querying Sanshodhak Retrieval Engine: "{query}"', {
                'query': query,
                'top_k': top_k,
                'model': 'deepseek-r1:7b',
                'index_size': self.rag_system.index.ntotal
            })
        
        try:
            # Execute RAG query
            answer, sources = self.rag_system.query(query, top_k=top_k)
            
            # Extract citations
            citations = [
                {
                    'file': src['file'],
                    'score': f"{src['score']:.3f}",
                    'snippet': src['snippet'][:200] + '...' if len(src['snippet']) > 200 else src['snippet']
                } for src in sources
            ]
            
            self._emit_progress('retrieve', 'completed',
                f'✅ Generated answer with {len(sources)} citations', {
                    'answer_length': len(answer),
                    'citation_count': len(sources),
                    'citations': citations
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'citations': citations
            }
            
        except Exception as e:
            self._emit_progress('retrieve', 'failed', f'❌ Retrieval error: {str(e)}')
            logger.error(f"Retrieve failed: {e}", exc_info=True)
            return {}
    
    async def recommend_resources(self, query: str) -> Dict[str, Any]:
        """
        Stage 6: RECOMMEND - Find related resources
        
        Sources:
        - HuggingFace (models, datasets)
        - GitHub (repositories)
        - Web (tutorials, documentation)
        - Papers with Code (implementations)
        """
        self._emit_progress('recommend', 'running',
            f'🎯 Finding resources for: "{query}"')
        
        try:
            # Initialize recommender if needed
            if not self.recommender:
                self.recommender = EnhancedResourceRecommender(self.rag_system)
            
            # Get recommendations
            resources = self.recommender.recommend(query)
            
            summary = {
                'huggingface_count': len(resources.get('huggingface_models', [])),
                'github_count': len(resources.get('github_repos', [])),
                'web_count': len(resources.get('web_resources', []))
            }
            
            self._emit_progress('recommend', 'completed',
                f'✅ Found {sum(summary.values())} resources', summary)
            
            return resources
            
        except Exception as e:
            self._emit_progress('recommend', 'failed', f'❌ Recommendation error: {str(e)}')
            logger.error(f"Recommend failed: {e}", exc_info=True)
            return {}
    
    async def execute_full_pipeline(
        self,
        query: str,
        search_new_papers: bool = True,
        paper_count: int = 10,
        min_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute COMPLETE Sanshodhak pipeline with REAL paper-intel integration.
        
        Pipeline:
        1. SEARCH: Multi-source paper discovery → List[PaperMetadata]
        2. DOWNLOAD: Intelligent PDF downloading → Dict[paper_id, pdf_path]
        3. PARSE: Extract text from PDFs → List[parsed_docs]
        4. EMBED: Build/update FAISS index → bool
        5. RETRIEVE: Generate answer with RAG → answer + citations
        6. RECOMMEND: Find resources → HF models + GitHub repos + Web
        
        Args:
            query: User's research question
            search_new_papers: Whether to search and download new papers
            paper_count: Number of papers to fetch
            min_year: Minimum publication year
            
        Returns:
            Complete results with all stage outputs
        """
        start_time = datetime.now()
        
        self._emit_progress('pipeline', 'running',
            f'🚀 Starting Sanshodhak Full Pipeline: "{query}"', {
                'session_id': self.session_id,
                'search_enabled': search_new_papers,
                'paper_count': paper_count
            })
        
        results = {
            'query': query,
            'session_id': self.session_id,
            'stages': {}
        }
        
        try:
            # Stage 1: SEARCH (if enabled)
            if search_new_papers:
                papers = await self.search_papers(query, required_count=paper_count, min_year=min_year)
                results['stages']['search'] = {
                    'paper_count': len(papers),
                    'papers': [{'title': p.title, 'year': p.year, 'doi': p.doi} for p in papers[:5]]
                }
                
                # Stage 2: DOWNLOAD
                if papers:
                    pdf_paths = await self.download_papers(papers)
                    results['stages']['download'] = {
                        'success_count': len(pdf_paths),
                        'total': len(papers)
                    }
                    
                    # Stage 3: PARSE
                    if pdf_paths:
                        parsed_docs = await self.parse_pdfs(pdf_paths)
                        results['stages']['parse'] = {
                            'parsed_count': len(parsed_docs),
                            'total_words': sum(d['word_count'] for d in parsed_docs)
                        }
                        
                        # Stage 4: EMBED
                        if parsed_docs:
                            embed_success = await self.embed_and_index(parsed_docs)
                            results['stages']['embed'] = {
                                'success': embed_success,
                                'index_size': self.rag_system.index.ntotal if self.rag_system else 0
                            }
            
            # Stage 5: RETRIEVE (always, uses existing or new index)
            retrieval_result = await self.retrieve_answer(query, top_k=5)
            results['stages']['retrieve'] = retrieval_result
            
            # Stage 6: RECOMMEND (always)
            recommendations = await self.recommend_resources(query)
            results['stages']['recommend'] = recommendations
            
            # Calculate timing
            duration = (datetime.now() - start_time).total_seconds()
            results['duration_seconds'] = duration
            
            self._emit_progress('pipeline', 'completed',
                f'✅ Pipeline completed in {duration:.2f}s', {
                    'duration': duration
                })
            
            return results
            
        except Exception as e:
            self._emit_progress('pipeline', 'failed', f'❌ Pipeline error: {str(e)}')
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of all pipeline stages"""
        return {
            'session_id': self.session_id,
            'stages': {
                name: {
                    'status': stage['status'].value,
                    'data': stage['data']
                }
                for name, stage in self.stages.items()
            }
        }
