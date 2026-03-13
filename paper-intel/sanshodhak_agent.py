"""
Sanshodhak: Agentic Research Assistant
=======================================
COMPLETE pipeline with paper-intel integration:
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
import shutil
import hashlib

# Import paper-intel components
from ingestion.ingestion_engine import IngestionEngine
from ingestion.fallback_extractor import extract_text_fast
from enhanced_rag import EnhancedRAG
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
    Agentic Research Assistant - Full Pipeline Orchestrator
    
    Complete Stages:
    1. SEARCH: Multi-source paper discovery (OpenAlex, CORE, S2, CrossRef, Unpaywall, arXiv)
    2. DOWNLOAD: PDF retrieval with fallbacks (OA → Sci-Hub → LibGen)
    3. PARSE: Text extraction (GROBID → PyMuPDF → pdfplumber)
    4. EMBED: Vector embeddings (bge-m3) + FAISS index
    5. RETRIEVE: RAG query with DeepSeek-R1
    6. RECOMMEND: HuggingFace/GitHub/Web resources
    """
    
    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        base_output_dir: str = "/home/admin-/Desktop/Sanshodhak/paper-intel"
    ):
        """
        Initialize Sanshodhak Agent.
        
        Args:
            progress_callback: Function to call with progress updates
            base_output_dir: Base directory for outputs
        """
        self.progress_callback = progress_callback
        self.base_output_dir = Path(base_output_dir)
        
        # Session-specific directories (created per query)
        self.session_id = None
        self.session_dir = None
        self.papers_dir = None
        self.text_dir = None
        self.index_dir = None
        
        # Components (initialized per session)
        self.ingestion_engine = None
        self.rag_system = None
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
        
        self.results_cache = {}
    
    def _create_session_directories(self, query: str):
        """Create session-specific directories"""
        # Generate session ID from query hash + timestamp
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{query_hash}_{timestamp}"
        
        # Create directories
        self.session_dir = self.base_output_dir / "sessions" / self.session_id
        self.papers_dir = self.session_dir / "papers"
        self.text_dir = self.session_dir / "text"
        self.index_dir = self.session_dir / "index"
        
        for dir_path in [self.papers_dir, self.text_dir, self.index_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created session: {self.session_id}")
        logger.info(f"  Papers: {self.papers_dir}")
        logger.info(f"  Text: {self.text_dir}")
        logger.info(f"  Index: {self.index_dir}")
    
    def _emit_progress(self, stage: str, status: str, message: str, data: Optional[Dict] = None):
        """Emit progress update"""
        update = ProgressUpdate(
            stage=stage,
            status=status,
            message=message,
            timestamp=datetime.now().timestamp(),
            data=data or {}
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
        required_count: int = 30,
        min_year: Optional[int] = 2020
    ) -> List[Any]:
        """
        Stage 1: SEARCH - Multi-source paper discovery with 10X STRATEGY
        
        10X SEARCH STRATEGY:
        1. User wants N papers → Search 10×N papers for maximum quality
        2. Search order (comprehensive):
           - CrossRef: 10×N papers (metadata + DOIs)
           - Unpaywall: Enrich with OA PDF links
           - OpenAlex: 10×N papers (open access focus)
           - CORE: 10×N papers (academic repositories)
           - Semantic Scholar: 10×N papers (AI-powered)
           - arXiv: Preprints
        3. Merge & deduplicate all sources
        4. Rank by: citations × recency × OA availability
        5. Filter to TOP N highest-quality papers
        """
        search_multiplier = 10  # 10X search for quality
        search_target = required_count * search_multiplier
        
        self._emit_progress('search', 'running', 
            f'🔍 10X SEARCH: User wants {required_count} papers → Searching {search_target} papers for best quality', {
                'query': query,
                'user_requested': required_count,
                'searching': search_target,
                'multiplier': search_multiplier,
                'min_year': min_year
            })
        
        try:
            # Initialize ingestion engine with 10X target
            self.ingestion_engine = IngestionEngine(
                query=query,
                required_count=search_target,  # Search 10X more!
                output_dir=str(self.papers_dir),
                min_year=min_year,
                unpaywall_email="research@sanshodhak.ai",
                enable_core=True,
                prefer_open_access=True,
                max_iterations=2
            )
            
            # STEP 1: CrossRef (closed-access metadata) + Unpaywall enrichment
            self._emit_progress('search', 'running', 
                f'[1/3] 📚 Searching CrossRef + Unpaywall for {search_target} papers...', {
                    'stage': 'crossref',
                    'target': search_target
                })
            
            closed_papers = await self.ingestion_engine.search_closed_access()
            oa_count_closed = sum(1 for p in closed_papers if p.is_open_access)
            
            self._emit_progress('search', 'running', 
                f'[1/3] ✅ CrossRef: Found {len(closed_papers)} papers ({oa_count_closed} with OA PDFs)', {
                    'closed_source_count': len(closed_papers),
                    'oa_count': oa_count_closed
                })
            
            # STEP 2: Open-source layer (OpenAlex, CORE, S2, ArcSieve, arXiv)
            self._emit_progress('search', 'running', 
                f'[2/3] 🌐 Searching OpenAlex + CORE + Semantic Scholar + arXiv...', {
                    'stage': 'open_source',
                    'target': search_target
                })
            
            open_papers = await self.ingestion_engine.search_open_source()
            
            self._emit_progress('search', 'running', 
                f'[2/3] ✅ Open-source: Found {len(open_papers)} papers', {
                    'open_source_count': len(open_papers)
                })
            
            # STEP 3: Merge, deduplicate, rank by quality
            total_before = len(closed_papers) + len(open_papers)
            self._emit_progress('search', 'running', 
                f'[3/3] 🔗 Merging {total_before} papers → deduplicating → ranking...', {
                    'stage': 'merge',
                    'total_before_dedup': total_before
                })
            
            papers = await self.ingestion_engine.merge_and_deduplicate(open_papers, closed_papers)
            
            self._emit_progress('search', 'running', 
                f'[3/3] 🏆 After dedup: {len(papers)} unique papers. Ranking by citations × recency × OA...', {
                    'unique_count': len(papers)
                })
            
            papers = await self.ingestion_engine.rank_and_filter(papers)
            
            # STEP 4: Take TOP N papers (highest quality)
            top_papers = papers[:required_count]
            
            # Calculate quality metrics
            avg_citations = sum(p.citations or 0 for p in top_papers) / len(top_papers) if top_papers else 0
            oa_percentage = (sum(1 for p in top_papers if p.is_open_access) / len(top_papers) * 100) if top_papers else 0
            avg_year = sum(p.year or 0 for p in top_papers if p.year) / len([p for p in top_papers if p.year]) if top_papers else 0
            
            self._emit_progress('search', 'completed', 
                f'✅ 10X Search complete: Selected TOP {len(top_papers)} papers from {len(papers)} candidates', {
                    'user_requested': required_count,
                    'total_searched': len(papers),
                    'selected': len(top_papers),
                    'avg_citations': round(avg_citations, 1),
                    'oa_percentage': round(oa_percentage, 1),
                    'avg_year': round(avg_year, 1),
                    'top_papers': [
                        {
                            'title': p.title[:80],
                            'year': p.year,
                            'citations': p.citations,
                            'is_oa': p.is_open_access,
                            'source': p.source
                        } for p in top_papers[:10]  # Show top 10
                    ]
                })
            
            return top_papers
            
        except Exception as e:
            self._emit_progress('search', 'failed', f'Search error: {str(e)}')
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    async def download_papers(self, papers: List[Any]) -> Dict[str, str]:
        """
        Stage 2: DOWNLOAD - PDF retrieval with fallbacks
        
        Strategy:
        1. Try open access URLs (Unpaywall, arXiv)
        2. Fallback to Sci-Hub (if enabled)
        3. Fallback to LibGen (if enabled)
        """
        if not papers:
            self._emit_progress('download', 'skipped', 'No papers to download')
            return {}
        
        self._emit_progress('download', 'running', 
            f'Downloading {len(papers)} PDFs with fallbacks...', {
                'total_papers': len(papers)
            })
        
        try:
            pdf_paths = await self.ingestion_engine.download_pdfs(papers)
            
            success_count = len(pdf_paths)
            failed_count = len(papers) - success_count
            
            self._emit_progress('download', 'completed', 
                f'Downloaded {success_count}/{len(papers)} PDFs', {
                    'success_count': success_count,
                    'failed_count': failed_count,
                    'success_rate': success_count / len(papers) if papers else 0
                })
            
            return pdf_paths
            
        except Exception as e:
            self._emit_progress('download', 'failed', f'Download error: {str(e)}')
            logger.error(f"Download failed: {e}", exc_info=True)
            return {}
    
    async def parse_papers(self, papers: List[Any]) -> List[Dict]:
        """
        Stage 3: PARSE - Extract text from PDFs
        
        Methods:
        1. GROBID TEI XML parsing (structured, best quality)
        2. PyMuPDF fallback (fast, good for most papers)
        3. pdfplumber fallback (accurate tables/complex layouts)
        """
        valid_papers = [p for p in papers if hasattr(p, 'pdf_path') and p.pdf_path]
        
        if not valid_papers:
            self._emit_progress('parse', 'skipped', 'No PDFs available to parse')
            return []
        
        self._emit_progress('parse', 'running', 
            f'Parsing {len(valid_papers)} PDFs with fallback extractors...', {
                'total_pdfs': len(valid_papers)
            })
        
        parsed_papers = []
        
        for i, paper in enumerate(valid_papers):
            try:
                pdf_path = Path(paper.pdf_path)
                
                if not pdf_path.exists():
                    logger.warning(f"PDF not found: {pdf_path}")
                    continue
                
                self._emit_progress('parse', 'running', 
                    f'Parsing [{i+1}/{len(valid_papers)}]: {paper.title[:60]}...', {
                        'progress': (i+1) / len(valid_papers) * 100,
                        'current_paper': paper.title[:80]
                    })
                
                # Extract text using fallback extractor
                success, text = extract_text_fast(pdf_path)
                
                if success and text:
                    # Save text file
                    text_filename = pdf_path.stem + '.txt'
                    text_path = self.text_dir / text_filename
                    text_path.write_text(text, encoding='utf-8')
                    
                    parsed_papers.append({
                        'title': paper.title,
                        'text': text,
                        'text_path': str(text_path),
                        'metadata': {
                            'doi': paper.doi,
                            'year': paper.year,
                            'authors': paper.authors[:5] if paper.authors else [],
                            'citations': paper.citations
                        }
                    })
                    
                    logger.info(f"✓ Parsed: {paper.title[:60]} ({len(text)} chars)")
                else:
                    logger.warning(f"✗ Failed to parse: {paper.title[:60]}")
                    
            except Exception as e:
                logger.error(f"Parse error for {paper.title}: {e}")
        
        self._emit_progress('parse', 'completed', 
            f'Successfully parsed {len(parsed_papers)}/{len(valid_papers)} papers', {
                'parsed_count': len(parsed_papers),
                'failed_count': len(valid_papers) - len(parsed_papers),
                'total_characters': sum(len(p['text']) for p in parsed_papers)
            })
        
        return parsed_papers
    
    async def embed_and_index(self, parsed_papers: List[Dict]) -> bool:
        """
        Stage 4: EMBED - Create embeddings and build NEW FAISS index
        
        Model: bge-m3 (multilingual, 1024-dim)
        Chunking: 512 tokens with 128 overlap
        Index: FAISS IndexFlatIP (cosine similarity)
        """
        if not parsed_papers:
            self._emit_progress('embed', 'skipped', 'No parsed papers to embed')
            return False
        
        self._emit_progress('embed', 'running', 
            f'Creating NEW embeddings for {len(parsed_papers)} papers...', {
                'model': 'bge-m3',
                'dimension': 1024,
                'paper_count': len(parsed_papers)
            })
        
        try:
            # Initialize NEW RAG system (hybrid BM25 + dense + RRF + reranking)
            self.rag_system = EnhancedRAG(
                ollama_embed='bge-m3',
                ollama_llm='deepseek-r1:7b',
                use_ollama_fallback=True
            )
            
            # Build index from text files
            self._emit_progress('embed', 'running', 
                'Chunking documents and generating embeddings...', {})
            
            self.rag_system.build_index(text_dir=str(self.text_dir))
            
            # Save NEW index
            self._emit_progress('embed', 'running', 
                'Saving vector index...', {})
            
            self.rag_system.save(str(self.index_dir))
            
            chunk_count = self.rag_system.index.ntotal if self.rag_system.index else 0
            
            self._emit_progress('embed', 'completed',
                f'NEW index built: {chunk_count} chunks from {len(parsed_papers)} papers', {
                    'chunk_count': chunk_count,
                    'paper_count': len(parsed_papers),
                    'index_path': str(self.index_dir)
                })
            
            return True
            
        except Exception as e:
            self._emit_progress('embed', 'failed', f'Embedding error: {str(e)}')
            logger.error(f"Embed failed: {e}", exc_info=True)
            return False
    
    async def retrieve_answer(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Stage 5: RETRIEVE - Query NEW RAG index
        
        Method: Hybrid retrieval (semantic + keyword)
        LLM: DeepSeek-R1:7b (reasoning model)
        Context: Top-K chunks from NEW index
        """
        if not self.rag_system:
            self._emit_progress('retrieve', 'failed', 'RAG system not initialized')
            return {}
        
        self._emit_progress('retrieve', 'running', 
            f'Querying Sanshodhak Retrieval Engine: {query}', {
                'top_k': top_k,
                'model': 'deepseek-r1:7b',
                'index_size': self.rag_system.index.ntotal if self.rag_system.index else 0
            })
        
        try:
            # Execute RAG query on NEW index (hybrid BM25 + dense + RRF)
            answer, sources = self.rag_system.query(query, top_k=top_k, verbose=False,
                                                     use_hybrid=True, use_rerank=False)
            
            # Extract citations from results
            citations = [
                {
                    'file': src['metadata']['file'],
                    'score': src['score'],
                    'snippet': src['text'][:200]
                } for src in sources
            ]
            
            self._emit_progress('retrieve', 'completed',
                f'Generated answer from {len(sources)} sources', {
                    'answer_length': len(answer),
                    'source_count': len(sources),
                    'citations': citations
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'citations': citations
            }
            
        except Exception as e:
            self._emit_progress('retrieve', 'failed', f'Retrieval error: {str(e)}')
            logger.error(f"Retrieve failed: {e}", exc_info=True)
            return {}
    
    async def recommend_resources(self, query: str) -> Dict[str, Any]:
        """
        Stage 6: RECOMMEND - Find related resources
        
        Sources:
        - HuggingFace (models, datasets)
        - GitHub (repositories)
        - Web (tutorials, documentation)
        """
        self._emit_progress('recommend', 'running',
            f'Finding resources for: {query}')
        
        try:
            # Initialize recommender if needed
            if not self.recommender:
                self.recommender = EnhancedResourceRecommender(self.rag_system)
            
            # Get recommendations
            resources = self.recommender.recommend_resources(query)
            
            hf = resources.get('huggingface', {})
            summary = {
                'huggingface_count': len(hf.get('models', [])) + len(hf.get('datasets', [])),
                'github_count': len(resources.get('github_repositories', [])),
                'web_count': len(resources.get('web_resources', [])),
                'pwc_count': len(resources.get('papers_with_code', [])),
            }
            
            self._emit_progress('recommend', 'completed',
                f'Found {sum(summary.values())} resources', summary)
            
            return resources
            
        except Exception as e:
            self._emit_progress('recommend', 'failed', f'Recommendation error: {str(e)}')
            logger.error(f"Recommend failed: {e}", exc_info=True)
            return {}
    
    async def execute_full_pipeline(
        self,
        query: str,
        required_papers: int = 30,
        min_year: Optional[int] = 2020
    ) -> Dict[str, Any]:
        """
        Execute COMPLETE Sanshodhak pipeline with NEW index creation.
        
        Pipeline:
        1. SEARCH: Multi-source paper discovery
        2. DOWNLOAD: PDF retrieval with fallbacks
        3. PARSE: Text extraction
        4. EMBED: Build NEW embeddings and index
        5. RETRIEVE: Query NEW index
        6. RECOMMEND: Find resources
        
        Args:
            query: User's research question/topic
            required_papers: Target number of papers
            min_year: Minimum publication year
            
        Returns:
            Complete results with all stage outputs
        """
        start_time = datetime.now()
        
        # Create session directories
        self._create_session_directories(query)
        
        self._emit_progress('pipeline', 'running', 
            f'Starting Sanshodhak FULL pipeline for: {query}', {
                'session_id': self.session_id,
                'target_papers': required_papers
            })
        
        results = {
            'query': query,
            'session_id': self.session_id,
            'stages': {}
        }
        
        try:
            # Stage 1: SEARCH
            papers = await self.search_papers(query, required_papers, min_year)
            results['stages']['search'] = {
                'papers_found': len(papers),
                'papers': [p.title for p in papers[:10]]
            }
            
            if not papers:
                raise Exception("No papers found in search")
            
            # Stage 2: DOWNLOAD
            pdf_paths = await self.download_papers(papers)
            results['stages']['download'] = {
                'downloaded': len(pdf_paths),
                'total': len(papers)
            }
            
            if not pdf_paths:
                raise Exception("No PDFs downloaded")
            
            # Stage 3: PARSE
            parsed = await self.parse_papers(papers)
            results['stages']['parse'] = {
                'parsed_count': len(parsed),
                'total_chars': sum(len(p['text']) for p in parsed)
            }
            
            if not parsed:
                raise Exception("No papers parsed successfully")
            
            # Stage 4: EMBED (Build NEW index)
            embed_success = await self.embed_and_index(parsed)
            results['stages']['embed'] = {
                'success': embed_success,
                'chunk_count': self.rag_system.index.ntotal if self.rag_system and self.rag_system.index else 0
            }
            
            if not embed_success:
                raise Exception("Failed to build embeddings")
            
            # Stage 5: RETRIEVE (Query NEW index)
            retrieval_result = await self.retrieve_answer(query, top_k=5)
            results['stages']['retrieve'] = retrieval_result
            
            # Stage 6: RECOMMEND
            recommendations = await self.recommend_resources(query)
            results['stages']['recommend'] = recommendations
            
            # Calculate timing
            duration = (datetime.now() - start_time).total_seconds()
            results['duration_seconds'] = duration
            
            self._emit_progress('pipeline', 'completed',
                f'Pipeline completed in {duration:.2f}s - NEW index with {results["stages"]["embed"]["chunk_count"]} chunks created!',
                {'duration': duration})
            
            return results
            
        except Exception as e:
            self._emit_progress('pipeline', 'failed', f'Pipeline error: {str(e)}')
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
