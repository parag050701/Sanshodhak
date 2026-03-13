"""
Main Ingestion Engine - orchestrates the complete pipeline.

This is the top-level API for the Sanshodhak ingestion system.
"""
import logging
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter

from .models import PaperMetadata, DownloadResult
from .discovery import SearchEngine, PDFDownloader

logger = logging.getLogger(__name__)


class IngestionEngine:
    """
    Top-level orchestrator for paper discovery, download, and preprocessing.
    
    Complete pipeline:
    1. Search (open-source + closed-access layers)
    2. Merge & deduplicate
    3. Rank & filter
    4. Expand search if needed
    5. Download PDFs
    6. (Optional) Preprocess with GROBID
    
    Usage:
        engine = IngestionEngine("machine learning", required_count=20)
        results = await engine.run()
    """
    
    def __init__(
        self,
        query: str,
        required_count: int = 20,
        output_dir: str = "paper-intel/papers",
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        # API keys
        s2_api_key: Optional[str] = None,
        core_api_key: Optional[str] = None,
        unpaywall_email: Optional[str] = None,
        # Options
        enable_core: bool = True,
        enable_scihub: bool = False,
        enable_libgen: bool = False,
        prefer_open_access: bool = True,
        max_iterations: int = 3,
        min_success_rate: float = 0.7
    ):
        """
        Initialize ingestion engine.
        
        Args:
            query: Research topic/query
            required_count: Target number of papers
            output_dir: Where to save PDFs
            min_year: Minimum publication year
            max_year: Maximum publication year
            s2_api_key: Semantic Scholar API key
            core_api_key: CORE API key
            unpaywall_email: Email for Unpaywall
            enable_core: Use CORE API
            enable_scihub: Enable Sci-Hub fallback (use responsibly)
            enable_libgen: Enable LibGen fallback
            prefer_open_access: Prefer OA papers in ranking
            max_iterations: Max search iterations for expansion
            min_success_rate: Min download success rate before expanding
        """
        self.query = query
        self.required_count = required_count
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_year = min_year
        self.max_year = max_year
        self.prefer_open_access = prefer_open_access
        self.max_iterations = max_iterations
        self.min_success_rate = min_success_rate
        
        # Initialize search engine
        self.search_engine = SearchEngine(
            s2_api_key=s2_api_key,
            core_api_key=core_api_key,
            unpaywall_email=unpaywall_email,
            enable_core=enable_core
        )
        
        # Initialize PDF downloader
        self.pdf_downloader = PDFDownloader(
            output_dir=str(self.output_dir),
            enable_scihub=enable_scihub,
            enable_libgen=enable_libgen
        )
        
        # State tracking
        self.all_papers: List[PaperMetadata] = []
        self.download_results: List[DownloadResult] = []
        self.iteration = 0
        
        logger.info(f"Initialized IngestionEngine for query: '{query}'")
        logger.info(f"Target: {required_count} papers, Output: {output_dir}")
    
    async def search_open_source(self) -> List[PaperMetadata]:
        """
        Search open-source layer (OpenAlex, CORE, S2, DOAJ, ArcSieve).
        
        Returns:
            List of papers from open-source APIs
        """
        logger.info("\n[SEARCH] Open-Source Layer...")
        
        papers = self.search_engine.search_open_source(
            query=self.query,
            limit_per_source=self.required_count,
            min_year=self.min_year,
            parallel=True
        )
        
        logger.info(f"Open-source layer: {len(papers)} unique papers")
        return papers
    
    async def search_closed_access(self) -> List[PaperMetadata]:
        """
        Search closed-access layer (CrossRef → Unpaywall → arXiv).
        ALWAYS searches minimum 100 papers in CrossRef regardless of user input.
        
        Returns:
            List of papers from closed-access APIs + OA enrichment
        """
        logger.info("\n[SEARCH] Closed-Access Layer...")
        
        # ALWAYS search minimum 100 papers in CrossRef
        search_limit = max(100, self.required_count * 10)
        logger.info(f"Strategy: Searching {search_limit} papers (minimum 100)")
        
        papers = self.search_engine.search_closed_access(
            query=self.query,
            limit=search_limit,
            min_year=self.min_year,
            max_year=self.max_year,
            enrich_with_unpaywall=True
        )
        
        logger.info(f"Closed-access layer: {len(papers)} papers")
        return papers
    
    async def merge_and_deduplicate(
        self,
        open_src: List[PaperMetadata],
        closed_src: List[PaperMetadata]
    ) -> List[PaperMetadata]:
        """
        Merge results from both layers and deduplicate.
        
        Args:
            open_src: Papers from open-source APIs
            closed_src: Papers from closed-access APIs
            
        Returns:
            Deduplicated list
        """
        logger.info("\n[MERGE] Deduplicating...")
        
        from .discovery.doi_utils import deduplicate_papers
        
        all_papers = open_src + closed_src
        unique_papers = deduplicate_papers(all_papers)
        
        logger.info(f"Merged: {len(all_papers)} → {len(unique_papers)} unique")
        
        return unique_papers
    
    async def rank_and_filter(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """
        Rank papers by quality metrics and filter to required count.
        
        Ranking factors:
        - Citation count
        - Recency
        - Open access availability
        - Source relevance
        
        Returns:
            Ranked and filtered list
        """
        logger.info("\n[RANK] Ranking papers...")
        
        ranked = self.search_engine._rank_papers(papers, self.prefer_open_access)
        
        # Log top papers
        logger.info(f"Top {min(5, len(ranked))} papers:")
        for i, paper in enumerate(ranked[:5], 1):
            logger.info(f"  {i}. {paper.title[:60]} ({paper.year}, {paper.citations} cites, OA={paper.is_open_access})")
        
        return ranked
    
    async def expand_search_if_needed(self, current_papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """
        Expand search if we don't have enough papers.
        
        Strategies:
        1. Broaden year range
        2. Use recommended papers from S2
        3. Try related keywords
        
        Args:
            current_papers: Current paper list
            
        Returns:
            Expanded paper list
        """
        if len(current_papers) >= self.required_count:
            return current_papers
        
        logger.info(f"\n[EXPAND] Need {self.required_count - len(current_papers)} more papers...")
        
        new_papers = []
        
        # Strategy 1: Broaden year range
        if self.min_year and self.min_year > 2010:
            broader_min_year = self.min_year - 5
            logger.info(f"Expanding year range: {self.min_year} → {broader_min_year}")
            
            expanded = self.search_engine.search_unified(
                query=self.query,
                limit=self.required_count,
                min_year=broader_min_year,
                prefer_open_access=self.prefer_open_access
            )
            new_papers.extend(expanded)
        
        # Strategy 2: Get recommendations from existing papers
        if current_papers and len(new_papers) < self.required_count:
            logger.info("Getting recommendations from top papers...")
            top_papers = current_papers[:3]  # Use top 3
            
            for paper in top_papers:
                if paper.s2_id:
                    recs = self.search_engine.get_recommendations(paper, limit=10)
                    new_papers.extend(recs)
                    
                    if len(new_papers) >= self.required_count:
                        break
        
        # Strategy 3: Try related queries (simple keyword expansion)
        if len(new_papers) < self.required_count:
            related_queries = self._generate_related_queries(self.query)
            logger.info(f"Trying {len(related_queries)} related queries...")
            
            for related_query in related_queries:
                results = self.search_engine.search_unified(
                    query=related_query,
                    limit=self.required_count // 2,
                    min_year=self.min_year,
                    prefer_open_access=self.prefer_open_access
                )
                new_papers.extend(results)
                
                if len(new_papers) >= self.required_count:
                    break
        
        # Merge with existing papers and deduplicate
        from .discovery.doi_utils import deduplicate_papers
        all_papers = current_papers + new_papers
        unique_papers = deduplicate_papers(all_papers)
        
        logger.info(f"After expansion: {len(current_papers)} → {len(unique_papers)} papers")
        
        return unique_papers
    
    def _generate_related_queries(self, query: str) -> List[str]:
        """Generate related search queries."""
        # Simple keyword expansion (can be enhanced with LLM)
        variations = [
            f"{query} review",
            f"{query} survey",
            f"{query} applications",
            f"{query} methods",
        ]
        return variations[:2]  # Limit to 2 to avoid over-expansion
    
    async def download_pdfs(self, papers: List[PaperMetadata]) -> Dict[str, str]:
        """
        Download PDFs for papers.
        
        Args:
            papers: Papers to download
            
        Returns:
            Dict mapping paper_id -> pdf_path
        """
        logger.info(f"\n[DOWNLOAD] Downloading {len(papers)} PDFs...")
        
        download_results = self.pdf_downloader.download_batch(
            papers=papers,
            parallel=True,
            max_workers=5
        )
        
        self.download_results = download_results
        
        # Update paper metadata with paths
        pdf_paths = {}
        for result in download_results:
            if result.success:
                pdf_paths[result.paper_id] = result.pdf_path
        
        # Map paths back to papers
        for paper in papers:
            paper_id = paper.get_primary_id() or paper.title
            if paper_id in pdf_paths:
                paper.pdf_path = pdf_paths[paper_id]
        
        success_count = sum(1 for r in download_results if r.success)
        logger.info(f"Downloaded: {success_count}/{len(papers)} PDFs")
        
        return pdf_paths
    
    async def run(self) -> Dict[str, Any]:
        """
        Run complete ingestion pipeline.
        
        Returns:
            Results dictionary with papers, downloads, metadata
        """
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("STARTING SANSHODHAK INGESTION PIPELINE")
        logger.info(f"Query: {self.query}")
        logger.info(f"Target: {self.required_count} papers")
        logger.info("="*80)
        
        # Main pipeline loop
        for self.iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {self.iteration}/{self.max_iterations}")
            logger.info("="*80)
            
            # Search both layers
            open_papers = await self.search_open_source()
            closed_papers = await self.search_closed_access()
            
            # Merge and deduplicate
            papers = await self.merge_and_deduplicate(open_papers, closed_papers)
            
            # Rank and filter
            papers = await self.rank_and_filter(papers)
            
            # Take top N
            papers_to_download = papers[:self.required_count * 2]  # Get extra for buffer
            
            # Download PDFs
            pdf_paths = await self.download_pdfs(papers_to_download)
            
            # Check success rate
            success_count = len(pdf_paths)
            total_attempted = len(papers_to_download)
            success_rate = success_count / total_attempted if total_attempted > 0 else 0
            
            logger.info(f"\nIteration {self.iteration} Results:")
            logger.info(f"  Papers found: {len(papers)}")
            logger.info(f"  PDFs downloaded: {success_count}/{total_attempted}")
            logger.info(f"  Success rate: {success_rate:.1%}")
            
            # Check if we should continue
            if success_count >= self.required_count and success_rate >= self.min_success_rate:
                logger.info(f"✓ Target achieved! ({success_count} papers, {success_rate:.1%} success rate)")
                break
            
            # Expand search if needed and not last iteration
            if self.iteration < self.max_iterations:
                papers = await self.expand_search_if_needed(papers)
                self.all_papers = papers
        
        # Final results
        elapsed = time.time() - start_time
        
        successful_papers = [p for p in self.all_papers if p.pdf_path][:self.required_count]
        
        # Count sources
        source_counts = Counter(p.source for p in successful_papers)
        
        results = {
            'query': self.query,
            'total_found': len(self.all_papers),
            'total_downloaded': len(successful_papers),
            'target': self.required_count,
            'success_rate': len(successful_papers) / len(self.all_papers) if self.all_papers else 0,
            'iterations': self.iteration,
            'elapsed_time_seconds': elapsed,
            'papers': [p.to_dict() for p in successful_papers],
            'pdf_paths': {p.get_primary_id(): p.pdf_path for p in successful_papers if p.pdf_path},
            'sources_used': dict(source_counts),
            'metadata': {
                'output_dir': str(self.output_dir),
                'min_year': self.min_year,
                'max_year': self.max_year,
                'prefer_open_access': self.prefer_open_access,
            }
        }
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        output_file = self.output_dir / "ingestion_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Results saved to: {output_file}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print pipeline summary."""
        logger.info("\n" + "="*80)
        logger.info("INGESTION PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Query: {results['query']}")
        logger.info(f"Papers found: {results['total_found']}")
        logger.info(f"PDFs downloaded: {results['total_downloaded']}/{results['target']}")
        logger.info(f"Success rate: {results['success_rate']:.1%}")
        logger.info(f"Time elapsed: {results['elapsed_time_seconds']:.1f}s")
        logger.info(f"\nSources used:")
        for source, count in results['sources_used'].items():
            logger.info(f"  {source}: {count} papers")
        logger.info(f"\nOutput directory: {results['metadata']['output_dir']}")
        logger.info("="*80)
    
    def close(self):
        """Clean up resources."""
        self.search_engine.close()
        self.pdf_downloader.close()


# Convenience function for quick runs
async def ingest_papers(
    query: str,
    count: int = 20,
    output_dir: str = "paper-intel/papers",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick ingestion function.
    
    Args:
        query: Research topic
        count: Number of papers
        output_dir: Output directory
        **kwargs: Additional IngestionEngine options
        
    Returns:
        Results dictionary
    """
    engine = IngestionEngine(query, count, output_dir, **kwargs)
    try:
        results = await engine.run()
        return results
    finally:
        engine.close()
