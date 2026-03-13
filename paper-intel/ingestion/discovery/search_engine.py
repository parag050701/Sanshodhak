"""
Search engine orchestrator - coordinates all discovery clients.

Implements two-layer search strategy:
1. Open-source layer (OpenAlex, CORE, SemanticScholar, DOAJ, ArcSieve)
2. Closed-access layer (CrossRef → Unpaywall → arXiv → fallbacks)
"""
import logging
import asyncio
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .openalex_client import OpenAlexClient
from .core_client import COREClient
from .semanticscholar_client import SemanticScholarClient
from .arcsieve_client import ArcSieveClient
from .crossref_client import CrossRefClient
from .unpaywall_client import UnpaywallClient
from .arxiv_client import ArxivClient
from .doaj_client import DOAJClient
try:
    from .pubmed_client import PubMedClient
    _PUBMED_AVAILABLE = True
except ImportError:
    _PUBMED_AVAILABLE = False
    
from .europepmc_client import EuropePMCClient
from .doi_utils import deduplicate_papers, merge_paper_metadata
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Multi-source search orchestrator.
    
    Coordinates all API clients and implements intelligent search strategies.
    """
    
    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        core_api_key: Optional[str] = None,
        unpaywall_email: Optional[str] = None,
        enable_core: bool = True
    ):
        """
        Initialize search engine with API credentials.
        
        Args:
            s2_api_key: Semantic Scholar API key (optional)
            core_api_key: CORE API key (optional but recommended)
            unpaywall_email: Email for Unpaywall (required for Unpaywall)
            enable_core: Whether to use CORE (can disable if rate limited)
        """
        # Open-source clients
        self.openalex = OpenAlexClient(email=unpaywall_email)
        self.core = COREClient(api_key=core_api_key) if enable_core else None
        self.semantic_scholar = SemanticScholarClient(api_key=s2_api_key)
        self.arc_sieve = ArcSieveClient()
        self.doaj = DOAJClient()
        self.europe_pmc = EuropePMCClient()
        self.pubmed = PubMedClient() if _PUBMED_AVAILABLE else None

        # Closed-access clients
        self.crossref = CrossRefClient(email=unpaywall_email)
        self.unpaywall = UnpaywallClient(email=unpaywall_email) if unpaywall_email else None
        self.arxiv = ArxivClient()

        self.enable_core = enable_core
    
    def search_open_source(
        self,
        query: str,
        limit_per_source: int = 20,
        min_year: Optional[int] = None,
        parallel: bool = True
    ) -> List[PaperMetadata]:
        """
        Search all open-source APIs in parallel.
        
        Sources: OpenAlex, CORE, SemanticScholar, ArcSieve
        
        Args:
            query: Search query
            limit_per_source: Max results per source
            min_year: Minimum publication year
            parallel: Execute searches in parallel (faster)
            
        Returns:
            Deduplicated list of papers
        """
        logger.info(f"Starting open-source search for: '{query}'")
        
        all_papers = []
        
        if parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        self.openalex.search,
                        query,
                        limit_per_source,
                        min_year=min_year
                    ): 'OpenAlex',
                    executor.submit(
                        self.semantic_scholar.search,
                        query,
                        limit_per_source,
                        min_year=min_year
                    ): 'SemanticScholar',
                    executor.submit(
                        self.arc_sieve.search,
                        query,
                        limit_per_source
                    ): 'ArcSieve',
                    executor.submit(
                        self.doaj.search,
                        query,
                        limit_per_source,
                        min_year=min_year
                    ): 'DOAJ',
                }

                # Add CORE if enabled
                if self.enable_core and self.core:
                    futures[executor.submit(
                        self.core.search,
                        query,
                        limit_per_source,
                        min_year=min_year
                    )] = 'CORE'

                # Add PubMed if available
                if self.pubmed:
                    futures[executor.submit(
                        self.pubmed.search,
                        query,
                        limit_per_source,
                        min_year=min_year
                    )] = 'PubMed'
                    
                # Add arXiv explicitly in open-source layer
                futures[executor.submit(
                    self.arxiv.search,
                    query,
                    limit_per_source,
                )] = 'arXiv'
                
                # Add Europe PMC explicitly in open-source layer
                futures[executor.submit(
                    self.europe_pmc.search,
                    query,
                    limit_per_source,
                    min_year=min_year
                )] = 'EuropePMC'
                
                for future in as_completed(futures):
                    source_name = futures[future]
                    try:
                        result: SearchResult = future.result(timeout=60)
                        if result.success:
                            logger.info(f"{source_name}: {result.fetched_count} papers ({result.search_time_ms:.0f}ms)")
                            all_papers.extend(result.papers)
                        else:
                            logger.warning(f"{source_name} failed: {result.error}")
                    except Exception as e:
                        logger.error(f"{source_name} error: {e}")
        else:
            # Sequential execution
            sources = [
                ('OpenAlex', self.openalex),
                ('SemanticScholar', self.semantic_scholar),
                ('ArcSieve', self.arc_sieve),
            ]
            
            if self.enable_core and self.core:
                sources.append(('CORE', self.core))
                
            sources.append(('arXiv', self.arxiv))
            sources.append(('EuropePMC', self.europe_pmc))
            
            for source_name, client in sources:
                try:
                    result = client.search(query, limit_per_source, min_year=min_year)
                    if result.success:
                        logger.info(f"{source_name}: {result.fetched_count} papers")
                        all_papers.extend(result.papers)
                    else:
                        logger.warning(f"{source_name} failed: {result.error}")
                except Exception as e:
                    logger.error(f"{source_name} error: {e}")
        
        # Deduplicate
        unique_papers = deduplicate_papers(all_papers)
        logger.info(f"Open-source search: {len(all_papers)} → {len(unique_papers)} after dedup")
        
        return unique_papers
    
    def search_closed_access(
        self,
        query: str,
        limit: int = 50,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        enrich_with_unpaywall: bool = True
    ) -> List[PaperMetadata]:
        """
        Search closed-access sources (CrossRef) and enrich with OA info.
        
        Strategy:
        1. Search CrossRef for metadata
        2. For each DOI, check Unpaywall for OA PDF
        3. Check arXiv for preprints
        
        Args:
            query: Search query
            limit: Max results
            min_year: Minimum year
            max_year: Maximum year
            enrich_with_unpaywall: Check Unpaywall for OA PDFs
            
        Returns:
            List of papers with metadata (and OA URLs if available)
        """
        logger.info(f"Starting closed-access search (CrossRef) for: '{query}'")
        
        # Search CrossRef
        result = self.crossref.search(
            query,
            limit=limit,
            min_year=min_year,
            max_year=max_year
        )
        
        if not result.success:
            logger.warning(f"CrossRef search failed: {result.error}")
            return []
        
        papers = result.papers
        logger.info(f"CrossRef: {len(papers)} papers found")
        
        # Enrich with Unpaywall
        if enrich_with_unpaywall and self.unpaywall:
            logger.info("Enriching with Unpaywall OA info...")
            enriched_count = 0
            
            for paper in papers:
                if paper.doi:
                    try:
                        self.unpaywall.enrich_paper(paper)
                        if paper.is_open_access:
                            enriched_count += 1
                    except Exception as e:
                        logger.debug(f"Unpaywall enrich failed for {paper.doi}: {e}")
            
            logger.info(f"Unpaywall: {enriched_count}/{len(papers)} papers have OA PDFs")
        
        # Final deduplication
        papers = deduplicate_papers(papers)
        
        return papers
    
    def search_unified(
        self,
        query: str,
        limit: int = 50,
        min_year: Optional[int] = None,
        prefer_open_access: bool = True
    ) -> List[PaperMetadata]:
        """
        Unified search across all sources (both layers).
        
        Combines open-source and closed-access results.
        
        Args:
            query: Search query
            limit: Target number of papers
            min_year: Minimum publication year
            prefer_open_access: Boost OA papers in ranking
            
        Returns:
            Deduplicated and ranked list of papers
        """
        logger.info(f"Starting unified search for: '{query}' (target: {limit} papers)")
        
        # Execute both layers
        open_papers = self.search_open_source(
            query,
            limit_per_source=limit // 2,
            min_year=min_year
        )
        
        closed_papers = self.search_closed_access(
            query,
            limit=limit,
            min_year=min_year
        )
        
        # Merge and deduplicate
        all_papers = open_papers + closed_papers
        unique_papers = deduplicate_papers(all_papers)
        
        logger.info(f"Unified search: {len(all_papers)} → {len(unique_papers)} unique papers")
        
        # Rank papers
        ranked_papers = self._rank_papers(unique_papers, prefer_open_access)
        
        return ranked_papers[:limit]
    
    def _rank_papers(
        self,
        papers: List[PaperMetadata],
        prefer_open_access: bool = True
    ) -> List[PaperMetadata]:
        """
        Rank papers by composite quality-recency-accessibility score.

        Uses the composite_rank_score() function from query_expander which
        combines log-normalised citations, exponential temporal decay
        (half-life ≈ 5 years), and an open-access accessibility bonus.

        Score = 0.5 * norm_citations + 0.3 * temporal_decay + 0.2 * oa_bonus
        """
        try:
            from query_expander import composite_rank_score
            use_composite = True
        except ImportError:
            use_composite = False

        def compute_rank_score(paper: PaperMetadata) -> float:
            if use_composite:
                return composite_rank_score(
                    citations=paper.citations,
                    year=paper.year,
                    is_open_access=paper.is_open_access and prefer_open_access,
                ) + paper.score * 0.1
            else:
                # Legacy fallback (no query_expander installed)
                import math
                s = 0.0
                if paper.citations > 0:
                    s += math.log10(min(paper.citations, 10000) + 1) * 2.0
                if paper.year:
                    from datetime import datetime
                    years_ago = datetime.now().year - paper.year
                    if years_ago <= 5:
                        s += (5 - years_ago) * 0.5
                if prefer_open_access and paper.is_open_access:
                    s += 2.0
                s += paper.score * 0.5
                return s

        papers_with_scores = [(p, compute_rank_score(p)) for p in papers]
        papers_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in papers_with_scores]
    
    def get_recommendations(
        self,
        paper: PaperMetadata,
        limit: int = 10
    ) -> List[PaperMetadata]:
        """
        Get recommended/related papers.
        
        Uses Semantic Scholar's recommendation API.
        """
        if not paper.s2_id:
            logger.warning("Cannot get recommendations without S2 paper ID")
            return []
        
        try:
            recommendations = self.semantic_scholar.get_recommendations(
                paper.s2_id,
                limit=limit
            )
            logger.info(f"Got {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"Recommendations error: {e}")
            return []
    
    def close(self):
        """Close all client connections."""
        clients = [
            self.openalex, self.core, self.semantic_scholar,
            self.doaj, self.arc_sieve, self.crossref,
            self.unpaywall, self.arxiv, self.europe_pmc
        ]
        
        for client in clients:
            if client:
                try:
                    client.close()
                except:
                    pass
