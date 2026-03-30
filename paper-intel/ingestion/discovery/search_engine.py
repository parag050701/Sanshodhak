"""
Search engine orchestrator - coordinates all discovery clients.

Implements two-layer search strategy:
1. Open-source layer (OpenAlex, CORE, SemanticScholar, DOAJ, ArcSieve)
2. Closed-access layer (CrossRef → Unpaywall → arXiv → fallbacks)
"""
import logging
import os
import time
from typing import List, Optional, Dict, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from .openalex_client import OpenAlexClient
from .core_client import COREClient
from .semanticscholar_client import SemanticScholarClient
from .arcsieve_client import ArcSieveClient
from .crossref_client import CrossRefClient
from .unpaywall_client import UnpaywallClient
from .arxiv_client import ArxivClient
from .doaj_client import DOAJClient
from .openaire_client import OpenAIREClient
from .zenodo_client import ZenodoClient
from .plos_client import PLOSClient
from .eric_client import ERICClient
from .oatd_client import OATDClient
from .hal_client import HALClient
from .scielo_client import SciELOClient
from .biorxiv_client import BioRxivMedRxivClient
from .springeropen_client import SpringerOpenClient
from .oai_pmh_client import OAIPMHClient
try:
    from .pubmed_client import PubMedClient
    _PUBMED_AVAILABLE = True
except ImportError:
    _PUBMED_AVAILABLE = False

from .europepmc_client import EuropePMCClient
from .doi_utils import deduplicate_papers, merge_paper_metadata
from .health_report import get_health_tracker, SourceHealthTracker
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


# Source reliability classification
ROBUST_SOURCES = {'openalex', 'arxiv', 'europepmc', 'semanticscholar', 'crossref', 'unpaywall', 'core'}
UNSTABLE_SOURCES = {'oatd', 'openaire_oai', 'scielo', 'zenodo', 'openaire', 'eric', 'hal', 'springeropen', 'plos', 'doaj', 'biorxiv_medrxiv'}
PLACEHOLDER_SOURCES = {'arc_sieve', 'arcsieve'}

# Source configuration: (client_attr, source_name, is_robust)
SOURCE_CONFIG: List[Tuple[str, str, bool]] = [
    ('openalex', 'OpenAlex', True),
    ('semantic_scholar', 'SemanticScholar', True),
    ('arxiv', 'arXiv', True),
    ('europe_pmc', 'EuropePMC', True),
    ('core', 'CORE', True),
    ('crossref', 'CrossRef', True),
    ('unpaywall', 'Unpaywall', True),
    ('doaj', 'DOAJ', False),
    ('openaire', 'OpenAIRE', False),
    ('zenodo', 'Zenodo', False),
    ('plos', 'PLOS', False),
    ('eric', 'ERIC', False),
    ('oatd', 'OATD', False),
    ('hal', 'HAL', False),
    ('scielo', 'SciELO', False),
    ('biorxiv_medrxiv', 'bioRxiv/medRxiv', False),
    ('springeropen', 'SpringerOpen', False),
    ('arc_sieve', 'ArcSieve', False),
]


class SearchEngine:
    """
    Multi-source search orchestrator.

    Coordinates all API clients and implements intelligent search strategies.
    Uses health tracking and circuit breaker patterns for resilience.
    """

    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        core_api_key: Optional[str] = None,
        unpaywall_email: Optional[str] = None,
        enable_core: bool = True,
        openaire_oai_base_url: Optional[str] = None,
        only_robust: bool = False,
    ):
        """
        Initialize search engine with API credentials.

        Args:
            s2_api_key: Semantic Scholar API key (optional)
            core_api_key: CORE API key (optional but recommended)
            unpaywall_email: Email for Unpaywall (required for Unpaywall)
            enable_core: Whether to use CORE (can disable if rate limited)
            only_robust: If True, only use robust sources (skip unstable sources)
        """
        # Health tracker
        self.health = get_health_tracker()
        self.only_robust = only_robust

        # Check env flags
        self.enable_unstable = os.getenv("ENABLE_UNSTABLE_SOURCES", "").lower() == "true"
        self.enable_placeholders = os.getenv("ENABLE_PLACEHOLDER_SOURCES", "").lower() == "true"

        # Open-source clients
        self.openalex = OpenAlexClient(email=unpaywall_email)
        self.core = COREClient(api_key=core_api_key) if enable_core else None
        self.semantic_scholar = SemanticScholarClient(api_key=s2_api_key)
        self.arc_sieve = ArcSieveClient()
        self.doaj = DOAJClient()
        self.openaire = OpenAIREClient()
        self.zenodo = ZenodoClient()
        self.plos = PLOSClient()
        self.eric = ERICClient()
        self.oatd = OATDClient()
        self.hal = HALClient()
        self.scielo = SciELOClient()
        self.biorxiv_medrxiv = BioRxivMedRxivClient()
        self.springeropen = SpringerOpenClient()
        oai_base = openaire_oai_base_url or os.getenv("OPENAIRE_OAI_BASE_URL")
        self.openaire_oai = OAIPMHClient(base_url=oai_base, source_name="openaire_oai") if oai_base else None
        self.europe_pmc = EuropePMCClient()
        self.pubmed = PubMedClient() if _PUBMED_AVAILABLE else None

        # Closed-access clients
        self.crossref = CrossRefClient(email=unpaywall_email)
        self.unpaywall = UnpaywallClient(email=unpaywall_email) if unpaywall_email else None
        self.arxiv = ArxivClient()

        self.enable_core = enable_core

    def _is_source_enabled(self, source_name: str, is_robust: bool) -> bool:
        """Check if a source should be used based on configuration."""
        normalized = source_name.lower().replace("-", "_").replace("/", "_")

        # Check if explicitly disabled by env
        if not self.enable_placeholders and normalized in PLACEHOLDER_SOURCES:
            logger.debug(f"Source {source_name} disabled (placeholder)")
            return False

        # Check if unstable sources are allowed
        if self.only_robust and not is_robust:
            logger.debug(f"Source {source_name} skipped (not robust, only_robust=True)")
            return False

        if not is_robust and not self.enable_unstable:
            # Check individual source env var
            env_var = f"ENABLE_{normalized.upper()}"
            if os.getenv(env_var, "").lower() != "true":
                logger.debug(f"Source {source_name} disabled (unstable, set {env_var}=true or ENABLE_UNSTABLE_SOURCES=true)")
                return False

        # Check circuit breaker
        if not self.health.can_use_source(source_name):
            logger.warning(f"Source {source_name} circuit breaker open, skipping")
            return False

        return True

    def _execute_search(self, client, source_name: str, query: str,
                       limit: int, min_year: Optional[int]) -> SearchResult:
        """Execute search with health tracking."""
        if client is None:
            self.health.record_request(source_name, False, 0, "Client not initialized", None)
            return SearchResult(
                source=source_name,
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error="Client not initialized"
            )

        start_time = time.time()
        try:
            result = client.search(query, limit, min_year=min_year)
            elapsed_ms = (time.time() - start_time) * 1000

            if result.success:
                self.health.record_request(source_name, True, elapsed_ms)
            else:
                self.health.record_request(source_name, False, elapsed_ms, result.error, None)
            return result
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            self.health.record_request(source_name, False, elapsed_ms, error_msg, None)
            return SearchResult(
                source=source_name,
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=error_msg
            )
    
    def search_open_source(
        self,
        query: str,
        limit_per_source: int = 20,
        min_year: Optional[int] = None,
        parallel: bool = True
    ) -> List[PaperMetadata]:
        """
        Search all open-source APIs in parallel with health tracking.

        Args:
            query: Search query
            limit_per_source: Max results per source
            min_year: Minimum publication year
            parallel: Execute searches in parallel (faster)

        Returns:
            Deduplicated list of papers
        """
        logger.info(f"Starting open-source search for: '{query}'")
        logger.info(f"Robust sources only: {self.only_robust}")

        all_papers = []

        # Build list of sources to query
        sources_to_query: List[Tuple[str, Any, bool]] = []

        for attr_name, source_name, is_robust in SOURCE_CONFIG:
            client = getattr(self, attr_name, None)
            if client is None:
                continue
            if self._is_source_enabled(source_name, is_robust):
                sources_to_query.append((source_name, client, is_robust))
            else:
                logger.debug(f"Skipping {source_name} (disabled)")

        # Add special sources
        if self._is_source_enabled('PubMed', True) and self.pubmed:
            sources_to_query.append(('PubMed', self.pubmed, True))
        if self._is_source_enabled('OpenAIRE-OAI', False) and self.openaire_oai:
            sources_to_query.append(('OpenAIRE-OAI', self.openaire_oai, False))

        if parallel:
            open_deadline_s = int(os.getenv("OPEN_SOURCE_DEADLINE_S", "75"))
            max_workers = int(os.getenv("OPEN_SOURCE_WORKERS", "8"))
            # Parallel execution with health tracking
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {
                    executor.submit(
                        self._execute_search,
                        client,
                        source_name,
                        query,
                        limit_per_source,
                        min_year
                    ): source_name
                    for source_name, client, _ in sources_to_query
                }

                completed_futures = set()
                try:
                    for future in as_completed(futures, timeout=open_deadline_s):
                        completed_futures.add(future)
                        source_name = futures[future]
                        try:
                            result: SearchResult = future.result()
                            if result.success:
                                logger.info(f"{source_name}: {result.fetched_count} papers ({result.search_time_ms:.0f}ms)")
                                all_papers.extend(result.papers)
                            else:
                                logger.warning(f"{source_name} failed: {result.error}")
                        except Exception as e:
                            logger.error(f"{source_name} error: {e}")
                            self.health.record_request(source_name, False, 0, f"Exception: {e}", None)
                except FuturesTimeoutError:
                    logger.warning(f"Open-source search deadline reached ({open_deadline_s}s). Proceeding with completed sources.")
                finally:
                    pending = [f for f in futures if f not in completed_futures and not f.done()]
                    for future in pending:
                        source_name = futures[future]
                        future.cancel()
                        self.health.record_request(source_name, False, 0, f"source_deadline_exceeded_{open_deadline_s}s", None)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            # Sequential execution with health tracking
            for source_name, client, _ in sources_to_query:
                result = self._execute_search(client, source_name, query, limit_per_source, min_year)
                if result.success:
                    logger.info(f"{source_name}: {result.fetched_count} papers")
                    all_papers.extend(result.papers)
                else:
                    logger.warning(f"{source_name} failed: {result.error}")

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
            unpaywall_workers = int(os.getenv("UNPAYWALL_WORKERS", "8"))

            def _enrich_one(p: PaperMetadata) -> bool:
                if not p.doi:
                    return False
                try:
                    self.unpaywall.enrich_paper(p)
                    return bool(p.is_open_access)
                except Exception as e:
                    logger.debug(f"Unpaywall enrich failed for {p.doi}: {e}")
                    return False

            with ThreadPoolExecutor(max_workers=unpaywall_workers) as executor:
                futures = [executor.submit(_enrich_one, paper) for paper in papers if paper.doi]
                for future in as_completed(futures):
                    try:
                        if future.result():
                            enriched_count += 1
                    except Exception:
                        pass
            
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

        # Execute both lanes in parallel:
        # 1) Free/open bulk sources
        # 2) CrossRef + Unpaywall enrichment
        unified_deadline_s = int(os.getenv("UNIFIED_SEARCH_DEADLINE_S", "120"))
        open_papers: List[PaperMetadata] = []
        closed_papers: List[PaperMetadata] = []

        executor = ThreadPoolExecutor(max_workers=2)
        try:
            future_open = executor.submit(
                self.search_open_source,
                query,
                max(1, limit // 2),
                min_year,
                True,
            )
            future_closed = executor.submit(
                self.search_closed_access,
                query,
                limit,
                min_year,
                None,
                True,
            )

            lane_futures = {
                future_open: "open_lane",
                future_closed: "closed_lane",
            }
            completed = set()

            try:
                for future in as_completed(lane_futures, timeout=unified_deadline_s):
                    completed.add(future)
                    lane = lane_futures[future]
                    try:
                        lane_result = future.result()
                        if lane == "open_lane":
                            open_papers = lane_result
                        else:
                            closed_papers = lane_result
                    except Exception as e:
                        logger.warning(f"{lane} failed: {e}")
            except FuturesTimeoutError:
                logger.warning(f"Unified search deadline reached ({unified_deadline_s}s). Returning partial results.")
            finally:
                for future in lane_futures:
                    if future not in completed and not future.done():
                        future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        
        # Merge and deduplicate
        all_papers = open_papers + closed_papers
        unique_papers = deduplicate_papers(all_papers)
        
        logger.info(f"Unified search: {len(all_papers)} → {len(unique_papers)} unique papers")
        
        # Rank papers
        ranked_papers = self._rank_papers(unique_papers, prefer_open_access)

        # Print health report
        self.health.print_health_report()

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
