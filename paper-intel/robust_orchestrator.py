"""
ROBUST Research Orchestrator - ACTUALLY Downloads PDFs!
Uses CrossRef for metadata + parallel downloads from multiple sources
"""

import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.discovery import (
    CrossRefClient, UnpaywallClient, ArxivClient, OpenAlexClient, 
    COREClient, SciHubClient, AnnasArchiveClient
)
from ingestion.models import PaperMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class RobustOrchestrator:
    """Downloads papers PROPERLY with aggressive multi-source fallback."""
    
    def __init__(self, output_dir: str = "research_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize ALL clients
        self.crossref = CrossRefClient(email="research@sanshodhak.org")
        self.unpaywall = UnpaywallClient(email="research@sanshodhak.org")
        self.arxiv = ArxivClient()
        self.openalex = OpenAlexClient(email="research@sanshodhak.org")
        self.core = COREClient(api_key="fU7PRKOIE5Nz3roXDSvCwVZ46Wd2TpYi")
        self.scihub = SciHubClient()
        self.annas = AnnasArchiveClient()
        
        logger.info("✅ All download clients initialized")
    
    def search_crossref(self, topic: str, count: int, min_year: int = 2020):
        """Search CrossRef for papers with DOIs."""
        print(f"\n🔍 Searching CrossRef for: {topic}")
        result = self.crossref.search(topic, limit=count, min_year=min_year)
        
        if not result.success:
            logger.error(f"CrossRef search failed: {result.error}")
            return []
        
        print(f"✅ Found {len(result.papers)} papers with DOIs")
        return result.papers
    
    def download_single_paper(self, paper: PaperMetadata, index: int, total: int):
        """Download ONE paper using ALL available sources."""
        print(f"\n[{index}/{total}] {paper.title[:60]}...")
        
        filename = f"{index:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Already exists: {filename}")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # Source priority order
        sources = [
            ('OpenAlex', self._try_openalex),
            ('CORE', self._try_core),
            ('arXiv', self._try_arxiv),
            ('Unpaywall', self._try_unpaywall),
            ('Sci-Hub', self._try_scihub),
            ("Anna's Archive", self._try_annas),
        ]
        
        for source_name, download_func in sources:
            try:
                pdf_content = download_func(paper)
                
                if pdf_content and len(pdf_content) > 10000:
                    # Validate PDF
                    if pdf_content[:4] == b'%PDF':
                        filepath.write_bytes(pdf_content)
                        size_mb = len(pdf_content) / (1024 * 1024)
                        print(f"   ✅ Downloaded via {source_name} ({size_mb:.2f} MB)")
                        return {'paper': paper, 'path': str(filepath), 'source': source_name}
                    
            except Exception as e:
                logger.debug(f"   {source_name} failed: {e}")
        
        print(f"   ❌ All sources failed")
        return None
    
    def _try_openalex(self, paper: PaperMetadata):
        """Try OpenAlex for OA PDF."""
        if not paper.doi:
            return None
        
        # Search by DOI
        result = self.openalex.search(f"doi:{paper.doi}", limit=1)
        if result.success and result.papers:
            oa_paper = result.papers[0]
            if oa_paper.pdf_url:
                import requests
                resp = requests.get(oa_paper.pdf_url, timeout=30, allow_redirects=True)
                if resp.status_code == 200:
                    return resp.content
        return None
    
    def _try_core(self, paper: PaperMetadata):
        """Try CORE for OA PDF."""
        # Search by title
        result = self.core.search(paper.title[:100], limit=1)
        if result.success and result.papers:
            core_paper = result.papers[0]
            if core_paper.pdf_url:
                import requests
                resp = requests.get(core_paper.pdf_url, timeout=30, allow_redirects=True)
                if resp.status_code == 200:
                    return resp.content
        return None
    
    def _try_arxiv(self, paper: PaperMetadata):
        """Try arXiv for preprints."""
        # Search by title
        result = self.arxiv.search(paper.title[:100], limit=1)
        if result.success and result.papers:
            arxiv_paper = result.papers[0]
            if arxiv_paper.pdf_url:
                import requests
                resp = requests.get(arxiv_paper.pdf_url, timeout=30, allow_redirects=True)
                if resp.status_code == 200:
                    return resp.content
        return None
    
    def _try_unpaywall(self, paper: PaperMetadata):
        """Try Unpaywall for OA version."""
        if not paper.doi:
            return None
        
        info = self.unpaywall.get_pdf_url(paper.doi)
        if info and info.get('pdf_url'):
            import requests
            resp = requests.get(info['pdf_url'], timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                return resp.content
        return None
    
    def _try_scihub(self, paper: PaperMetadata):
        """Try Sci-Hub mirrors."""
        if not paper.doi:
            return None
        
        result = self.scihub.get_pdf(paper.doi)
        if result:
            pdf_content, mirror = result
            return pdf_content
        return None
    
    def _try_annas(self, paper: PaperMetadata):
        """Try Anna's Archive (requires MD5)."""
        # Note: Anna's requires MD5 hash which we don't have
        # This is a placeholder for future enhancement
        return None
    
    def _safe_filename(self, paper: PaperMetadata):
        """Generate safe filename."""
        import re
        author = paper.authors[0].split()[-1] if paper.authors else "Unknown"
        author = re.sub(r'[^\w]', '', author)[:15]
        year = paper.year or "0000"
        return f"{author}_{year}.pdf"
    
    def conduct_research(self, topic: str, num_papers: int = 20, min_year: int = 2020):
        """
        Main research pipeline:
        1. Search CrossRef for DOIs
        2. Download from ALL sources in parallel
        3. Generate report
        """
        print("\n" + "="*80)
        print("🔬 ROBUST RESEARCH ORCHESTRATOR")
        print("="*80)
        print(f"\n📋 Topic: {topic}")
        print(f"📊 Target: {num_papers} papers")
        print(f"📅 Min Year: {min_year}")
        print("\n" + "-"*80)
        
        # Create topic directory
        topic_slug = topic.lower().replace(' ', '_')[:50]
        topic_dir = self.output_dir / topic_slug
        topic_dir.mkdir(exist_ok=True)
        self.output_dir = topic_dir  # Update for downloads
        
        # PHASE 1: Get metadata from CrossRef
        print("\n📋 PHASE 1: CrossRef Metadata Search")
        print("-"*80)
        papers = self.search_crossref(topic, num_papers, min_year)
        
        if not papers:
            print("\n❌ No papers found!")
            return
        
        # PHASE 2: Parallel PDF downloads
        print(f"\n📥 PHASE 2: Multi-Source PDF Download")
        print("-"*80)
        print("Sources: OpenAlex → CORE → arXiv → Unpaywall → Sci-Hub → Anna's")
        print("-"*80)
        
        downloaded = []
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, paper in enumerate(papers, 1):
                future = executor.submit(
                    self.download_single_paper, 
                    paper, idx, len(papers)
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    if result:
                        downloaded.append(result)
                except Exception as e:
                    logger.error(f"Download error: {e}")
        
        # PHASE 3: Generate report
        print(f"\n📝 PHASE 3: Generate Report")
        print("-"*80)
        
        success_rate = len(downloaded) / len(papers) * 100
        
        # Save metadata
        metadata = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'target_papers': num_papers,
            'found_papers': len(papers),
            'downloaded_papers': len(downloaded),
            'success_rate': f"{success_rate:.1f}%",
            'papers': [
                {
                    'title': d['paper'].title,
                    'authors': d['paper'].authors,
                    'year': d['paper'].year,
                    'doi': d['paper'].doi,
                    'file': Path(d['path']).name,
                    'source': d['source']
                }
                for d in downloaded
            ]
        }
        
        metadata_file = topic_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        print(f"✅ Metadata: {metadata_file}")
        
        # Generate report
        report = f"""# Research Report: {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Papers Found:** {len(papers)}
- **PDFs Downloaded:** {len(downloaded)}
- **Success Rate:** {success_rate:.1f}%
- **Min Year:** {min_year}

## Downloaded Papers

"""
        for d in downloaded:
            p = d['paper']
            report += f"### {p.title}\n\n"
            report += f"**Authors:** {', '.join(p.authors[:3])}\n\n"
            report += f"**Year:** {p.year} | **DOI:** {p.doi}\n\n"
            report += f"**File:** `{Path(d['path']).name}` | **Source:** {d['source']}\n\n"
            report += "---\n\n"
        
        report_file = topic_dir / "RESEARCH_REPORT.md"
        report_file.write_text(report)
        print(f"✅ Report: {report_file}")
        
        # Final summary
        print("\n" + "="*80)
        print("✅ RESEARCH COMPLETE!")
        print("="*80)
        print(f"\n📁 Location: {topic_dir.absolute()}")
        print(f"📄 Papers Found: {len(papers)}")
        print(f"📥 PDFs Downloaded: {len(downloaded)}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"📝 Files: {len(list(topic_dir.glob('*.pdf')))} PDFs")
        print("\n" + "="*80 + "\n")


def main():
    """CLI interface."""
    if len(sys.argv) < 2:
        print("\nUsage: python robust_orchestrator.py \"topic\" [num_papers] [min_year]")
        print("\nExample: python robust_orchestrator.py \"quantum computing\" 10 2023")
        sys.exit(1)
    
    topic = sys.argv[1]
    num_papers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2020
    
    orchestrator = RobustOrchestrator()
    orchestrator.conduct_research(topic, num_papers, min_year)


if __name__ == "__main__":
    main()
