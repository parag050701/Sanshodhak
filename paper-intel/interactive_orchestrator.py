#!/usr/bin/env python3
"""
Interactive Research Orchestrator - Following test7.5-wrkin.py Flow
1. User inputs topic + number of papers
2. Search CrossRef for metadata/DOIs
3. Forward to Unpaywall for max OA papers
4. Simultaneously search open-source: arXiv, CORE, OpenAlex
5. Collect all PDFs
"""

import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import Counter
import json

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.discovery import (
    CrossRefClient, UnpaywallClient, ArxivClient, 
    OpenAlexClient, COREClient, SciHubClient
)
from ingestion.models import PaperMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class InteractiveOrchestrator:
    """
    Interactive orchestrator following test7.5-wrkin.py flow:
    CrossRef → Unpaywall → (arXiv + CORE + OpenAlex) simultaneous
    """
    
    def __init__(self, output_dir: str = "research_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize clients
        self.crossref = CrossRefClient(email="research@sanshodhak.org")
        self.unpaywall = UnpaywallClient(email="research@sanshodhak.org")
        self.arxiv = ArxivClient()
        self.openalex = OpenAlexClient(email="research@sanshodhak.org")
        self.core = COREClient(api_key="fU7PRKOIE5Nz3roXDSvCwVZ46Wd2TpYi")
        self.scihub = SciHubClient()
        
        logger.info("✅ All clients initialized")
    
    def get_user_input(self):
        """Interactive user input."""
        print("\n" + "="*80)
        print("🔬 INTELLIGENT RESEARCH PAPER COLLECTOR")
        print("="*80)
        print("\nSearches CrossRef → Unpaywall → Open-source (arXiv/CORE/OpenAlex)")
        print("Collects PDFs from multiple sources simultaneously\n")
        print("-"*80)
        
        # Get topic
        topic = input("\n📋 Enter research topic: ").strip()
        if not topic:
            print("❌ Topic cannot be empty!")
            sys.exit(1)
        
        # Get number of papers
        try:
            num_papers = input("📊 How many papers? (default: 20): ").strip()
            num_papers = int(num_papers) if num_papers else 20
        except ValueError:
            print("⚠️  Invalid number, using default: 20")
            num_papers = 20
        
        # Get minimum year
        try:
            min_year = input("📅 Minimum year? (default: 2015): ").strip()
            min_year = int(min_year) if min_year else 2015
        except ValueError:
            print("⚠️  Invalid year, using default: 2015")
            min_year = 2015
        
        return topic, num_papers, min_year
    
    def search_crossref(self, topic: str, count: int, min_year: int):
        """Phase 1: Search CrossRef for papers with DOIs."""
        print(f"\n🔍 PHASE 1: CrossRef Metadata Search")
        print("-"*80)
        print(f"   Query: {topic}")
        print(f"   Target: {count} papers")
        print(f"   Min year: {min_year}\n")
        
        result = self.crossref.search(topic, limit=count * 2, min_year=min_year)
        
        if not result.success:
            logger.error(f"CrossRef search failed: {result.error}")
            return []
        
        # Get best papers
        papers = result.papers[:count]
        
        print(f"✅ Found {len(papers)} papers with DOIs")
        print(f"   Papers with citations: {sum(1 for p in papers if p.citations > 0)}")
        print(f"   Year range: {min([p.year for p in papers if p.year])} - {max([p.year for p in papers if p.year])}")
        
        return papers
    
    def fetch_from_unpaywall(self, paper: PaperMetadata):
        """Try Unpaywall for legal OA version."""
        if not paper.doi:
            return None
        
        try:
            info = self.unpaywall.get_pdf_url(paper.doi)
            if info and info.get('pdf_url'):
                import requests
                resp = requests.get(info['pdf_url'], timeout=30, allow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 10000:
                    return resp.content, 'unpaywall'
        except:
            pass
        return None
    
    def fetch_from_open_sources(self, paper: PaperMetadata):
        """
        Simultaneously try open-source: arXiv, CORE, OpenAlex.
        Returns first successful download.
        """
        sources = []
        
        # arXiv - search by title
        def try_arxiv():
            try:
                result = self.arxiv.search(paper.title[:100], limit=1)
                if result.success and result.papers and result.papers[0].pdf_url:
                    import requests
                    resp = requests.get(result.papers[0].pdf_url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 10000:
                        return resp.content, 'arxiv'
            except:
                pass
            return None
        
        # CORE - search by title
        def try_core():
            try:
                result = self.core.search(paper.title[:100], limit=1)
                if result.success and result.papers and result.papers[0].pdf_url:
                    import requests
                    resp = requests.get(result.papers[0].pdf_url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 10000:
                        return resp.content, 'core'
            except:
                pass
            return None
        
        # OpenAlex - search by DOI or title
        def try_openalex():
            try:
                query = f"doi:{paper.doi}" if paper.doi else paper.title[:100]
                result = self.openalex.search(query, limit=1)
                if result.success and result.papers and result.papers[0].pdf_url:
                    import requests
                    resp = requests.get(result.papers[0].pdf_url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 10000:
                        return resp.content, 'openalex'
            except:
                pass
            return None
        
        # Try all sources simultaneously
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(try_arxiv),
                executor.submit(try_core),
                executor.submit(try_openalex)
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        return result
                except:
                    pass
        
        return None
    
    def fetch_from_scihub(self, paper: PaperMetadata):
        """Fallback: Try Sci-Hub."""
        if not paper.doi:
            return None
        
        try:
            result = self.scihub.get_pdf(paper.doi)
            if result:
                pdf_content, mirror = result
                return pdf_content, f'scihub-{mirror}'
        except:
            pass
        return None
    
    def download_single_paper(self, paper: PaperMetadata, index: int, total: int):
        """
        Download single paper following priority:
        1. Unpaywall (legal OA)
        2. Open-source simultaneous (arXiv + CORE + OpenAlex)
        3. Sci-Hub (fallback)
        """
        print(f"\n[{index}/{total}] {paper.title[:65]}...")
        
        filename = f"{index:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Cached: {filename}")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # Priority 1: Unpaywall
        result = self.fetch_from_unpaywall(paper)
        if result:
            pdf_content, source = result
            if self._save_pdf(pdf_content, filepath):
                print(f"   ✅ Unpaywall ({len(pdf_content)/1024/1024:.2f} MB)")
                return {'paper': paper, 'path': str(filepath), 'source': source}
        
        # Priority 2: Open-source simultaneous
        result = self.fetch_from_open_sources(paper)
        if result:
            pdf_content, source = result
            if self._save_pdf(pdf_content, filepath):
                print(f"   ✅ {source.upper()} ({len(pdf_content)/1024/1024:.2f} MB)")
                return {'paper': paper, 'path': str(filepath), 'source': source}
        
        # Priority 3: Sci-Hub fallback
        result = self.fetch_from_scihub(paper)
        if result:
            pdf_content, source = result
            if self._save_pdf(pdf_content, filepath):
                print(f"   ✅ {source} ({len(pdf_content)/1024/1024:.2f} MB)")
                return {'paper': paper, 'path': str(filepath), 'source': source}
        
        print(f"   ❌ All sources failed")
        return None
    
    def _save_pdf(self, pdf_content, filepath):
        """Validate and save PDF."""
        if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
            filepath.write_bytes(pdf_content)
            return True
        return False
    
    def _safe_filename(self, paper: PaperMetadata):
        """Generate safe filename."""
        import re
        author = paper.authors[0].split()[-1] if paper.authors else "Unknown"
        author = re.sub(r'[^\w]', '', author)[:15]
        year = paper.year or "0000"
        return f"{author}_{year}.pdf"
    
    def run(self):
        """Main orchestration flow."""
        # Get user input
        topic, num_papers, min_year = self.get_user_input()
        
        # Create topic directory
        topic_slug = topic.lower().replace(' ', '_')[:50]
        topic_dir = self.output_dir / topic_slug
        topic_dir.mkdir(exist_ok=True)
        self.output_dir = topic_dir
        
        print("\n" + "="*80)
        print("🚀 STARTING COLLECTION")
        print("="*80)
        
        # Phase 1: CrossRef search
        papers = self.search_crossref(topic, num_papers, min_year)
        
        if not papers:
            print("\n❌ No papers found!")
            return
        
        # Phase 2: Parallel download
        print(f"\n📥 PHASE 2: Multi-Source PDF Download")
        print("-"*80)
        print("Priority: Unpaywall → (arXiv+CORE+OpenAlex) → Sci-Hub")
        print("-"*80)
        
        downloaded = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, paper in enumerate(papers, 1):
                future = executor.submit(self.download_single_paper, paper, idx, len(papers))
                futures.append((future, paper))
            
            for future, paper in futures:
                try:
                    result = future.result(timeout=120)
                    if result:
                        downloaded.append(result)
                    else:
                        failed.append(paper)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    failed.append(paper)
        
        # Phase 3: Generate report
        self._generate_report(topic, papers, downloaded, failed, min_year, topic_dir)
    
    def _generate_report(self, topic, papers, downloaded, failed, min_year, topic_dir):
        """Generate comprehensive report."""
        print(f"\n📝 PHASE 3: Generate Report")
        print("-"*80)
        
        success_rate = len(downloaded) / len(papers) * 100
        
        # Count sources
        source_counts = Counter([d['source'] for d in downloaded])
        
        # Save metadata
        metadata = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'min_year': min_year,
            'target_papers': len(papers),
            'downloaded_papers': len(downloaded),
            'failed_papers': len(failed),
            'success_rate': f"{success_rate:.1f}%",
            'sources_used': dict(source_counts),
            'papers': [
                {
                    'title': d['paper'].title,
                    'authors': d['paper'].authors,
                    'year': d['paper'].year,
                    'doi': d['paper'].doi,
                    'citations': d['paper'].citations,
                    'file': Path(d['path']).name,
                    'source': d['source']
                }
                for d in downloaded
            ]
        }
        
        metadata_file = topic_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        print(f"✅ Metadata: {metadata_file.name}")
        
        # Generate markdown report
        report = f"""# Research Report: {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Target Papers:** {len(papers)}
- **PDFs Downloaded:** {len(downloaded)}
- **Success Rate:** {success_rate:.1f}%
- **Min Year:** {min_year}
- **Failed:** {len(failed)}

## Sources Distribution

"""
        for source, count in source_counts.most_common():
            pct = count / len(downloaded) * 100
            report += f"- **{source}**: {count} papers ({pct:.1f}%)\n"
        
        report += "\n## Downloaded Papers\n\n"
        
        for d in downloaded:
            p = d['paper']
            report += f"### {p.title}\n\n"
            report += f"**Authors:** {', '.join(p.authors[:3])}\n\n"
            report += f"**Year:** {p.year} | **Citations:** {p.citations} | **DOI:** {p.doi}\n\n"
            report += f"**File:** `{Path(d['path']).name}` | **Source:** {d['source']}\n\n"
            
            if p.abstract:
                abstract = p.abstract[:250] + "..." if len(p.abstract) > 250 else p.abstract
                report += f"**Abstract:** {abstract}\n\n"
            
            report += "---\n\n"
        
        if failed:
            report += f"\n## Failed Downloads ({len(failed)})\n\n"
            for p in failed[:10]:
                report += f"- {p.title} (DOI: {p.doi})\n"
        
        report_file = topic_dir / "RESEARCH_REPORT.md"
        report_file.write_text(report)
        print(f"✅ Report: {report_file.name}")
        
        # Final summary
        print("\n" + "="*80)
        print("✅ COLLECTION COMPLETE!")
        print("="*80)
        print(f"\n📁 Location: {topic_dir.absolute()}")
        print(f"📄 Papers Found: {len(papers)}")
        print(f"📥 PDFs Downloaded: {len(downloaded)} ({success_rate:.1f}%)")
        print(f"❌ Failed: {len(failed)}")
        print(f"\n📊 Source Breakdown:")
        for source, count in source_counts.most_common():
            print(f"   • {source}: {count}")
        print("\n" + "="*80 + "\n")


def main():
    """Entry point."""
    try:
        orchestrator = InteractiveOrchestrator()
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
