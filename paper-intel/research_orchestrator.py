"""
Research Orchestrator - End-to-End Paper Discovery & Download
Searches multiple sources, merges results, and downloads PDFs
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.ingestion_engine import IngestionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Orchestrates complete research paper discovery and download pipeline."""
    
    def __init__(self, output_dir: str = "research_papers"):
        """
        Initialize orchestrator.
        
        Args:
            output_dir: Directory to save downloaded papers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # SearchEngine and PDFDownloader will be initialized per-query
        self.search_engine = None
        self.pdf_downloader = None
        
        logger.info(f"📁 Output directory: {self.output_dir.absolute()}")
    
    def conduct_research(
        self,
        topic: str,
        num_papers: int = 20,
        min_year: int = 2020,
        open_access_only: bool = False,
        download_pdfs: bool = True,
        use_closed_sources: bool = True
    ):
        """
        Conduct complete research on a topic.
        
        Args:
            topic: Research topic/query
            num_papers: Number of papers to find
            min_year: Minimum publication year
            open_access_only: Only return open-access papers
            download_pdfs: Whether to download PDFs
            use_closed_sources: Whether to use CrossRef/Unpaywall/arXiv
            
        Returns:
            Dict with results and statistics
        """
        print("\n" + "="*80)
        print(f"🔬 RESEARCH ORCHESTRATION")
        print("="*80)
        print(f"\n📋 Research Topic: {topic}")
        print(f"📊 Target Papers: {num_papers}")
        print(f"📅 Min Year: {min_year}")
        print(f"🔓 Open Access Only: {open_access_only}")
        print(f"📥 Download PDFs: {download_pdfs}")
        print(f"🌐 Use Closed Sources: {use_closed_sources}")
        print("\n" + "-"*80)
        
        # Create topic-specific subdirectory
        topic_slug = self._slugify(topic)
        topic_dir = self.output_dir / topic_slug
        topic_dir.mkdir(exist_ok=True)
        
        # Step 1: Search across all sources
        print("\n🔍 PHASE 1: Multi-Source Search")
        print("-"*80)
        
        # Initialize search engine
        from ingestion.discovery import SearchEngine, PDFDownloader
        self.search_engine = SearchEngine()
        self.pdf_downloader = PDFDownloader()
        
        search_params = {
            'limit': num_papers,
            'min_year': min_year,
            'prefer_open_access': open_access_only
        }
        
        # Use search_unified for combined results
        result = self.search_engine.search_unified(
            query=topic,
            **search_params
        )
        
        # Convert list to SearchResult-like object
        class SimpleResult:
            def __init__(self, papers):
                self.success = len(papers) > 0
                self.papers = papers
                self.source = "unified"
                self.error = None if papers else "No papers found"
                self.fetched_count = len(papers)
                self.total_found = len(papers)
        
        result = SimpleResult(result if isinstance(result, list) else [])
        
        if not result.success or not result.papers:
            print("\n❌ No papers found!")
            return {'success': False, 'error': result.error or 'No papers found'}
        
        print(f"\n✅ Found {len(result.papers)} papers from {result.source}")
        
        # Step 2: Analyze results
        print("\n📊 PHASE 2: Result Analysis")
        print("-"*80)
        
        stats = self._analyze_papers(result.papers)
        self._print_stats(stats)
        
        # Step 3: Save metadata
        print("\n💾 PHASE 3: Save Metadata")
        print("-"*80)
        
        metadata_file = topic_dir / "metadata.json"
        self._save_metadata(result.papers, metadata_file, topic, search_params)
        print(f"✅ Metadata saved: {metadata_file}")
        
        # Step 4: Download PDFs
        downloaded_papers = []
        if download_pdfs:
            print("\n📥 PHASE 4: PDF Download")
            print("-"*80)
            
            downloaded_papers = self._download_all_pdfs(result.papers, topic_dir)
            
            print(f"\n✅ Downloaded {len(downloaded_papers)}/{len(result.papers)} PDFs")
        else:
            print("\n⏭️  PHASE 4: Skipped (download_pdfs=False)")
        
        # Step 5: Generate summary report
        print("\n📝 PHASE 5: Generate Report")
        print("-"*80)
        
        report_file = topic_dir / "RESEARCH_REPORT.md"
        self._generate_report(
            topic, result.papers, downloaded_papers, 
            stats, search_params, report_file
        )
        print(f"✅ Report saved: {report_file}")
        
        # Final summary
        print("\n" + "="*80)
        print("✅ RESEARCH COMPLETE!")
        print("="*80)
        print(f"\n📁 Location: {topic_dir.absolute()}")
        print(f"📄 Papers Found: {len(result.papers)}")
        print(f"📥 PDFs Downloaded: {len(downloaded_papers)}")
        print(f"📊 Metadata: {metadata_file.name}")
        print(f"📝 Report: {report_file.name}")
        print("\n" + "="*80 + "\n")
        
        return {
            'success': True,
            'topic': topic,
            'papers_found': len(result.papers),
            'pdfs_downloaded': len(downloaded_papers),
            'output_directory': str(topic_dir.absolute()),
            'metadata_file': str(metadata_file),
            'report_file': str(report_file),
            'statistics': stats
        }
    
    def _analyze_papers(self, papers):
        """Analyze paper collection statistics."""
        stats = {
            'total': len(papers),
            'with_pdf': sum(1 for p in papers if p.pdf_url),
            'open_access': sum(1 for p in papers if p.is_open_access),
            'with_doi': sum(1 for p in papers if p.doi),
            'years': {},
            'sources': {}
        }
        
        for paper in papers:
            # Year distribution
            if paper.year:
                stats['years'][paper.year] = stats['years'].get(paper.year, 0) + 1
            
            # Source distribution
            source = getattr(paper, 'source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        return stats
    
    def _print_stats(self, stats):
        """Print statistics summary."""
        print(f"   Total Papers: {stats['total']}")
        print(f"   With PDF URL: {stats['with_pdf']} ({stats['with_pdf']/stats['total']*100:.1f}%)")
        print(f"   Open Access: {stats['open_access']} ({stats['open_access']/stats['total']*100:.1f}%)")
        print(f"   With DOI: {stats['with_doi']} ({stats['with_doi']/stats['total']*100:.1f}%)")
        
        if stats['years']:
            year_range = f"{min(stats['years'].keys())}-{max(stats['years'].keys())}"
            print(f"   Year Range: {year_range}")
        
        if stats['sources']:
            print(f"   Sources: {', '.join(stats['sources'].keys())}")
    
    def _save_metadata(self, papers, filename, topic, search_params):
        """Save paper metadata to JSON."""
        import json
        
        metadata = {
            'topic': topic,
            'search_params': search_params,
            'timestamp': datetime.now().isoformat(),
            'total_papers': len(papers),
            'papers': []
        }
        
        for paper in papers:
            paper_dict = {
                'title': paper.title,
                'authors': paper.authors,
                'year': paper.year,
                'doi': paper.doi,
                'pdf_url': paper.pdf_url,
                'is_open_access': paper.is_open_access,
                'citations': paper.citations,
                'abstract': paper.abstract,
                'source': getattr(paper, 'source', 'unknown')
            }
            metadata['papers'].append(paper_dict)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _download_all_pdfs(self, papers, output_dir):
        """Download PDFs for all papers."""
        downloaded = []
        
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] {paper.title[:60]}...")
            
            if not paper.pdf_url and not paper.doi:
                print("   ⏭️  No PDF URL or DOI - skipping")
                continue
            
            try:
                # Generate filename
                filename = self._generate_filename(paper, i)
                filepath = output_dir / filename
                
                # Try to download
                success = self._download_single_pdf(paper, filepath)
                
                if success:
                    downloaded.append({
                        'paper': paper,
                        'filename': filename,
                        'filepath': str(filepath)
                    })
                    print(f"   ✅ Saved: {filename}")
                else:
                    print(f"   ❌ Download failed")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return downloaded
    
    def _download_single_pdf(self, paper, filepath):
        """Download a single PDF file."""
        # First try direct PDF URL
        if paper.pdf_url:
            try:
                import requests
                response = requests.get(paper.pdf_url, timeout=30, allow_redirects=True)
                
                if response.status_code == 200 and len(response.content) > 10000:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' in content_type or response.content.startswith(b'%PDF'):
                        filepath.write_bytes(response.content)
                        return True
            except:
                pass
        
        # Try PDF downloader (Unpaywall, Sci-Hub, etc.)
        if paper.doi and self.pdf_downloader:
            try:
                pdf_path = self.pdf_downloader.download(
                    paper.doi, 
                    output_dir=str(filepath.parent),
                    filename=filepath.name
                )
                if pdf_path and Path(pdf_path).exists():
                    return True
            except:
                pass
        
        return False
    
    def _generate_filename(self, paper, index):
        """Generate safe filename for paper."""
        # Use first author + year + index
        author = paper.authors[0] if paper.authors else "Unknown"
        author = author.split()[-1] if ' ' in author else author  # Last name
        year = paper.year or "0000"
        
        # Sanitize
        author = "".join(c for c in author if c.isalnum())[:20]
        
        return f"{index:03d}_{author}_{year}.pdf"
    
    def _generate_report(self, topic, papers, downloaded, stats, params, filename):
        """Generate markdown research report."""
        report = f"""# Research Report: {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Search Parameters

- **Topic:** {topic}
- **Target Papers:** {params.get('required_count', 'N/A')}
- **Min Year:** {params.get('min_year', 'N/A')}
- **Open Access Only:** {params.get('open_access_only', False)}

## Summary Statistics

- **Total Papers Found:** {stats['total']}
- **PDFs Downloaded:** {len(downloaded)}
- **Open Access:** {stats['open_access']} ({stats['open_access']/stats['total']*100:.1f}%)
- **With DOI:** {stats['with_doi']} ({stats['with_doi']/stats['total']*100:.1f}%)

### Year Distribution

"""
        for year in sorted(stats['years'].keys(), reverse=True):
            count = stats['years'][year]
            report += f"- **{year}:** {count} papers\n"
        
        report += "\n### Source Distribution\n\n"
        for source, count in stats['sources'].items():
            report += f"- **{source}:** {count} papers\n"
        
        report += f"\n## Papers List\n\n"
        
        for i, paper in enumerate(papers, 1):
            downloaded_status = "✅" if any(d['paper'] == paper for d in downloaded) else "❌"
            
            report += f"### {i}. {paper.title}\n\n"
            report += f"**Status:** {downloaded_status} | "
            report += f"**Year:** {paper.year or 'N/A'} | "
            report += f"**Citations:** {paper.citations or 0} | "
            report += f"**OA:** {'Yes' if paper.is_open_access else 'No'}\n\n"
            
            if paper.authors:
                authors_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f" et al. ({len(paper.authors)} total)"
                report += f"**Authors:** {authors_str}\n\n"
            
            if paper.doi:
                report += f"**DOI:** [{paper.doi}](https://doi.org/{paper.doi})\n\n"
            
            if paper.pdf_url:
                report += f"**PDF URL:** {paper.pdf_url}\n\n"
            
            if paper.abstract:
                abstract = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
                report += f"**Abstract:** {abstract}\n\n"
            
            report += "---\n\n"
        
        filename.write_text(report, encoding='utf-8')
    
    def _slugify(self, text):
        """Convert text to filesystem-safe slug."""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '_', text)
        text = text.strip('_')
        return text[:50]


def main():
    """Interactive research orchestrator."""
    print("\n" + "="*80)
    print("🔬 SANSHODHAK RESEARCH ORCHESTRATOR")
    print("="*80)
    print("\nAutomated paper discovery and download system")
    print("Searches: OpenAlex, CORE, Semantic Scholar, CrossRef, Unpaywall, arXiv")
    print("\n" + "-"*80)
    
    # Get user input
    topic = input("\n📋 Enter research topic: ").strip()
    if not topic:
        print("❌ No topic provided!")
        return
    
    num_papers = input("📊 Number of papers (default 20): ").strip()
    num_papers = int(num_papers) if num_papers.isdigit() else 20
    
    min_year = input("📅 Minimum year (default 2020): ").strip()
    min_year = int(min_year) if min_year.isdigit() else 2020
    
    oa_only = input("🔓 Open access only? (y/N): ").strip().lower() == 'y'
    
    download = input("📥 Download PDFs? (Y/n): ").strip().lower() != 'n'
    
    # Create orchestrator and run
    orchestrator = ResearchOrchestrator()
    
    try:
        result = orchestrator.conduct_research(
            topic=topic,
            num_papers=num_papers,
            min_year=min_year,
            open_access_only=oa_only,
            download_pdfs=download
        )
        
        if result['success']:
            print(f"\n🎉 Research complete! Check: {result['output_directory']}")
        else:
            print(f"\n❌ Research failed: {result.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
