#!/usr/bin/env python3
"""
ENHANCED Interactive Orchestrator with Semantic Filtering
- Searches CrossRef for MORE papers (2x target)
- Filters irrelevant papers using title embeddings
- Multi-source: Unpaywall + (arXiv + OpenAlex + CORE) simultaneous
- Minimum year 2015, tests with 25 papers
"""

import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import Counter
import json
import re

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.discovery import (
    CrossRefClient, UnpaywallClient, ArxivClient, 
    OpenAlexClient, COREClient, SciHubClient
)
from ingestion.models import PaperMetadata

# Semantic similarity imports
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SemanticOrchestrator:
    """
    Smart orchestrator with semantic filtering:
    1. Search CrossRef for 2x papers
    2. Filter using title embeddings (semantic similarity)
    3. Multi-source parallel download
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
        
        # Semantic model
        self.embedding_model = None
        self.similarity_threshold = 0.55  # Lower threshold for more papers
        self.has_embeddings = HAS_EMBEDDINGS
        
        if self.has_embeddings:
            print("🧠 Loading semantic model (all-MiniLM-L6-v2)...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic filtering enabled")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                self.has_embeddings = False
        
        logger.info("✅ All clients initialized")
    
    def get_user_input(self):
        """Interactive user input."""
        print("\n" + "="*80)
        print("🔬 SEMANTIC RESEARCH PAPER COLLECTOR")
        print("="*80)
        print("\n🧠 Features:")
        print("   • Semantic filtering - removes irrelevant papers")
        print("   • Multi-source search: CrossRef → Unpaywall + Open-access")
        print("   • Parallel downloads from arXiv, CORE, OpenAlex")
        print("\n" + "-"*80)
        
        # Get topic
        topic = input("\n📋 Enter research topic: ").strip()
        if not topic:
            print("❌ Topic cannot be empty!")
            sys.exit(1)
        
        # Get number of papers
        try:
            num_papers = input("📊 How many papers? (default: 25): ").strip()
            num_papers = int(num_papers) if num_papers else 25
        except ValueError:
            print("⚠️  Invalid number, using default: 25")
            num_papers = 25
        
        # Get minimum year
        try:
            min_year = input("📅 Minimum year? (default: 2015): ").strip()
            min_year = int(min_year) if min_year else 2015
        except ValueError:
            print("⚠️  Invalid year, using default: 2015")
            min_year = 2015
        
        return topic, num_papers, min_year
    
    def get_embedding(self, text: str):
        """Get embedding for text."""
        if not self.has_embeddings or not self.embedding_model:
            return None
        
        try:
            # Clean text
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
            text = ' '.join(text.split())  # Normalize whitespace
            return self.embedding_model.encode(text, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def semantic_similarity(self, text1: str, text2: str):
        """Calculate cosine similarity between two texts."""
        if not self.has_embeddings:
            return 1.0  # No filtering if embeddings unavailable
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 1.0
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def filter_papers_semantic(self, query: str, papers: list, target_count: int):
        """
        Filter papers based on semantic similarity to query.
        Returns top N most relevant papers.
        """
        if not self.has_embeddings or not papers:
            return papers[:target_count]
        
        print(f"\n🧠 Semantic Filtering")
        print("-"*80)
        print(f"   Analyzing {len(papers)} papers...")
        
        # Calculate similarity for each paper
        scored_papers = []
        query_embedding = self.get_embedding(query)
        
        if query_embedding is None:
            return papers[:target_count]
        
        for paper in papers:
            # Use title + abstract for similarity
            paper_text = paper.title
            if paper.abstract:
                paper_text += " " + paper.abstract[:500]
            
            paper_embedding = self.get_embedding(paper_text)
            
            if paper_embedding is not None:
                similarity = np.dot(query_embedding, paper_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(paper_embedding)
                )
                scored_papers.append((paper, float(similarity)))
            else:
                scored_papers.append((paper, 0.5))  # Default score
        
        # Sort by similarity
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        filtered = [
            (p, score) for p, score in scored_papers 
            if score >= self.similarity_threshold
        ]
        
        if not filtered:
            print(f"   ⚠️  No papers above threshold {self.similarity_threshold}, using top papers")
            filtered = scored_papers
        
        # Take top N
        selected = filtered[:target_count]
        
        print(f"   ✅ Selected {len(selected)} papers")
        print(f"   Similarity range: {selected[-1][1]:.3f} - {selected[0][1]:.3f}")
        print(f"   Filtered out: {len(papers) - len(selected)} irrelevant papers")
        
        return [p for p, _ in selected]
    
    def search_crossref_large(self, topic: str, count: int, min_year: int):
        """
        Phase 1: Search CrossRef for 2x papers (to filter later).
        """
        print(f"\n🔍 PHASE 1: CrossRef Search (2x Target)")
        print("-"*80)
        print(f"   Query: {topic}")
        print(f"   Searching for: {count * 2} papers")
        print(f"   Min year: {min_year}")
        
        result = self.crossref.search(topic, limit=count * 2, min_year=min_year)
        
        if not result.success:
            logger.error(f"CrossRef search failed: {result.error}")
            return []
        
        papers = result.papers
        
        print(f"\n✅ Found {len(papers)} papers from CrossRef")
        print(f"   With abstracts: {sum(1 for p in papers if p.abstract)}")
        print(f"   With citations: {sum(1 for p in papers if p.citations and p.citations > 0)}")
        
        if papers:
            years = [p.year for p in papers if p.year]
            if years:
                print(f"   Year range: {min(years)} - {max(years)}")
        
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
                    if resp.content[:4] == b'%PDF':
                        return resp.content, 'unpaywall'
        except Exception as e:
            logger.debug(f"Unpaywall failed: {e}")
        return None
    
    def fetch_from_open_sources(self, paper: PaperMetadata):
        """
        Simultaneously try: arXiv, CORE, OpenAlex.
        Returns first successful download.
        """
        def try_arxiv():
            try:
                result = self.arxiv.search(paper.title[:100], limit=1)
                if result.success and result.papers and result.papers[0].pdf_url:
                    import requests
                    resp = requests.get(result.papers[0].pdf_url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 10000:
                        if resp.content[:4] == b'%PDF':
                            return resp.content, 'arxiv'
            except Exception as e:
                logger.debug(f"arXiv failed: {e}")
            return None
        
        def try_core():
            try:
                result = self.core.search(paper.title[:100], limit=1)
                if result.success and result.papers and result.papers[0].pdf_url:
                    import requests
                    resp = requests.get(result.papers[0].pdf_url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 10000:
                        if resp.content[:4] == b'%PDF':
                            return resp.content, 'core'
            except Exception as e:
                logger.debug(f"CORE failed: {e}")
            return None
        
        def try_openalex():
            try:
                query = f"doi:{paper.doi}" if paper.doi else paper.title[:100]
                result = self.openalex.search(query, limit=1)
                if result.success and result.papers and result.papers[0].pdf_url:
                    import requests
                    resp = requests.get(result.papers[0].pdf_url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 10000:
                        if resp.content[:4] == b'%PDF':
                            return resp.content, 'openalex'
            except Exception as e:
                logger.debug(f"OpenAlex failed: {e}")
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
                except Exception as e:
                    logger.debug(f"Source failed: {e}")
        
        return None
    
    def fetch_from_scihub(self, paper: PaperMetadata):
        """Fallback: Try Sci-Hub."""
        if not paper.doi:
            return None
        
        try:
            result = self.scihub.get_pdf(paper.doi)
            if result:
                pdf_content, mirror = result
                if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
                    return pdf_content, 'scihub'
        except Exception as e:
            logger.debug(f"Sci-Hub failed: {e}")
        return None
    
    def download_single_paper(self, paper: PaperMetadata, index: int, total: int):
        """
        Download with priority: Unpaywall → Open-source → Sci-Hub.
        """
        title_short = paper.title[:65] + "..." if len(paper.title) > 65 else paper.title
        print(f"\n[{index}/{total}] {title_short}")
        
        filename = f"{index:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Cached")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # Try Unpaywall first
        result = self.fetch_from_unpaywall(paper)
        if result:
            pdf_content, source = result
            filepath.write_bytes(pdf_content)
            print(f"   ✅ Unpaywall ({len(pdf_content)/1024/1024:.2f} MB)")
            return {'paper': paper, 'path': str(filepath), 'source': source}
        
        # Try open-source simultaneous
        result = self.fetch_from_open_sources(paper)
        if result:
            pdf_content, source = result
            filepath.write_bytes(pdf_content)
            print(f"   ✅ {source.upper()} ({len(pdf_content)/1024/1024:.2f} MB)")
            return {'paper': paper, 'path': str(filepath), 'source': source}
        
        # Fallback to Sci-Hub
        result = self.fetch_from_scihub(paper)
        if result:
            pdf_content, source = result
            filepath.write_bytes(pdf_content)
            print(f"   ✅ Sci-Hub ({len(pdf_content)/1024/1024:.2f} MB)")
            return {'paper': paper, 'path': str(filepath), 'source': source}
        
        print(f"   ❌ All sources failed")
        return None
    
    def _safe_filename(self, paper: PaperMetadata):
        """Generate safe filename."""
        author = paper.authors[0].split()[-1] if paper.authors else "Unknown"
        author = re.sub(r'[^\w]', '', author)[:15]
        year = paper.year or "0000"
        return f"{author}_{year}.pdf"
    
    def run(self):
        """Main orchestration with semantic filtering."""
        # Get user input
        topic, num_papers, min_year = self.get_user_input()
        
        # Create topic directory
        topic_slug = re.sub(r'[^\w\s]', '', topic.lower()).replace(' ', '_')[:50]
        topic_dir = self.output_dir / topic_slug
        topic_dir.mkdir(exist_ok=True)
        self.output_dir = topic_dir
        
        print("\n" + "="*80)
        print("🚀 STARTING COLLECTION")
        print("="*80)
        
        # Phase 1: Search CrossRef (2x papers)
        all_papers = self.search_crossref_large(topic, num_papers, min_year)
        
        if not all_papers:
            print("\n❌ No papers found!")
            return
        
        # Phase 2: Semantic filtering
        if self.has_embeddings:
            selected_papers = self.filter_papers_semantic(topic, all_papers, num_papers)
        else:
            print(f"\n⚠️  Semantic filtering disabled, using top {num_papers} papers")
            selected_papers = all_papers[:num_papers]
        
        print(f"\n📋 Selected {len(selected_papers)} papers for download")
        
        # Phase 3: Multi-source parallel download
        print(f"\n📥 PHASE 2: Multi-Source PDF Download")
        print("-"*80)
        print("Priority: Unpaywall → (arXiv + CORE + OpenAlex) simultaneous → Sci-Hub")
        print("-"*80)
        
        downloaded = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, paper in enumerate(selected_papers, 1):
                future = executor.submit(self.download_single_paper, paper, idx, len(selected_papers))
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
        
        # Phase 4: Generate report
        self._generate_report(topic, selected_papers, downloaded, failed, min_year, topic_dir)
    
    def _generate_report(self, topic, papers, downloaded, failed, min_year, topic_dir):
        """Generate comprehensive report."""
        print(f"\n📝 PHASE 3: Generate Report")
        print("-"*80)
        
        success_rate = len(downloaded) / len(papers) * 100 if papers else 0
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
            'semantic_filtering': self.has_embeddings,
            'sources_used': dict(source_counts),
            'papers': [
                {
                    'title': d['paper'].title,
                    'authors': d['paper'].authors,
                    'year': d['paper'].year,
                    'doi': d['paper'].doi,
                    'citations': d['paper'].citations,
                    'file': Path(d['path']).name,
                    'source': d['source'],
                    'abstract': d['paper'].abstract[:200] if d['paper'].abstract else None
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
**Semantic Filtering:** {'✅ Enabled' if self.has_embeddings else '❌ Disabled'}

## Summary

- **Target Papers:** {len(papers)}
- **PDFs Downloaded:** {len(downloaded)}
- **Success Rate:** {success_rate:.1f}%
- **Min Year:** {min_year}
- **Failed:** {len(failed)}

## Sources Distribution

"""
        for source, count in source_counts.most_common():
            pct = count / len(downloaded) * 100 if downloaded else 0
            report += f"- **{source}**: {count} papers ({pct:.1f}%)\n"
        
        report += "\n## Downloaded Papers\n\n"
        
        for d in sorted(downloaded, key=lambda x: x['paper'].citations or 0, reverse=True):
            p = d['paper']
            report += f"### {p.title}\n\n"
            report += f"**Authors:** {', '.join(p.authors[:3])}\n\n"
            report += f"**Year:** {p.year} | **Citations:** {p.citations or 0} | **DOI:** {p.doi}\n\n"
            report += f"**File:** `{Path(d['path']).name}` | **Source:** {d['source']}\n\n"
            
            if p.abstract:
                abstract = p.abstract[:300] + "..." if len(p.abstract) > 300 else p.abstract
                report += f"**Abstract:** {abstract}\n\n"
            
            report += "---\n\n"
        
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
        
        if downloaded:
            total_size = sum(Path(d['path']).stat().st_size for d in downloaded)
            print(f"\n💾 Total Size: {total_size / (1024*1024):.1f} MB")
        
        print("\n" + "="*80 + "\n")


def main():
    """Entry point."""
    try:
        orchestrator = SemanticOrchestrator()
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
