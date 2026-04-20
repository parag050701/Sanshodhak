"""
Main PDF -> TEXT -> JSON pipeline for Phase 1.

Process PDFs through GROBID, extract structured data,
and save to JSON format ready for knowledge graph construction.
"""

import asyncio
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml
import sys

# Handle both module and standalone execution
try:
    from .grobid_client import GrobidClient
    from .tei_parser import TEIParser
    from .text_cleaner import TextCleaner
    from .section_splitter import SectionSplitter
    from .fallback_extractor import FallbackExtractor
except ImportError:
    # Add parent directory to path for standalone execution
    sys.path.insert(0, str(Path(__file__).parent))
    from grobid_client import GrobidClient
    from tei_parser import TEIParser
    from text_cleaner import TextCleaner
    from section_splitter import SectionSplitter
    from fallback_extractor import FallbackExtractor

logger = logging.getLogger(__name__)


class PDFToTextPipeline:
    """
    Complete PDF -> TEXT -> JSON pipeline.
    
    Steps:
        1. Send PDF to GROBID -> get TEI XML
        2. Parse TEI -> extract metadata + body text
        3. Clean text
        4. Split into sections
        5. Create structured JSON
        6. Save outputs
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        grobid_url: str = "http://localhost:8070",
        output_dir: Path = Path("ingestion"),
        workers: int = 4
    ):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to settings.yaml
            grobid_url: GROBID server URL
            output_dir: Base output directory
            workers: Number of concurrent workers
        """
        # Load config if provided
        if config_path and config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                grobid_url = config.get('grobid_url', grobid_url)
                workers = config.get('workers', workers)
        
        # Initialize components
        self.grobid = GrobidClient(base_url=grobid_url)
        self.tei_parser = TEIParser()
        self.text_cleaner = TextCleaner()
        self.section_splitter = SectionSplitter()
        self.fallback_extractor = FallbackExtractor()
        self.workers = workers
        self.use_fallback = True  # Enable fallback extraction
        
        # Setup output directories
        self.output_dir = output_dir
        self.json_dir = output_dir / "output_json"
        self.text_dir = output_dir / "raw_text"
        self.error_dir = output_dir / "errors"
        
        self._create_directories()
        
        logger.info(f"Pipeline initialized: GROBID={grobid_url}, workers={workers}")
    
    def _create_directories(self):
        """Create output directories if they don't exist."""
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_paper_id(self, pdf_path: Path) -> str:
        """
        Generate unique paper ID from PDF filename.
        
        Uses hash of filename for collision resistance.
        """
        # Use stem (filename without extension)
        stem = pdf_path.stem
        
        # Create short hash for uniqueness
        hash_obj = hashlib.md5(stem.encode())
        hash_short = hash_obj.hexdigest()[:8]
        
        # Clean stem for readability
        clean_stem = stem[:30]  # Limit length
        clean_stem = clean_stem.replace(' ', '_')
        
        return f"{clean_stem}_{hash_short}"
    
    async def process_single_pdf(
        self,
        pdf_path: Path
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process single PDF through complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (success, paper_id, error_message)
        """
        paper_id = self.generate_paper_id(pdf_path)
        
        logger.info(f"Processing: {pdf_path.name} -> {paper_id}")
        
        try:
            # Step 1: GROBID processing (primary method)
            success, tei_xml, error = await self.grobid.process_fulltext(pdf_path)
            
            if not success or not tei_xml:
                # Try fallback extraction if GROBID fails
                if self.use_fallback:
                    logger.warning(f"GROBID failed for {paper_id}, trying fallback extraction")
                    return await self._process_with_fallback(pdf_path, paper_id)
                else:
                    error_msg = f"GROBID failed: {error}"
                    self._save_error(paper_id, pdf_path, error_msg)
                    return False, paper_id, error_msg
            
            # Step 2: Parse TEI
            parsed = self.tei_parser.parse(tei_xml)
            
            if not parsed.get('title'):
                logger.warning(f"No title extracted for {paper_id}")
            
            # Step 3: Clean body text
            body_text = parsed.get('body_text', '')
            
            # If body text is too short, try fallback
            if len(body_text) < 500 and self.use_fallback:
                logger.warning(f"Short body text for {paper_id}, trying fallback")
                fallback_result = await self._process_with_fallback(pdf_path, paper_id)
                # Use fallback if it got more text
                if fallback_result[0]:  # success
                    return fallback_result
            
            clean_text = self.text_cleaner.clean(body_text)
            
            # Clean abstract
            abstract = self.text_cleaner.clean_abstract(parsed.get('abstract', ''))
            
            # Clean title
            title = self.text_cleaner.clean_title(parsed.get('title', ''))
            
            # Clean author names
            authors = [
                self.text_cleaner.clean_author_name(author)
                for author in parsed.get('authors', [])
            ]
            
            # Step 4: Split into sections
            sections = self.section_splitter.split(clean_text)
            
            # Step 5: Create structured JSON
            output_json = {
                'paper_id': paper_id,
                'source_pdf': pdf_path.name,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'sections': sections,
                'references': parsed.get('references', []),
                'metadata': {
                    'year': parsed['metadata'].get('year', ''),
                    'doi': parsed['metadata'].get('doi', ''),
                    'venue': parsed['metadata'].get('venue', ''),
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            # Step 6: Save outputs
            self._save_json(paper_id, output_json)
            self._save_raw_text(paper_id, clean_text)
            
            logger.info(f"[OK] Success: {paper_id}")
            return True, paper_id, None
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(f"❌ Failed {paper_id}: {error_msg}")
            self._save_error(paper_id, pdf_path, error_msg)
            return False, paper_id, error_msg
    
    async def _process_with_fallback(
        self,
        pdf_path: Path,
        paper_id: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process PDF using fallback extraction (PyMuPDF/pdfplumber/PyPDF2).
        
        Used when GROBID fails or extracts insufficient text.
        """
        try:
            # Extract text with fallback
            success, text, metadata = self.fallback_extractor.extract(pdf_path)
            
            if not success or not text:
                error_msg = "Fallback extraction failed"
                self._save_error(paper_id, pdf_path, error_msg)
                return False, paper_id, error_msg
            
            logger.info(f"Fallback extraction: {len(text)} chars via {metadata.get('extractor')}")
            
            # Clean text
            clean_text = self.text_cleaner.clean(text)
            
            # Split into sections
            sections = self.section_splitter.split(clean_text)
            
            # Create minimal structured output
            # Try to extract title from metadata or first lines
            title = metadata.get('title', '')
            if not title:
                # Use first non-empty line as title
                lines = clean_text.split('\n')
                for line in lines[:10]:
                    line = line.strip()
                    if len(line) > 10 and len(line) < 200:
                        title = line
                        break
            
            # Try to extract authors from metadata
            authors = []
            if metadata.get('author'):
                authors = [metadata['author']]
            
            # Create JSON output
            output_json = {
                'paper_id': paper_id,
                'source_pdf': pdf_path.name,
                'title': self.text_cleaner.clean_title(title),
                'authors': authors,
                'abstract': sections.get('introduction', '')[:500],  # Use intro as abstract
                'sections': sections,
                'references': [],  # Fallback doesn't extract references
                'metadata': {
                    'year': '',
                    'doi': '',
                    'venue': '',
                    'processed_at': datetime.now().isoformat(),
                    'extraction_method': 'fallback',
                    'extractor': metadata.get('extractor', 'unknown'),
                    'pages': metadata.get('pages', 0)
                }
            }
            
            # Save outputs
            self._save_json(paper_id, output_json)
            self._save_raw_text(paper_id, clean_text)
            
            logger.info(f"[OK] Success (fallback): {paper_id}")
            return True, paper_id, None
            
        except Exception as e:
            error_msg = f"Fallback processing error: {str(e)}"
            logger.error(f"❌ Fallback failed {paper_id}: {error_msg}")
            self._save_error(paper_id, pdf_path, error_msg)
            return False, paper_id, error_msg
    
    def process_single_pdf_sync(self, pdf_path: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Synchronous wrapper for process_single_pdf.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (success, paper_id, error_message)
        """
        # Create new event loop for this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async method
        return loop.run_until_complete(self.process_single_pdf(pdf_path))
    
    async def process_batch(
        self,
        pdf_paths: List[Path]
    ) -> Dict[str, any]:
        """
        Process multiple PDFs in parallel.
        
        Args:
            pdf_paths: List of PDF paths
            
        Returns:
            Summary statistics
        """
        logger.info(f"Processing batch of {len(pdf_paths)} PDFs with {self.workers} workers")
        
        # Process with limited concurrency
        semaphore = asyncio.Semaphore(self.workers)
        
        async def process_with_semaphore(pdf_path: Path):
            async with semaphore:
                return await self.process_single_pdf(pdf_path)
        
        # Process all
        tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile statistics
        stats = {
            'total': len(pdf_paths),
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                stats['failed'] += 1
                stats['errors'].append({
                    'pdf': pdf_paths[i].name,
                    'error': str(result)
                })
            else:
                success, paper_id, error = result
                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    if error:
                        stats['errors'].append({
                            'pdf': pdf_paths[i].name,
                            'paper_id': paper_id,
                            'error': error
                        })
        
        return stats
    
    def _save_json(self, paper_id: str, data: Dict):
        """Save structured JSON output."""
        output_path = self.json_dir / f"{paper_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved JSON: {output_path}")
    
    def _save_raw_text(self, paper_id: str, text: str):
        """Save cleaned raw text."""
        output_path = self.text_dir / f"{paper_id}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.debug(f"Saved text: {output_path}")
    
    def _save_error(self, paper_id: str, pdf_path: Path, error: str):
        """Save error log."""
        output_path = self.error_dir / f"{paper_id}.log"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"PDF: {pdf_path}\n")
            f.write(f"Paper ID: {paper_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error}\n")
        
        logger.debug(f"Saved error log: {output_path}")
    
    async def close(self):
        """Close clients."""
        await self.grobid.close()


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PDF -> TEXT -> JSON pipeline for Phase 1"
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('ingestion'),
        help='Output directory (default: ingestion/)'
    )
    parser.add_argument(
        '--grobid-url',
        default='http://localhost:8070',
        help='GROBID server URL'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of concurrent workers'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to settings.yaml'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Find all PDFs
    pdf_paths = list(args.input_dir.glob('**/*.pdf'))
    
    if not pdf_paths:
        logger.error(f"No PDFs found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(pdf_paths)} PDFs in {args.input_dir}")
    
    # Initialize pipeline
    pipeline = PDFToTextPipeline(
        config_path=args.config,
        grobid_url=args.grobid_url,
        output_dir=args.output_dir,
        workers=args.workers
    )
    
    # Check GROBID health
    is_healthy = await pipeline.grobid.health_check()
    if not is_healthy:
        logger.error(f"❌ GROBID server not available at {args.grobid_url}")
        logger.error("Start GROBID with: docker run -p 8070:8070 lfoppiano/grobid:0.8.0")
        return
    
    logger.info(f"[OK] GROBID server is healthy")
    
    # Process all PDFs
    start_time = datetime.now()
    
    stats = await pipeline.process_batch(pdf_paths)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"Total PDFs:    {stats['total']}")
    print(f"[OK] Success:    {stats['success']}")
    print(f"❌ Failed:     {stats['failed']}")
    print(f"Duration:      {duration:.1f}s")
    print(f"Rate:          {stats['total']/duration:.1f} PDFs/sec")
    print(f"\nOutputs:")
    print(f"  JSON:        {pipeline.json_dir}")
    print(f"  Raw text:    {pipeline.text_dir}")
    print(f"  Errors:      {pipeline.error_dir}")
    
    if stats['errors']:
        print(f"\n[WARN]  {len(stats['errors'])} errors occurred:")
        for error in stats['errors'][:5]:  # Show first 5
            print(f"  - {error['pdf']}: {error['error']}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors'])-5} more")
    
    print("="*80 + "\n")
    
    # Cleanup
    await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
