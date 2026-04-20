"""
Auto-processing hook for paper orchestrator.

Automatically processes PDFs through Phase 1 pipeline as soon as they're downloaded.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List
import sys

# Handle imports
try:
    from .pdf_to_text import PDFToTextPipeline
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from pdf_to_text import PDFToTextPipeline

logger = logging.getLogger(__name__)


class AutoProcessor:
    """
    Automatically process downloaded PDFs through Phase 1 pipeline.
    
    Integrates with max_orchestrator to process papers immediately after download.
    """
    
    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        output_dir: Path = Path("ingestion"),
        workers: int = 4,
        auto_process: bool = True
    ):
        """
        Initialize auto-processor.
        
        Args:
            grobid_url: GROBID server URL
            output_dir: Output directory for processed files
            workers: Number of concurrent workers
            auto_process: Enable automatic processing
        """
        self.pipeline = None
        self.grobid_url = grobid_url
        self.output_dir = output_dir
        self.workers = workers
        self.auto_process = auto_process
        self.processed_files = set()
        
        if auto_process:
            logger.info("Auto-processing enabled")
    
    def initialize_sync(self):
        """Initialize pipeline synchronously (lazy loading)."""
        if not self.pipeline:
            self.pipeline = PDFToTextPipeline(
                grobid_url=self.grobid_url,
                output_dir=self.output_dir,
                workers=self.workers
            )
            logger.info("Pipeline initialized")
    
    async def initialize(self):
        """Initialize pipeline (lazy loading)."""
        if not self.pipeline:
            self.pipeline = PDFToTextPipeline(
                grobid_url=self.grobid_url,
                output_dir=self.output_dir,
                workers=self.workers
            )
            
            # Check GROBID health
            is_healthy = await self.pipeline.grobid.health_check()
            if not is_healthy:
                logger.warning(f"GROBID not available at {self.grobid_url}")
                logger.warning("Fallback extraction will be used")
    
    def process_downloaded_pdf(self, pdf_path: str) -> dict:
        """
        Process single PDF synchronously (for orchestrator integration).
        
        Args:
            pdf_path: Path to PDF file (string or Path)
            
        Returns:
            Result dict with success, json_path, text_path, extraction_method
        """
        pdf_path = Path(pdf_path)
        
        if not self.auto_process:
            return {'success': False, 'error': 'Auto-processing disabled'}
        
        # Skip if already processed
        if pdf_path in self.processed_files:
            paper_id = self.pipeline.generate_paper_id(pdf_path) if self.pipeline else pdf_path.stem
            json_path = self.output_dir / "processed_json" / f"{paper_id}.json"
            text_path = self.output_dir / "processed_text" / f"{paper_id}.txt"
            return {
                'success': True,
                'json_path': str(json_path),
                'text_path': str(text_path),
                'extraction_method': 'cached'
            }
        
        # Initialize if needed
        if not self.pipeline:
            self.initialize_sync()
        
        # Process PDF synchronously
        try:
            # Call the pipeline's sync method
            success, paper_id, error = self.pipeline.process_single_pdf_sync(pdf_path)
            
            if success:
                self.processed_files.add(pdf_path)
                json_path = self.pipeline.json_dir / f"{paper_id}.json"
                text_path = self.pipeline.text_dir / f"{paper_id}.txt"
                
                # Detect extraction method from JSON
                extraction_method = 'grobid'
                if json_path.exists():
                    import json
                    with open(json_path) as f:
                        data = json.load(f)
                        extraction_method = data.get('extraction_method', 'grobid')
                
                return {
                    'success': True,
                    'json_path': str(json_path),
                    'text_path': str(text_path),
                    'extraction_method': extraction_method
                }
            else:
                return {'success': False, 'error': error}
                
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def process_pdf(self, pdf_path: Path) -> bool:
        """
        Process single PDF through pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if successful
        """
        if not self.auto_process:
            return False
        
        # Skip if already processed
        if pdf_path in self.processed_files:
            return True
        
        # Initialize if needed
        if not self.pipeline:
            await self.initialize()
        
        # Process PDF
        success, paper_id, error = await self.pipeline.process_single_pdf(pdf_path)
        
        if success:
            self.processed_files.add(pdf_path)
            logger.info(f"Auto-processed: {pdf_path.name} -> {paper_id}")
        else:
            logger.error(f"Auto-processing failed: {pdf_path.name}: {error}")
        
        return success
    
    async def process_batch(self, pdf_paths: List[Path]) -> dict:
        """
        Process batch of PDFs.
        
        Args:
            pdf_paths: List of PDF paths
            
        Returns:
            Processing statistics
        """
        if not self.auto_process:
            return {'processed': 0, 'total': len(pdf_paths)}
        
        # Initialize if needed
        if not self.pipeline:
            await self.initialize()
        
        # Filter out already processed
        to_process = [p for p in pdf_paths if p not in self.processed_files]
        
        if not to_process:
            logger.info("All PDFs already processed")
            return {'processed': 0, 'total': len(pdf_paths), 'skipped': len(pdf_paths)}
        
        logger.info(f"Auto-processing {len(to_process)} PDFs")
        
        # Process batch
        stats = await self.pipeline.process_batch(to_process)
        
        # Update processed set
        for pdf_path in to_process:
            paper_id = self.pipeline.generate_paper_id(pdf_path)
            json_path = self.pipeline.json_dir / f"{paper_id}.json"
            if json_path.exists():
                self.processed_files.add(pdf_path)
        
        return stats
    
    async def close(self):
        """Close pipeline."""
        if self.pipeline:
            await self.pipeline.close()


# Singleton instance
_auto_processor = None


def get_auto_processor(
    grobid_url: str = "http://localhost:8070",
    output_dir: Path = Path("ingestion"),
    workers: int = 4,
    auto_process: bool = True
) -> AutoProcessor:
    """
    Get singleton auto-processor instance.
    
    Args:
        grobid_url: GROBID server URL
        output_dir: Output directory
        workers: Number of workers
        auto_process: Enable auto-processing
        
    Returns:
        AutoProcessor instance
    """
    global _auto_processor
    
    if _auto_processor is None:
        _auto_processor = AutoProcessor(
            grobid_url=grobid_url,
            output_dir=output_dir,
            workers=workers,
            auto_process=auto_process
        )
    
    return _auto_processor


async def process_downloaded_pdf(pdf_path: Path) -> bool:
    """
    Hook function: Process PDF immediately after download.
    
    Call this from orchestrator after successful PDF download.
    
    Args:
        pdf_path: Path to downloaded PDF
        
    Returns:
        True if processing successful
    """
    processor = get_auto_processor()
    return await processor.process_pdf(pdf_path)


async def process_downloaded_batch(pdf_paths: List[Path]) -> dict:
    """
    Hook function: Process batch of PDFs after download.
    
    Args:
        pdf_paths: List of downloaded PDF paths
        
    Returns:
        Processing statistics
    """
    processor = get_auto_processor()
    return await processor.process_batch(pdf_paths)


if __name__ == "__main__":
    # Test auto-processing
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-process PDFs")
    parser.add_argument('pdf_dir', type=Path, help='Directory with PDFs')
    parser.add_argument('--grobid-url', default='http://localhost:8070')
    parser.add_argument('--workers', type=int, default=4)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        processor = get_auto_processor(
            grobid_url=args.grobid_url,
            workers=args.workers
        )
        
        pdf_paths = list(args.pdf_dir.glob('**/*.pdf'))
        print(f"Found {len(pdf_paths)} PDFs")
        
        stats = await processor.process_batch(pdf_paths)
        
        print(f"\nProcessed: {stats['success']}/{stats['total']}")
        
        await processor.close()
    
    asyncio.run(main())
