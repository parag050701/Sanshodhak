"""
Document Chunking with Smart Splitting
Handles text chunking with semantic awareness
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import nltk
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class Chunk:
    """Document chunk with metadata"""
    text: str
    chunk_id: int
    paper_id: str
    start_char: int
    end_char: int
    metadata: Dict


class SmartChunker:
    """Smart document chunker with semantic awareness"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        use_semantic_split: bool = True
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            chunk_overlap: Overlap between chunks
            use_semantic_split: Use semantic boundaries (sentences)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_split = use_semantic_split
        
    def chunk_text(
        self,
        text: str,
        paper_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text with semantic boundaries
        
        Args:
            text: Text to chunk
            paper_id: Paper identifier
            metadata: Additional metadata
            
        Returns:
            List of chunks
        """
        if not text.strip():
            return []
        
        metadata = metadata or {}
        
        if self.use_semantic_split:
            return self._semantic_chunk(text, paper_id, metadata)
        else:
            return self._simple_chunk(text, paper_id, metadata)
    
    def _semantic_chunk(
        self,
        text: str,
        paper_id: str,
        metadata: Dict
    ) -> List[Chunk]:
        """Chunk by semantic boundaries (sentences)"""
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_end = chunk_start + len(chunk_text)
                
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata=metadata
                ))
                
                # Keep overlap sentences
                overlap_tokens = 0
                overlap_start = len(current_chunk) - 1
                while overlap_start >= 0 and overlap_tokens < self.chunk_overlap:
                    overlap_tokens += len(current_chunk[overlap_start].split())
                    overlap_start -= 1
                
                overlap_start = max(0, overlap_start + 1)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s.split()) for s in current_chunk)
                chunk_start = chunk_end - len(" ".join(current_chunk))
                chunk_id += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                paper_id=paper_id,
                start_char=chunk_start,
                end_char=chunk_start + len(chunk_text),
                metadata=metadata
            ))
        
        return chunks
    
    def _simple_chunk(
        self,
        text: str,
        paper_id: str,
        metadata: Dict
    ) -> List[Chunk]:
        """Simple sliding window chunking"""
        words = text.split()
        chunks = []
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                paper_id=paper_id,
                start_char=sum(len(w) + 1 for w in words[:start_idx]),
                end_char=sum(len(w) + 1 for w in words[:end_idx]),
                metadata=metadata
            ))
            
            start_idx += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        return chunks


class PaperChunker:
    """Chunk academic papers with structure awareness"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128
    ):
        self.chunker = SmartChunker(chunk_size, chunk_overlap)
    
    def chunk_paper(
        self,
        text: str,
        paper_id: str,
        paper_metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk paper with section awareness
        
        Args:
            text: Paper text
            paper_id: Paper identifier
            paper_metadata: Paper metadata (title, authors, etc.)
            
        Returns:
            List of chunks with enhanced metadata
        """
        # Extract sections if available
        sections = self._extract_sections(text)
        
        all_chunks = []
        for section_name, section_text in sections:
            metadata = {
                "section": section_name,
                **(paper_metadata or {})
            }
            
            chunks = self.chunker.chunk_text(
                section_text,
                paper_id,
                metadata
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _extract_sections(self, text: str) -> List[Tuple[str, str]]:
        """Extract paper sections"""
        # Common section headers
        section_pattern = r'\n\s*(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References|Appendix|Background|Related Work|Experiments?|Evaluation|Implementation|Future Work)\s*\n'
        
        matches = list(re.finditer(section_pattern, text, re.IGNORECASE))
        
        if not matches:
            return [("full_text", text)]
        
        sections = []
        for i, match in enumerate(matches):
            section_name = match.group(1).lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            
            if section_text:
                sections.append((section_name, section_text))
        
        return sections
