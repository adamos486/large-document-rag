"""
Base interfaces for document chunking strategies.

This module provides the core abstractions for all chunking strategies,
with a focus on preserving document semantics during the chunking process.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path

from ..base_processor import DocumentChunk, Document


class BaseChunker(ABC):
    """
    Abstract base class for all document chunking strategies.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 respect_semantics: bool = True):
        """
        Initialize the chunker with configuration parameters.
        
        Args:
            chunk_size: Target size of chunks in characters/tokens
            chunk_overlap: Number of characters/tokens to overlap between chunks
            respect_semantics: Whether to respect semantic boundaries when chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_semantics = respect_semantics
    
    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks based on the strategy.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
        """
        pass
    
    def get_chunk_id(self, doc_id: str, index: int) -> str:
        """Generate a chunk ID from document ID and chunk index."""
        return f"{doc_id}_chunk_{index}"
    
    def create_chunk(self, 
                    content: str, 
                    metadata: Dict[str, Any], 
                    doc_id: str, 
                    chunk_index: int) -> DocumentChunk:
        """
        Create a document chunk with proper metadata.
        
        Args:
            content: The chunk content
            metadata: The document metadata to extend
            doc_id: Document identifier
            chunk_index: Index of this chunk within the document
            
        Returns:
            A DocumentChunk instance
        """
        chunk_id = self.get_chunk_id(doc_id, chunk_index)
        
        # Create basic chunk metadata
        chunk_metadata = {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "doc_id": doc_id,
        }
        
        # Add document metadata
        if metadata:
            # Don't overwrite chunk-specific metadata
            for key, value in metadata.items():
                if key not in chunk_metadata:
                    chunk_metadata[key] = value
        
        return DocumentChunk(
            content=content,
            metadata=chunk_metadata,
            chunk_id=chunk_id
        )


class TextChunker(BaseChunker):
    """Base class for text document chunking strategies."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 respect_semantics: bool = True,
                 separator: str = "\n\n"):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            respect_semantics: Whether to respect semantic boundaries
            separator: Default separator to use for chunking
        """
        super().__init__(chunk_size, chunk_overlap, respect_semantics)
        self.separator = separator
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split a text document into chunks.
        
        This implementation provides a basic text chunking strategy
        that splits on separators while respecting target chunk size.
        Subclasses should override this for more specific behavior.
        """
        content = document.content
        metadata = document.metadata
        doc_id = document.doc_id
        
        # Split content on separator
        segments = content.split(self.separator)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for segment in segments:
            # Skip empty segments
            if not segment.strip():
                continue
                
            # If adding this segment would exceed the chunk size
            # and we already have content, create a new chunk
            if (len(current_chunk) + len(segment) > self.chunk_size and 
                current_chunk.strip()):
                
                # Create chunk
                chunks.append(self.create_chunk(
                    current_chunk, metadata, doc_id, chunk_index
                ))
                chunk_index += 1
                
                # Start new chunk, possibly with overlap
                if self.respect_semantics:
                    # With semantic respect, we start fresh with the new segment
                    current_chunk = segment
                else:
                    # Without semantic respect, we might split mid-segment
                    # Here we implement a basic overlap mechanism
                    overlap_point = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_point:] + self.separator + segment
            else:
                # Add separator if the chunk isn't empty
                if current_chunk:
                    current_chunk += self.separator
                current_chunk += segment
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self.create_chunk(
                current_chunk, metadata, doc_id, chunk_index
            ))
        
        return chunks


class ChunkerFactory:
    """Factory for creating appropriate chunkers based on document types."""
    
    @staticmethod
    def get_chunker(document_type: str, **kwargs) -> BaseChunker:
        """
        Get a chunker instance appropriate for the document type.
        
        Args:
            document_type: Type of document (e.g., "financial", "text", "tabular")
            **kwargs: Additional parameters to pass to the chunker
            
        Returns:
            An instance of a BaseChunker subclass
        """
        # This will be implemented by the subclass that imports all chunkers
        raise NotImplementedError("ChunkerFactory must be implemented by a subclass")
