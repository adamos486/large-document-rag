from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

class Document:
    """Base document class to store document metadata and content."""
    def __init__(
        self,
        content: Any,
        metadata: Dict[str, Any] = None,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id
        if source:
            self.metadata["source"] = source

class DocumentChunk:
    """Represents a chunk of a document."""
    def __init__(
        self,
        content: Any,
        metadata: Dict[str, Any] = None,
        chunk_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.chunk_id = chunk_id
        self.embedding = embedding


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    def load_document(self, file_path: Path) -> Document:
        """Load a document from a file path."""
        pass
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks."""
        pass
    
    @abstractmethod
    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from a document."""
        pass
    
    def process(self, file_path: Path) -> List[DocumentChunk]:
        """Process a document from start to finish."""
        document = self.load_document(file_path)
        document.metadata.update(self.extract_metadata(document))
        return self.chunk_document(document)
