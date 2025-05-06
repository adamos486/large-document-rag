import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import time
import uuid

from ..config.config import settings
from .base_agent import Agent
from ..document_processing.base_processor import DocumentChunk
from ..document_processing.financial_document_processor import FinancialDocumentProcessor
from ..document_processing.financial_indexer import FinancialIndexer
from ..utils.embeddings import EmbeddingModel
from ..vector_store.vector_db import VectorStore

# Set up logging
logger = logging.getLogger(__name__)

class DocumentProcessorAgent(Agent):
    """Agent for processing documents and storing them in the vector database."""
    
    def __init__(
        self, 
        file_path: Union[str, Path],
        collection_name: str = "default",
        compute_embeddings: bool = True,
        agent_id: Optional[str] = None,
        use_indexer: bool = True
    ):
        """Initialize the document processor agent.
        
        Args:
            file_path: Path to the document to process.
            collection_name: Name of the vector database collection to use.
            compute_embeddings: Whether to compute embeddings for the document chunks.
            agent_id: Unique identifier for this agent.
            use_indexer: Whether to use the financial indexer for enhanced search capabilities.
        """
        super().__init__(agent_id=agent_id, name="DocumentProcessorAgent")
        self.file_path = Path(file_path)
        self.collection_name = collection_name
        self.compute_embeddings = compute_embeddings
        self.use_indexer = use_indexer
        
        # Check if file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        # Determine processor type based on file extension
        self.processor = self._get_processor_for_file()
        
        if self.compute_embeddings:
            self.embedding_model = EmbeddingModel()
            
        self.vector_store = VectorStore(collection_name=self.collection_name)
        
        # Initialize financial indexer if enabled
        if self.use_indexer:
            index_path = settings.DATA_DIR / 'financial_indices' / self.collection_name
            self.indexer = FinancialIndexer(index_path=index_path)
    
    def _get_processor_for_file(self):
        """Determine the appropriate document processor based on file extension."""
        file_ext = self.file_path.suffix.lower()
        
        # Supported financial document formats
        supported_formats = [
            '.pdf', '.docx', '.xlsx', '.xls', '.csv', '.tsv', '.txt',
            '.ppt', '.pptx', '.json', '.html', '.xml', '.md'
        ]
        
        if file_ext in supported_formats:
            return FinancialDocumentProcessor()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Process the document and store it in the vector database.
        
        Returns:
            Dictionary with processing results.
        """
        logger.info(f"Processing document: {self.file_path}")
        
        try:
            # Process the document
            start_time = time.time()
            chunks = self.processor.process(self.file_path)
            processing_time = time.time() - start_time
            
            logger.info(f"Document processed into {len(chunks)} chunks in {processing_time:.2f}s")
            
            # Compute embeddings if required
            if self.compute_embeddings and chunks:
                start_time = time.time()
                embeddings = self.embedding_model.batch_embed_chunks(chunks)
                embedding_time = time.time() - start_time
                
                logger.info(f"Embeddings computed in {embedding_time:.2f}s")
            else:
                embeddings = None
                embedding_time = 0
            
            # Store in vector database
            start_time = time.time()
            self.vector_store.add_chunks(chunks, embeddings)
            storage_time = time.time() - start_time
            
            logger.info(f"Chunks stored in vector database in {storage_time:.2f}s")
            
            # Index chunks if indexer is enabled
            if self.use_indexer:
                index_start_time = time.time()
                self.indexer.index_chunks(chunks)
                index_time = time.time() - index_start_time
                logger.info(f"Chunks indexed in {index_time:.2f}s")
            
            # Return processing statistics
            result = {
                "document_path": str(self.file_path),
                "document_type": self.file_path.suffix,
                "chunk_count": len(chunks),
                "collection_name": self.collection_name,
                "processing_time": processing_time,
                "embedding_time": embedding_time,
                "storage_time": storage_time,
                "indexing_time": index_time if self.use_indexer else 0,
                "total_time": processing_time + embedding_time + storage_time + (index_time if self.use_indexer else 0),
                "chunk_sample": [c.metadata for c in chunks[:3]] if chunks else [],
                "document_category": chunks[0].metadata.get("doc_category", "unknown") if chunks else "unknown"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {self.file_path}: {e}")
            raise
