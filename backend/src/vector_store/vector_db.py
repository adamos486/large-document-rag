import os
import uuid
from typing import List, Dict, Any, Optional, Union
import numpy as np
import chromadb
from chromadb.config import Settings

from ..config.config import settings
from ..document_processing.base_processor import DocumentChunk


class VectorStore:
    """Vector database interface for storing and retrieving document chunks."""
    
    def __init__(self, collection_name: str = "default"):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use.
        """
        self.collection_name = collection_name
        self.initialize_db()
    
    def initialize_db(self):
        """Set up the vector database."""
        # Create ChromaDB client based on configuration
        try:
            # Try connecting to ChromaDB server
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_SERVER_HOST,
                port=settings.CHROMA_SERVER_PORT,
                ssl=settings.CHROMA_SERVER_SSL
            )
            print(f"\nConnected to ChromaDB server at {settings.CHROMA_SERVER_HOST}:{settings.CHROMA_SERVER_PORT}/api/v2\n")
        except Exception as e:
            print(f"\nERROR: Failed to connect to ChromaDB server: {str(e)}\n")
            print("Falling back to local SQLite storage...\n")
            self.client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH),
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": f"Collection for {self.collection_name}"}
            )
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: Optional[List[List[float]]] = None):
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects.
            embeddings: Optional list of embeddings. If not provided, embeddings will be 
                        computed by the embedding model specified in settings.
        """
        if not chunks:
            return
            
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add chunks to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Query the vector store for similar chunks.
        
        Args:
            query_text: The query text.
            n_results: Number of results to return.
            where: Optional metadata filter.
            where_document: Optional document content filter.
            query_embedding: Optional pre-computed embedding for the query text.
            
        Returns:
            Dictionary with query results including documents, metadatas, and distances.
        """
        try:
            # Query the collection with pre-computed embedding if provided
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            else:
                # Use query text if no embedding provided
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            
            return results
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error querying vector store: {e}")
            
            # Return an empty result structure on error
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: The ID of the chunk to retrieve.
            
        Returns:
            The DocumentChunk if found, otherwise None.
        """
        try:
            result = self.collection.get(ids=[chunk_id])
            
            if result['ids'] and len(result['ids']) > 0:
                content = result['documents'][0]
                metadata = result['metadatas'][0]
                
                # Reconstruct DocumentChunk
                return DocumentChunk(
                    content=content,
                    metadata=metadata,
                    chunk_id=chunk_id
                )
                
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
            
        return None
    
    def get_chunks_by_filter(self, where: Dict[str, Any]) -> List[DocumentChunk]:
        """Retrieve chunks that match a metadata filter.
        
        Args:
            where: Metadata filter conditions.
            
        Returns:
            List of DocumentChunk objects.
        """
        try:
            result = self.collection.get(where=where)
            
            chunks = []
            for i, chunk_id in enumerate(result['ids']):
                content = result['documents'][i]
                metadata = result['metadatas'][i]
                
                # Reconstruct DocumentChunk
                chunk = DocumentChunk(
                    content=content,
                    metadata=metadata,
                    chunk_id=chunk_id
                )
                
                chunks.append(chunk)
                
            return chunks
            
        except Exception as e:
            print(f"Error retrieving chunks with filter {where}: {e}")
            
        return []
    
    def delete_chunks(self, chunk_ids: List[str]):
        """Delete chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete.
        """
        try:
            self.collection.delete(ids=chunk_ids)
        except Exception as e:
            print(f"Error deleting chunks: {e}")
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"Error deleting collection: {e}")
            
    def count(self) -> int:
        """Get the number of chunks in the collection.
        
        Returns:
            Number of chunks in the collection.
        """
        return self.collection.count()
        
    def get_collections(self) -> List[str]:
        """Get all collection names in the database.
        
        Returns:
            List of collection names.
        """
        return [col.name for col in self.client.list_collections()]
