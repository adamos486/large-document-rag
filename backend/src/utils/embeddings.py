import os
import numpy as np
from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer
import torch

from ..config.config import settings

class EmbeddingModel:
    """Embedding model utility for converting text to vector representations."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model to use. If None, uses the model 
                        specified in settings.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = self._load_model()
        
    def _load_model(self) -> SentenceTransformer:
        """Load the embedding model.
        
        Returns:
            SentenceTransformer model.
        """
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            model = SentenceTransformer(self.model_name, device=device)
            return model
        except Exception as e:
            print(f"Error loading embedding model {self.model_name}: {e}")
            print("Falling back to 'all-MiniLM-L6-v2' model")
            
            # Fall back to a smaller, more common model
            return SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
            
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True
        )
        
        # Convert to list of lists for compatibility with various vector dbs
        return embeddings.tolist()
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding vector.
        """
        return self.get_embeddings([text])[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Compute cosine similarity
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        
        return float(np.dot(emb1_np, emb2_np) / 
                   (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np)))
    
    def batch_embed_chunks(self, chunks: List[Any]) -> List[List[float]]:
        """Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of objects with a 'content' attribute containing text.
            
        Returns:
            List of embedding vectors.
        """
        if not chunks:
            return []
            
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        return self.get_embeddings(texts)
