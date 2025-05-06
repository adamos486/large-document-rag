"""
Advanced Financial Embeddings Model

This module provides specialized embedding models optimized for financial text, 
going beyond generic embeddings or standard models like FinBERT.

Key features:
1. Financial domain adaptation layer on top of base embeddings
2. Financial entity-aware embedding generation
3. Support for finance-specific contrastive learning
4. Metric learning optimization for financial similarity
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

from ..config.config import settings
from ..llm.financial_task_detector import financial_task_detector

logger = logging.getLogger(__name__)

class FinancialProjectionLayer(nn.Module):
    """Neural projection layer optimized for financial text embeddings."""
    
    def __init__(self, input_dim: int, projection_dim: int = None, dropout: float = 0.1):
        """
        Initialize a financial projection layer.
        
        Args:
            input_dim: Dimension of input embeddings
            projection_dim: Dimension of projected embeddings (defaults to input_dim)
            dropout: Dropout rate for regularization
        """
        super(FinancialProjectionLayer, self).__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim or input_dim
        
        # Create a financial domain adaptation layer
        self.financial_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, self.projection_dim)
        )
        
    def forward(self, embeddings):
        """Project embeddings through the financial adaptation layer."""
        return self.financial_projection(embeddings)
    
    def save(self, path: str):
        """Save the projection layer weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """Load the projection layer weights."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            return True
        return False


class FinancialEmbeddingModel:
    """
    Advanced embedding model optimized for financial documents.
    
    This model goes beyond simple embeddings by:
    1. Using financial term weighting
    2. Applying domain-specific projection
    3. Incorporating finance-specific knowledge
    """
    
    def __init__(
        self, 
        base_model_name: str = None,
        projection_dim: int = None,
        use_projection_layer: bool = True,
        use_entity_weighting: bool = True,
        device: str = None
    ):
        """
        Initialize the financial embedding model.
        
        Args:
            base_model_name: Base embedding model to use
            projection_dim: Dimension of projected embeddings
            use_projection_layer: Whether to use financial projection layer
            use_entity_weighting: Whether to use financial entity weighting
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.base_model_name = base_model_name or settings.EMBEDDING_MODEL
        self.use_projection_layer = use_projection_layer
        self.use_entity_weighting = use_entity_weighting
        
        # Set device (use GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the base embedding model
        logger.info(f"Loading base embedding model: {self.base_model_name}")
        try:
            # Try sentence-transformers first
            self.model = SentenceTransformer(self.base_model_name)
            self.embedding_type = "sentence_transformer"
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.warning(f"Falling back to HuggingFace transformers for embedding model: {e}")
            # Fallback to HuggingFace transformers
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModel.from_pretrained(self.base_model_name).to(self.device)
            self.embedding_type = "transformers"
            self.embedding_dim = self.model.config.hidden_size
        
        # Initialize financial projection layer if enabled
        if self.use_projection_layer:
            self.projection_dim = projection_dim or self.embedding_dim
            logger.info(f"Initializing financial projection layer: {self.embedding_dim} -> {self.projection_dim}")
            self.projection_layer = FinancialProjectionLayer(
                input_dim=self.embedding_dim,
                projection_dim=self.projection_dim
            ).to(self.device)
            
            # Try to load pre-trained financial projection weights
            projection_path = settings.DATA_DIR / 'models' / 'financial_projection.pt'
            if Path(projection_path).exists():
                self.projection_layer.load(projection_path)
                logger.info(f"Loaded financial projection weights from {projection_path}")
            else:
                logger.info("Financial projection layer initialized with random weights")
        
        # Financial term importance dictionary
        self._initialize_financial_term_weights()
        
    def _initialize_financial_term_weights(self):
        """Initialize the financial term importance weighting dictionary."""
        # These weights will be used to enhance embeddings for financial terms
        self.financial_term_weights = {
            # Financial statements and sections
            "balance sheet": 1.8, "income statement": 1.8, "cash flow statement": 1.8, 
            "financial statement": 1.7, "annual report": 1.6, 
            "quarterly report": 1.6, "10-k": 1.7, "10-q": 1.7,
            
            # Important financial metrics and terms
            "revenue": 1.5, "ebitda": 1.7, "net income": 1.6, "profit": 1.5, 
            "gross margin": 1.6, "operating margin": 1.6, "profit margin": 1.6,
            "cash flow": 1.6, "free cash flow": 1.7, "liquidity": 1.5, 
            "solvency": 1.5, "debt": 1.5, "leverage": 1.5,
            
            # Financial ratios (highly important in analysis)
            "ratio": 1.6, "p/e": 1.8, "price to earnings": 1.8, 
            "price-to-earnings": 1.8, "eps": 1.6, "roe": 1.7, 
            "return on equity": 1.7, "roi": 1.7, "return on investment": 1.7,
            "debt to equity": 1.7, "current ratio": 1.7, "quick ratio": 1.7,
            
            # Risk terms
            "risk": 1.5, "volatility": 1.5, "default": 1.6, "bankruptcy": 1.6, 
            "material weakness": 1.8, "going concern": 1.8, "internal control": 1.6,
            
            # M&A specific terms (highly relevant for due diligence)
            "acquisition": 1.7, "merger": 1.7, "due diligence": 1.8, 
            "synergy": 1.6, "target company": 1.7, "valuation": 1.7,
            "earn-out": 1.7, "purchase price": 1.6, "deal structure": 1.6
        }
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate a financially optimized embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding
        """
        if not text:
            return [0.0] * (self.projection_dim if self.use_projection_layer else self.embedding_dim)
        
        # Extract financial entities if entity weighting is enabled
        if self.use_entity_weighting:
            financial_entities = financial_task_detector.extract_financial_entities(text)
            has_financial_entities = len(financial_entities) > 0
        else:
            has_financial_entities = False
        
        # Generate base embedding
        if self.embedding_type == "sentence_transformer":
            base_embedding = self.model.encode(text, convert_to_tensor=True)
        else:
            # Using HuggingFace transformers directly
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                base_embedding = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # Apply financial term weighting if entities are present
        if has_financial_entities and self.use_entity_weighting:
            base_embedding = self._apply_financial_weighting(text, base_embedding)
        
        # Apply financial projection layer if enabled
        if self.use_projection_layer:
            # Convert to correct tensor type if needed
            if not isinstance(base_embedding, torch.Tensor):
                base_embedding = torch.tensor(base_embedding, device=self.device)
            elif base_embedding.device != self.device:
                base_embedding = base_embedding.to(self.device)
                
            # Apply projection
            with torch.no_grad():
                projected_embedding = self.projection_layer(base_embedding)
                
            # Convert back to list
            embedding = projected_embedding.cpu().numpy().tolist()
            if isinstance(embedding[0], list):  # Handle batch dimension if present
                embedding = embedding[0]
        else:
            # Convert tensor to list if needed
            if isinstance(base_embedding, torch.Tensor):
                embedding = base_embedding.cpu().numpy().tolist()
                if isinstance(embedding[0], list):  # Handle batch dimension if present
                    embedding = embedding[0]
            else:
                embedding = base_embedding
        
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate financially optimized embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Process in batches for efficiency
        embeddings = []
        batch_size = 32  # Adjust based on memory constraints
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                batch_embeddings.append(self.get_embedding(text))
                
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for transformers models."""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _apply_financial_weighting(self, text: str, embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply financial term weighting to enhance embeddings.
        
        Args:
            text: Original text
            embedding: Base embedding tensor
            
        Returns:
            Enhanced embedding tensor with financial term weighting
        """
        text_lower = text.lower()
        weight_sum = 0.0
        weight_count = 0
        
        # Check for each financial term
        for term, weight in self.financial_term_weights.items():
            if term in text_lower:
                weight_sum += weight
                weight_count += 1
        
        # Apply weighting if financial terms are found
        if weight_count > 0:
            # Average weight of found terms
            avg_weight = weight_sum / weight_count
            # Apply weighting factor (0.7 base + 0.3 weighted)
            weighted_embedding = 0.7 * embedding + 0.3 * (avg_weight * embedding)
            return weighted_embedding
        
        return embedding
    
    def measure_similarity(self, text1: str, text2: str) -> float:
        """
        Measure financial-aware similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.cpu().numpy()
            
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return float(similarity)
    
    def find_most_similar(self, query: str, candidates: List[str], top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar texts to a query from a list of candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_n: Number of top matches to return
            
        Returns:
            List of tuples with (index, similarity_score)
        """
        query_embedding = self.get_embedding(query)
        candidate_embeddings = self.get_embeddings(candidates)
        
        # Convert to numpy arrays if they are not already
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        if isinstance(candidate_embeddings[0], torch.Tensor):
            candidate_embeddings = [e.cpu().numpy() for e in candidate_embeddings]
            
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # Get top N matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # Return (index, score) tuples
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def fine_tune(self, financial_texts: List[str], steps: int = 100, learning_rate: float = 1e-4):
        """
        Fine-tune the financial projection layer on domain-specific texts.
        
        Args:
            financial_texts: List of financial texts for fine-tuning
            steps: Number of training steps
            learning_rate: Learning rate for optimization
        """
        if not self.use_projection_layer:
            logger.warning("Cannot fine-tune without projection layer enabled")
            return False
            
        if len(financial_texts) < 10:
            logger.warning("Not enough texts for meaningful fine-tuning")
            return False
        
        logger.info(f"Fine-tuning financial projection layer on {len(financial_texts)} texts")
        
        # Set projection layer to training mode
        self.projection_layer.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.projection_layer.parameters(), lr=learning_rate)
        
        # Generate base embeddings for all texts
        base_embeddings = []
        for text in financial_texts:
            if self.embedding_type == "sentence_transformer":
                emb = self.model.encode(text, convert_to_tensor=True).to(self.device)
            else:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    emb = self._mean_pooling(outputs, inputs['attention_mask'])
            base_embeddings.append(emb)
        
        # Training loop using contrastive learning
        for step in range(steps):
            total_loss = 0
            
            # Process data in batches
            batch_size = min(16, len(financial_texts))
            indices = np.random.choice(len(financial_texts), size=batch_size, replace=False)
            
            for i in indices:
                # Get anchor embedding
                anchor_emb = base_embeddings[i]
                
                # Forward pass through projection layer
                projected_anchor = self.projection_layer(anchor_emb)
                
                # Generate positive and negative samples
                positive_idx = np.random.choice([j for j in range(len(financial_texts)) if j != i])
                negative_indices = np.random.choice(
                    [j for j in range(len(financial_texts)) if j != i and j != positive_idx], 
                    size=3, 
                    replace=len(financial_texts) > 4
                )
                
                positive_emb = self.projection_layer(base_embeddings[positive_idx])
                negative_embs = [self.projection_layer(base_embeddings[j]) for j in negative_indices]
                
                # Calculate contrastive loss (InfoNCE loss)
                loss = self._contrastive_loss(projected_anchor, positive_emb, negative_embs)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Log progress
            if (step + 1) % 10 == 0:
                logger.info(f"Fine-tuning step {step + 1}/{steps}, Loss: {total_loss/batch_size:.4f}")
        
        # Save fine-tuned projection layer
        save_path = settings.DATA_DIR / 'models' / 'financial_projection.pt'
        os.makedirs(settings.DATA_DIR / 'models', exist_ok=True)
        self.projection_layer.save(str(save_path))
        logger.info(f"Saved fine-tuned financial projection layer to {save_path}")
        
        # Set projection layer back to evaluation mode
        self.projection_layer.eval()
        return True
    
    def _contrastive_loss(self, anchor, positive, negatives, temperature=0.07):
        """Calculate contrastive loss (InfoNCE) for financial embeddings."""
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=0)
        positive = F.normalize(positive, p=2, dim=0)
        negatives = [F.normalize(neg, p=2, dim=0) for neg in negatives]
        
        # Calculate logits
        positive_logit = torch.sum(anchor * positive) / temperature
        negative_logits = torch.tensor([torch.sum(anchor * neg) / temperature for neg in negatives])
        
        # Combine logits and calculate loss
        logits = torch.cat([positive_logit.unsqueeze(0), negative_logits])
        labels = torch.zeros(len(logits), device=self.device, dtype=torch.long)  # Positive sample at index 0
        
        return F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))


# Create singleton instance with default parameters
financial_embedding_model = FinancialEmbeddingModel(
    use_projection_layer=True,
    use_entity_weighting=True
)
