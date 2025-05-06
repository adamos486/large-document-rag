import os
import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import re
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from .base_processor import DocumentChunk

# Set up logging
logger = logging.getLogger(__name__)

class FinancialIndexer:
    """
    Intelligent indexer for financial documents used in M&A due diligence.
    Provides enhanced search capabilities beyond basic vector similarity.
    """
    
    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize the financial indexer.
        
        Args:
            index_path: Directory path to store index files.
        """
        self.index_path = index_path
        if self.index_path and not self.index_path.exists():
            self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Financial entity extraction patterns
        self.entity_patterns = {
            "companies": r'(?:[A-Z][a-zA-Z0-9\s&,]+(?:Inc|LLC|Ltd|Corporation|Corp|Company|Co|LP|LLP|SA|GmbH|Plc|AG|NV)\.?)',
            "monetary_values": r'(?:\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|m|b|M|B))?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|CAD|AUD))',
            "percentages": r'(?:\b\d+(?:\.\d+)?%\b)',
            "dates": r'(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b)',
            "financial_metrics": r'(?:revenue|EBITDA|net income|profit|earnings|assets|liabilities|equity|P/E ratio|market cap)'
        }
        
        # Financial term dictionary for synonym mapping
        self.financial_term_map = {
            "revenue": ["sales", "turnover", "income", "earnings", "proceeds"],
            "profit": ["earnings", "net income", "bottom line", "gain", "surplus"],
            "liability": ["debt", "obligation", "payable", "responsibility"],
            "asset": ["property", "holding", "investment", "resource"],
            "merger": ["acquisition", "consolidation", "takeover", "integration", "combination"],
            "shareholder": ["stockholder", "investor", "owner", "equity holder"],
            "dividend": ["payout", "distribution", "return", "yield"],
            "capital": ["funds", "financing", "investment", "money", "equity"],
            "valuation": ["appraisal", "assessment", "evaluation", "worth", "value"],
            "dilution": ["watering down", "reduction", "decrease", "diminution"]
        }
        
        self.reverse_term_map = {}
        for canonical, synonyms in self.financial_term_map.items():
            for synonym in synonyms:
                self.reverse_term_map[synonym.lower()] = canonical
        
        # Initialize document index
        self.document_index = {}
        self.entity_index = {entity_type: {} for entity_type in self.entity_patterns}
        self.topic_index = {}
        
        # For topic modeling
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd_model = TruncatedSVD(n_components=50)
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        
        # Load existing index if available
        if self.index_path:
            self._load_index()
    
    def index_chunks(self, chunks: List[DocumentChunk]):
        """
        Index a list of document chunks.
        
        Args:
            chunks: List of document chunks to index.
        """
        if not chunks:
            return
        
        # Extract document texts for topic modeling
        doc_texts = [chunk.content for chunk in chunks]
        
        # Check if we have enough documents for topic modeling
        if len(doc_texts) >= 5:
            # Perform topic modeling
            try:
                self._extract_topics(chunks, doc_texts)
            except Exception as e:
                logger.error(f"Error during topic modeling: {e}")
        
        # Index each chunk
        for chunk in chunks:
            self._index_chunk(chunk)
        
        # Save index if path is specified
        if self.index_path:
            self._save_index()
    
    def _index_chunk(self, chunk: DocumentChunk):
        """
        Index a single document chunk.
        
        Args:
            chunk: The document chunk to index.
        """
        # Add to document index
        chunk_id = chunk.chunk_id
        doc_id = chunk.metadata.get("doc_id", "unknown")
        
        # Create document entry if it doesn't exist
        if doc_id not in self.document_index:
            self.document_index[doc_id] = {
                "chunks": set(),
                "metadata": chunk.metadata.copy(),
                "entities": {entity_type: set() for entity_type in self.entity_patterns}
            }
        
        # Add chunk to document
        self.document_index[doc_id]["chunks"].add(chunk_id)
        
        # Extract and index entities
        content = chunk.content
        for entity_type, pattern in self.entity_patterns.items():
            entities = set(re.findall(pattern, content))
            
            # Add entities to document
            self.document_index[doc_id]["entities"][entity_type].update(entities)
            
            # Index each entity
            for entity in entities:
                entity_key = entity.lower()
                if entity_key not in self.entity_index[entity_type]:
                    self.entity_index[entity_type][entity_key] = set()
                self.entity_index[entity_type][entity_key].add(chunk_id)
    
    def _extract_topics(self, chunks: List[DocumentChunk], texts: List[str]):
        """
        Extract topics from document chunks using topic modeling.
        
        Args:
            chunks: List of document chunks.
            texts: List of document texts corresponding to chunks.
        """
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Reduce dimensionality
            reduced_data = self.svd_model.fit_transform(tfidf_matrix)
            
            # Cluster documents
            cluster_labels = self.kmeans.fit_predict(reduced_data)
            
            # Get top terms for each cluster
            feature_names = self.vectorizer.get_feature_names_out()
            centroids = self.kmeans.cluster_centers_
            
            # Get top terms for each cluster/topic
            topic_terms = []
            for i in range(centroids.shape[0]):
                top_indices = np.argsort(centroids[i])[::-1][:10]
                top_terms = [feature_names[idx] for idx in top_indices]
                topic_terms.append(top_terms)
            
            # Assign topics to chunks
            for i, (chunk, label) in enumerate(zip(chunks, cluster_labels)):
                topic_id = f"topic_{label}"
                terms = topic_terms[label]
                
                # Add topic to chunk metadata
                chunk.metadata["topic_id"] = topic_id
                chunk.metadata["topic_terms"] = terms
                
                # Add to topic index
                if topic_id not in self.topic_index:
                    self.topic_index[topic_id] = {
                        "terms": terms,
                        "chunks": set()
                    }
                
                self.topic_index[topic_id]["chunks"].add(chunk.chunk_id)
        
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, max_results: int = 10) -> List[str]:
        """
        Search for document chunks relevant to a query.
        
        Args:
            query: Search query.
            filters: Optional filters to apply.
            max_results: Maximum number of results to return.
            
        Returns:
            List of chunk IDs matching the query.
        """
        if not query:
            return []
        
        matching_chunks = set()
        
        # Extract entities from query
        query_entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            entities = set(re.findall(pattern, query))
            if entities:
                query_entities[entity_type] = entities
        
        # Match by entities
        for entity_type, entities in query_entities.items():
            for entity in entities:
                entity_key = entity.lower()
                if entity_key in self.entity_index[entity_type]:
                    matching_chunks.update(self.entity_index[entity_type][entity_key])
        
        # Apply synonym expansion to query
        expanded_query = self._expand_financial_terms(query)
        
        # If no matches by entities, try keyword matching
        if not matching_chunks:
            # Simple keyword matching for now
            keywords = set(expanded_query.lower().split())
            for doc_id, doc_info in self.document_index.items():
                for chunk_id in doc_info["chunks"]:
                    if any(keyword in self.reverse_term_map for keyword in keywords):
                        matching_chunks.add(chunk_id)
        
        # Apply filters if provided
        if filters:
            filtered_chunks = set()
            for chunk_id in matching_chunks:
                # Find document ID for this chunk
                doc_id = None
                for d_id, doc_info in self.document_index.items():
                    if chunk_id in doc_info["chunks"]:
                        doc_id = d_id
                        break
                
                if doc_id:
                    # Check if document matches all filters
                    doc_info = self.document_index[doc_id]
                    matches_all = True
                    
                    for filter_key, filter_value in filters.items():
                        if filter_key in doc_info["metadata"]:
                            if doc_info["metadata"][filter_key] != filter_value:
                                matches_all = False
                                break
                    
                    if matches_all:
                        filtered_chunks.add(chunk_id)
            
            matching_chunks = filtered_chunks
        
        # Limit results
        return list(matching_chunks)[:max_results]
    
    def _expand_financial_terms(self, query: str) -> str:
        """
        Expand financial terms in the query using synonyms.
        
        Args:
            query: Original query string.
            
        Returns:
            Expanded query with financial term synonyms.
        """
        expanded_query = query
        words = query.lower().split()
        
        for word in words:
            if word in self.reverse_term_map:
                canonical = self.reverse_term_map[word]
                synonyms = " ".join(self.financial_term_map[canonical])
                expanded_query += f" {synonyms}"
        
        return expanded_query
    
    def get_related_documents(self, doc_id: str, max_results: int = 5) -> List[str]:
        """
        Find documents related to a given document ID.
        
        Args:
            doc_id: Document ID to find related documents for.
            max_results: Maximum number of results to return.
            
        Returns:
            List of related document IDs.
        """
        if doc_id not in self.document_index:
            return []
        
        related_docs = {}
        doc_info = self.document_index[doc_id]
        
        # Find documents with similar entities
        for entity_type, entities in doc_info["entities"].items():
            for entity in entities:
                entity_key = entity.lower()
                if entity_key in self.entity_index[entity_type]:
                    for chunk_id in self.entity_index[entity_type][entity_key]:
                        # Find document for this chunk
                        for other_doc_id, other_doc_info in self.document_index.items():
                            if other_doc_id != doc_id and chunk_id in other_doc_info["chunks"]:
                                if other_doc_id not in related_docs:
                                    related_docs[other_doc_id] = 0
                                related_docs[other_doc_id] += 1
        
        # Sort by relevance
        sorted_docs = sorted(related_docs.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:max_results]]
    
    def get_topic_summary(self, topic_id: str) -> Dict[str, Any]:
        """
        Get summary information about a topic.
        
        Args:
            topic_id: Topic ID to get summary for.
            
        Returns:
            Dictionary with topic summary information.
        """
        if topic_id not in self.topic_index:
            return {}
        
        topic_info = self.topic_index[topic_id]
        
        # Calculate documents in this topic
        doc_ids = set()
        for chunk_id in topic_info["chunks"]:
            for doc_id, doc_info in self.document_index.items():
                if chunk_id in doc_info["chunks"]:
                    doc_ids.add(doc_id)
        
        return {
            "topic_id": topic_id,
            "terms": topic_info["terms"],
            "document_count": len(doc_ids),
            "chunk_count": len(topic_info["chunks"]),
            "documents": list(doc_ids)
        }
    
    def _save_index(self):
        """Save index to disk."""
        try:
            # Convert sets to lists for JSON serialization
            serialized_doc_index = {}
            for doc_id, doc_info in self.document_index.items():
                serialized_doc_info = {
                    "chunks": list(doc_info["chunks"]),
                    "metadata": doc_info["metadata"],
                    "entities": {
                        entity_type: list(entities) 
                        for entity_type, entities in doc_info["entities"].items()
                    }
                }
                serialized_doc_index[doc_id] = serialized_doc_info
            
            serialized_entity_index = {}
            for entity_type, entity_dict in self.entity_index.items():
                serialized_entity_dict = {
                    entity: list(chunks) for entity, chunks in entity_dict.items()
                }
                serialized_entity_index[entity_type] = serialized_entity_dict
            
            serialized_topic_index = {}
            for topic_id, topic_info in self.topic_index.items():
                serialized_topic_info = {
                    "terms": topic_info["terms"],
                    "chunks": list(topic_info["chunks"])
                }
                serialized_topic_index[topic_id] = serialized_topic_info
            
            # Save to files
            with open(self.index_path / "document_index.json", 'w') as f:
                json.dump(serialized_doc_index, f)
            
            with open(self.index_path / "entity_index.json", 'w') as f:
                json.dump(serialized_entity_index, f)
            
            with open(self.index_path / "topic_index.json", 'w') as f:
                json.dump(serialized_topic_index, f)
            
            # Save vectorizer and models
            # Note: This requires scikit-learn joblib serialization
            from joblib import dump
            dump(self.vectorizer, self.index_path / "vectorizer.pkl")
            dump(self.svd_model, self.index_path / "svd_model.pkl")
            dump(self.kmeans, self.index_path / "kmeans.pkl")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_index(self):
        """Load index from disk."""
        try:
            # Check if index files exist
            doc_index_path = self.index_path / "document_index.json"
            entity_index_path = self.index_path / "entity_index.json"
            topic_index_path = self.index_path / "topic_index.json"
            
            if doc_index_path.exists() and entity_index_path.exists() and topic_index_path.exists():
                # Load document index
                with open(doc_index_path, 'r') as f:
                    serialized_doc_index = json.load(f)
                
                # Convert lists back to sets
                for doc_id, doc_info in serialized_doc_index.items():
                    self.document_index[doc_id] = {
                        "chunks": set(doc_info["chunks"]),
                        "metadata": doc_info["metadata"],
                        "entities": {
                            entity_type: set(entities) 
                            for entity_type, entities in doc_info["entities"].items()
                        }
                    }
                
                # Load entity index
                with open(entity_index_path, 'r') as f:
                    serialized_entity_index = json.load(f)
                
                for entity_type, entity_dict in serialized_entity_index.items():
                    self.entity_index[entity_type] = {
                        entity: set(chunks) for entity, chunks in entity_dict.items()
                    }
                
                # Load topic index
                with open(topic_index_path, 'r') as f:
                    serialized_topic_index = json.load(f)
                
                for topic_id, topic_info in serialized_topic_index.items():
                    self.topic_index[topic_id] = {
                        "terms": topic_info["terms"],
                        "chunks": set(topic_info["chunks"])
                    }
                
                # Load vectorizer and models
                from joblib import load
                vectorizer_path = self.index_path / "vectorizer.pkl"
                svd_path = self.index_path / "svd_model.pkl"
                kmeans_path = self.index_path / "kmeans.pkl"
                
                if vectorizer_path.exists() and svd_path.exists() and kmeans_path.exists():
                    self.vectorizer = load(vectorizer_path)
                    self.svd_model = load(svd_path)
                    self.kmeans = load(kmeans_path)
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            # Initialize empty indices if load fails
            self.document_index = {}
            self.entity_index = {entity_type: {} for entity_type in self.entity_patterns}
            self.topic_index = {}
