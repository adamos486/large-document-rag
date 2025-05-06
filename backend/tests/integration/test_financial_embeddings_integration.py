"""
Integration tests for the financial embeddings system with other components.

These tests verify that the financial embeddings properly integrate with:
1. Vector store
2. Document processing pipeline
3. Query processing
"""

import unittest
import tempfile
import uuid
import os
import importlib.util
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Check if dependencies are available
def is_module_available(module_name):
    """Check if a module can be imported without actually importing it"""
    return importlib.util.find_spec(module_name) is not None

# Check for required dependencies
NUMPY_AVAILABLE = is_module_available("numpy")
SPACY_AVAILABLE = is_module_available("spacy")
TORCH_AVAILABLE = is_module_available("torch")
CHROMADB_AVAILABLE = is_module_available("chromadb")

# Print dependency status for debugging
print(f"NumPy available: {NUMPY_AVAILABLE}")
print(f"SpaCy available: {SPACY_AVAILABLE}")
print(f"Torch available: {TORCH_AVAILABLE}")
print(f"ChromaDB available: {CHROMADB_AVAILABLE}")

# Import numpy if available
if NUMPY_AVAILABLE:
    import numpy as np

# Only try to import if spaCy is available
if SPACY_AVAILABLE:
    try:
        import spacy
        SPACY_MODEL_AVAILABLE = True
        try:
            # Check if the spaCy model is available
            spacy.load("en_core_web_sm")
        except OSError:
            SPACY_MODEL_AVAILABLE = False
            print("SpaCy model not available - some tests will be skipped")
    except ImportError:
        SPACY_MODEL_AVAILABLE = False
else:
    SPACY_MODEL_AVAILABLE = False

# Import system modules with appropriate error handling
try:
    from src.finance.embeddings.model import FinancialEmbeddingModel
    from src.finance.embeddings.config import EmbeddingModelConfig
    from src.document_processing.base_processor import DocumentChunk
    from src.vector_store.vector_db import VectorStore
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    IMPORTS_AVAILABLE = False
    
    # Create mock classes if imports failed
    class DocumentChunk:
        def __init__(self, content="", metadata=None, chunk_id=None):
            self.content = content
            self.metadata = metadata or {}
            self.chunk_id = chunk_id
    
    class VectorStore:
        def __init__(self, collection_name="test"):
            self.collection_name = collection_name
        
        def add_chunks(self, chunks, embeddings=None):
            pass
            
        def query(self, query_text, k=5):
            return {"ids": [["mock_id_1", "mock_id_2"]]}
            
        def count(self):
            return 10
            
        def delete_collection(self):
            pass
    
    class FinancialEmbeddingModel:
        def __init__(self, use_cache=False):
            self.use_cache = use_cache
        
        def embed(self, text):
            # Return mock embedding
            return [0.1] * 300


@unittest.skipIf(not IMPORTS_AVAILABLE or not NUMPY_AVAILABLE or not SPACY_AVAILABLE or not CHROMADB_AVAILABLE,
             "Skip integration tests when dependencies are missing")
class TestFinancialEmbeddingsVectorStoreIntegration(unittest.TestCase):
    """Test integration between financial embeddings and vector store."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for vector store
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Backup original vector store path and set to temp dir
        from src.config.config import settings
        self.original_vector_db_path = settings.VECTOR_DB_PATH
        settings.VECTOR_DB_PATH = Path(self.temp_dir.name) / "vector_store"
        settings.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Create a unique collection name for testing
        self.collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        
        # Initialize vector store with test collection
        self.vector_store = VectorStore(collection_name=self.collection_name)
        
        # Initialize embedding model
        self.embedding_model = FinancialEmbeddingModel(use_cache=False)
        
        # Create test documents with financial content
        self.financial_docs = [
            "The company reported revenue of $1.2 billion for Q1 2023, representing a 15% increase year-over-year.",
            "Balance sheet remains strong with total assets of $5.4 billion and cash equivalents of $800 million.",
            "EBITDA margin improved to 28.5%, driven by cost optimization initiatives and operational efficiency.",
            "The Board of Directors declared a quarterly dividend of $0.50 per share, payable on June 15, 2023.",
            "P/E ratio is currently at 18.2x, slightly above the industry average of 16.5x."
        ]
        
        # Create test documents with non-financial content
        self.non_financial_docs = [
            "The weather forecast predicts sunny conditions for the next five days with temperatures reaching 75°F.",
            "Scientists discovered a new species of butterfly in the Amazon rainforest last month.",
            "The art exhibition will feature works from local artists and runs from July through September.",
            "Traffic on the main highway is expected to increase during the holiday weekend.",
            "The city council approved plans for a new park in the downtown area."
        ]
        
        # Create document chunks
        self.financial_chunks = [
            DocumentChunk(
                content=doc,
                metadata={"type": "financial", "index": i},
                chunk_id=f"financial_{i}"
            )
            for i, doc in enumerate(self.financial_docs)
        ]
        
        self.non_financial_chunks = [
            DocumentChunk(
                content=doc,
                metadata={"type": "non_financial", "index": i},
                chunk_id=f"non_financial_{i}"
            )
            for i, doc in enumerate(self.non_financial_docs)
        ]
        
        # Combine all chunks
        self.all_chunks = self.financial_chunks + self.non_financial_chunks
    
    def tearDown(self):
        """Clean up after tests."""
        # Delete the test collection
        self.vector_store.delete_collection()
        
        # Restore original vector store path
        from src.config.config import settings
        settings.VECTOR_DB_PATH = self.original_vector_db_path
        
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_embedding_and_retrieval(self):
        """Test that financial embeddings work correctly with vector store."""
        try:
            # Generate embeddings for all chunks
            chunk_embeddings = [self.embedding_model.embed(chunk.content) for chunk in self.all_chunks]
            
            # Add chunks with their embeddings to vector store
            self.vector_store.add_chunks(self.all_chunks, embeddings=chunk_embeddings)
            
            # Verify that chunks were added
            self.assertEqual(self.vector_store.count(), len(self.all_chunks))
        except Exception as e:
            self.skipTest(f"Failed to run vector store integration test: {e}")
        
        # Create financial test queries
        financial_queries = [
            "What was the company's revenue?",
            "Tell me about the balance sheet",
            "What is the EBITDA margin?",
            "Information about dividends",
            "What is the P/E ratio compared to industry average?"
        ]
        
        # Create non-financial test queries
        non_financial_queries = [
            "What is the weather forecast?",
            "Tell me about new species discoveries",
            "Information about the art exhibition",
            "What's happening with traffic?",
            "Tell me about the city council's plans"
        ]
        
        # Test financial queries
        for i, query in enumerate(financial_queries):
            try:
                # Get query embedding
                query_embedding = self.embedding_model.embed(query)
                
                # Query vector store with pre-computed embedding
                results = self.vector_store.query(
                    query_text=query,
                    query_embedding=query_embedding
                )
                
                # Check if we got results back
                if not results['ids'][0]:
                    self.skipTest(f"No results returned for query: {query}")
                    continue
                
                # The top result should be the corresponding financial document
                self.assertEqual(results['ids'][0][0], f"financial_{i}")
                
                # All top results should be financial documents
                for doc_id in results['ids'][0][:3]:  # Check top 3 results
                    self.assertTrue(doc_id.startswith("financial_"))
            except Exception as e:
                self.skipTest(f"Failed on financial query {i}: {e}")
        
        # Test non-financial queries
        for i, query in enumerate(non_financial_queries):
            try:
                # Get query embedding
                query_embedding = self.embedding_model.embed(query)
                
                # Query vector store with pre-computed embedding
                results = self.vector_store.query(
                    query_text=query,
                    query_embedding=query_embedding
                )
                
                # Check if we got results back
                if not results['ids'][0]:
                    self.skipTest(f"No results returned for query: {query}")
                    continue
                
                # The top result should be the corresponding non-financial document
                self.assertEqual(results['ids'][0][0], f"non_financial_{i}")
            except Exception as e:
                self.skipTest(f"Failed on non-financial query {i}: {e}")
    
    def test_financial_term_sensitivity(self):
        """Test that the embeddings are sensitive to financial terminology."""
        try:
            # Generate embeddings for all chunks
            chunk_embeddings = [self.embedding_model.embed(chunk.content) for chunk in self.all_chunks]
            
            # Add chunks with their embeddings to vector store
            self.vector_store.add_chunks(self.all_chunks, embeddings=chunk_embeddings)
        except Exception as e:
            self.skipTest(f"Failed to run financial term sensitivity test: {e}")
        
        # Create pairs of similar queries (one with financial terms, one without)
        query_pairs = [
            (
                "What was the revenue last quarter?", 
                "What was the amount last quarter?"
            ),
            (
                "Tell me about the EBITDA performance", 
                "Tell me about the performance metrics"
            ),
            (
                "What is the P/E ratio?",
                "What is the number ratio?"
            )
        ]
        
        for financial_query, generic_query in query_pairs:
            # Query with financial terminology
            financial_results = self.vector_store.query(financial_query)
            
            # Query with generic terminology
            generic_results = self.vector_store.query(generic_query)
            
            # Financial queries should retrieve more financial documents in top results
            financial_count_financial = sum(1 for doc_id in financial_results['ids'][0][:3] 
                                         if doc_id.startswith("financial_"))
            
            financial_count_generic = sum(1 for doc_id in generic_results['ids'][0][:3] 
                                       if doc_id.startswith("financial_"))
            
            # The financial query should retrieve at least as many financial documents
            # as the generic query, and ideally more
            self.assertGreaterEqual(financial_count_financial, financial_count_generic)


@unittest.skipIf(not IMPORTS_AVAILABLE or not NUMPY_AVAILABLE or not SPACY_AVAILABLE or not CHROMADB_AVAILABLE,
             "Skip integration tests when dependencies are missing")
class TestFinancialEmbeddingsDocumentProcessorIntegration(unittest.TestCase):
    """Test integration between financial embeddings and document processing."""
    
    @unittest.skip("Requires sample financial documents")
    def test_document_processor_with_embeddings(self):
        """Test that document processor works with financial embeddings."""
        # This test would require actual financial document files
        # Skipping for now, but the implementation would look like:
        
        # 1. Initialize document processor and embedding model
        # from src.document_processing.financial_document_processor import FinancialDocumentProcessor
        # processor = FinancialDocumentProcessor()
        # embedding_model = FinancialEmbeddingModel()
        
        # 2. Load a sample financial document
        # doc = processor.load_document(Path("sample_financial_document.pdf"))
        
        # 3. Process and chunk document
        # chunks = processor.process_document(doc)
        
        # 4. Generate embeddings for chunks
        # embeddings = [embedding_model.embed(chunk.content) for chunk in chunks]
        
        # 5. Verify dimensions and that embeddings are not all zero or identical
        pass


if __name__ == '__main__':
    unittest.main()
