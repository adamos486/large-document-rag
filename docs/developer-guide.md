# Developer Guide

This guide provides detailed information about the Financial Due Diligence RAG system's architecture, components, and how to extend or customize the system for your specific needs.

## System Architecture

The Financial Due Diligence RAG system follows a modular, multi-agent architecture designed for scalability and extensibility. Here's an overview of the major components:

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│     REST API      │────▶│    Orchestrator   │────▶│  Document Agent   │
│                   │     │                   │     │                   │
└───────────────────┘     └────────┬──────────┘     └────────┬──────────┘
                                   │                          │
                                   │                          ▼
                                   │             ┌───────────────────────┐
                                   │             │ Financial Document    │
                                   │             │ Processor             │
                                   │             └───────────┬───────────┘
                                   │                         │
                                   │                         ▼
                                   │             ┌───────────────────────┐
                                   │             │ Financial Indexer     │
                                   │             └───────────┬───────────┘
                                   │                         │
                                   │                         ▼
                                   │             ┌───────────────────────┐
                                   │             │ Vector Database       │
                                   │             └───────────────────────┘
                                   │
                                   ▼
                          ┌───────────────────┐
                          │                   │
                          │    Query Agent    │◀────┐
                          │                   │     │
                          └────────┬──────────┘     │
                                   │                │
                                   ▼                │
                          ┌───────────────────┐     │
                          │  LLM Provider     │     │
                          │  Factory          │     │
                          └────────┬──────────┘     │
                                   │                │
           ┌───────────────────────┴───────────┐    │
           ▼                       ▼           ▼    │
┌───────────────────┐    ┌───────────────────┐      │
│                   │    │                   │      │
│   OpenAI LLM      │    │  Anthropic LLM    │      │
│                   │    │                   │      │
└───────────────────┘    └───────────────────┘      │
                                                    │
                         ┌───────────────────┐      │
                         │                   │      │
                         │ Vector Database   │──────┘
                         │                   │
                         └───────────────────┘
```

### Core Components

1. **REST API Layer** (`src/api/`)
   - Handles HTTP requests and responses
   - Manages file uploads and queries
   - Implements background task processing

2. **Orchestrator** (`src/agents/orchestrator.py`)
   - Coordinates the activities of other agents
   - Manages task scheduling and parallel processing
   - Maintains system state and task tracking

3. **Document Processor Agent** (`src/agents/document_processor_agent.py`)
   - Handles document loading and processing
   - Manages chunking and embedding generation
   - Coordinates with specialized document processors

4. **Financial Document Processor** (`src/document_processing/financial_document_processor.py`)
   - Implements specialized processing for financial documents
   - Extracts financial entities and metadata
   - Handles various document formats (PDF, Excel, etc.)

5. **Financial Indexer** (`src/document_processing/financial_indexer.py`)
   - Creates intelligent indices for financial documents
   - Extracts and categorizes financial entities
   - Enhances retrieval capabilities for financial queries

6. **Query Agent** (`src/agents/query_agent.py`)
   - Processes user queries
   - Retrieves relevant document chunks
   - Generates responses using LLMs

7. **LLM Provider Factory** (`src/llm/llm_provider.py`)
   - Manages multiple LLM providers
   - Implements provider-specific optimizations
   - Enables hybrid task routing

8. **Vector Database** (`src/vector_store/vector_db.py`)
   - Stores and retrieves document chunks
   - Implements vector similarity search
   - Handles metadata filtering

## Code Organization

```
src/
│
├── agents/                    # Multi-agent system components
│   ├── base_agent.py          # Abstract base agent class
│   ├── document_processor_agent.py
│   ├── query_agent.py
│   └── orchestrator.py        # Coordination layer
│
├── api/                       # API endpoints and models
│   ├── app.py                 # FastAPI application
│   ├── models.py              # Pydantic models for requests/responses
│   └── routes.py              # API route definitions
│
├── config/                    # Configuration
│   └── config.py              # Settings and configuration
│
├── document_processing/       # Document processing components
│   ├── base_processor.py      # Abstract base processor
│   ├── financial_document_processor.py
│   └── financial_indexer.py   # Financial entity extraction
│
├── llm/                       # LLM integration
│   ├── __init__.py
│   └── llm_provider.py        # LLM provider factory
│
├── utils/                     # Utility functions
│   ├── embeddings.py          # Embedding models
│   └── helpers.py             # Helper functions
│
├── vector_store/              # Vector database integration
│   └── vector_db.py           # Vector database wrapper
│
└── main.py                    # Application entry point
```

## Extending the System

### Creating a Custom Document Processor

To create a custom document processor for a specialized financial document type:

1. Create a new class that inherits from the base processor:

```python
from .base_processor import BaseDocumentProcessor
from pathlib import Path
from typing import List, Dict, Any

class CustomFinancialProcessor(BaseDocumentProcessor):
    """Custom processor for specialized financial documents"""
    
    def load_document(self, file_path: Path):
        """Load the document from the file path"""
        # Custom document loading logic
        pass
        
    def chunk_document(self, document, chunk_size: int = 1024, chunk_overlap: int = 128):
        """Split the document into chunks"""
        # Custom chunking logic
        pass
        
    def extract_metadata(self, document, chunk):
        """Extract metadata from document chunk"""
        # Custom metadata extraction logic
        pass
```

2. Register your processor in the document processor agent:

```python
# In document_processor_agent.py
def _get_processor_for_file(self, file_path: Path):
    """Get the appropriate processor for the file type."""
    file_ext = file_path.suffix.lower()
    
    # Add your custom condition
    if file_ext == '.your_custom_extension':
        return CustomFinancialProcessor()
    
    # Existing supported financial document formats
    supported_formats = [...]
    if file_ext in supported_formats:
        return FinancialDocumentProcessor()
        
    return None
```

### Adding a New LLM Provider

To add support for a new LLM provider:

1. Update the `LLMProvider` enum in `src/llm/llm_provider.py`:

```python
class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NEW_PROVIDER = "new_provider"  # Add your new provider
    HYBRID = "hybrid"
```

2. Add a creation method in the `LLMProviderFactory` class:

```python
def _create_new_provider_instance(self) -> Optional[Any]:
    """Create a new provider LLM instance"""
    if not os.environ.get("NEW_PROVIDER_API_KEY") and not settings.NEW_PROVIDER_API_KEY:
        logger.warning("NEW_PROVIDER_API_KEY not set. New provider will not be available.")
        return None
    
    try:
        # Implement provider-specific integration
        from new_provider_library import NewProviderAPI
        
        return NewProviderAPI(
            model=self._new_provider_model,
            temperature=self._temperature
        )
    except Exception as e:
        logger.error(f"Failed to initialize New Provider: {e}")
        return None
```

3. Update the configuration in `src/config/config.py`:

```python
# LLM settings
LLM_PROVIDER: str = "openai"  # Add 'new_provider' as an option
NEW_PROVIDER_MODEL: str = "model_name"  # Add model configuration
NEW_PROVIDER_API_KEY: Optional[str] = None  # Add API key
```

### Customizing the Chunking Strategy

To implement a custom chunking strategy for financial documents:

1. Create a new chunking method in your document processor:

```python
def custom_financial_chunking(self, document, chunk_size: int = 1024, chunk_overlap: int = 128):
    """
    Custom chunking strategy for financial documents that preserves
    important financial structures like tables and sections.
    """
    chunks = []
    
    # Implement custom chunking logic that preserves financial document structure
    # For example, keeping financial tables intact or respecting section boundaries
    
    return chunks
```

2. Update the chunking call in your document processor:

```python
def chunk_document(self, document, chunk_size: int = 1024, chunk_overlap: int = 128):
    """Split document into chunks with custom financial strategy."""
    if document.get("doc_type") == "financial_statement":
        return self.custom_financial_chunking(document, chunk_size, chunk_overlap)
    else:
        # Fall back to default chunking for other document types
        return super().chunk_document(document, chunk_size, chunk_overlap)
```

### Adding Custom Financial Entity Extraction

To enhance financial entity extraction:

1. Implement a custom extraction method in the financial indexer:

```python
def extract_custom_financial_entities(self, text: str) -> Dict[str, List[str]]:
    """
    Extract custom financial entities from text.
    
    Returns:
        Dictionary of entity types to lists of entity values
    """
    entities = {
        "financial_metrics": [],
        "risk_indicators": [],
        "custom_entities": []
    }
    
    # Implement custom extraction logic
    # For example, using regex patterns for specific financial metrics
    
    return entities
```

2. Integrate with the existing extraction process:

```python
def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
    """Extract financial entities from text."""
    # Get base entities from standard extraction
    entities = self._base_extract_financial_entities(text)
    
    # Enhance with custom entities
    custom_entities = self.extract_custom_financial_entities(text)
    
    # Merge dictionaries
    for entity_type, values in custom_entities.items():
        if entity_type in entities:
            entities[entity_type].extend(values)
        else:
            entities[entity_type] = values
            
    return entities
```

## Working with the Multi-Agent System

The multi-agent system uses a task-based approach to manage concurrent processing:

1. **Task Creation**: The orchestrator creates tasks and assigns them to agents
2. **Task Execution**: Agents process tasks in parallel using worker threads or Ray
3. **Task Status**: Tasks have status tracking and result storage
4. **Task Coordination**: The orchestrator manages dependencies between tasks

To implement a custom agent:

```python
from .base_agent import Agent
from typing import Dict, Any, Optional

class CustomAgent(Agent):
    """Custom agent for specialized processing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id=agent_id, name="CustomAgent")
        # Initialize agent-specific resources
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the agent's processing logic."""
        # Implement custom processing
        return {"result": "Custom processing completed"}
```

## Performance Optimization

### Vector Database Optimization

The system uses ChromaDB by default, but can be optimized:

1. **Embedding Models**: Choose embedding models based on financial domain performance:
   ```python
   # In config.py
   EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"  # Alternative: "sentence-transformers/all-MiniLM-L6-v2"
   ```

2. **Persistent Storage**: Configure for persistent storage:
   ```python
   # In vector_db.py
   self.db = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
   ```

3. **Similarity Search**: Customize the similarity function:
   ```python
   # In vector_db.py
   results = self.collection.query(
       query_embeddings=[query_embedding],
       n_results=n_results,
       where=filter_dict if filter_dict else None,
       include=["documents", "metadatas", "distances"]
   )
   ```

### Multi-Processing Optimization

For large-scale document processing:

1. **Worker Configuration**: Adjust the number of workers in config.py:
   ```python
   MAX_WORKERS: int = os.cpu_count() or 4  # Adjust based on available resources
   ```

2. **Ray Integration**: Enable Ray for distributed processing:
   ```python
   # In orchestrator.py
   if use_ray:
       import ray
       if not ray.is_initialized():
           ray.init()
       # Use Ray for task processing
   ```

## Testing and Evaluation

### Unit Testing

Add unit tests for system components:

```python
# In tests/test_financial_processor.py
import unittest
from pathlib import Path
from src.document_processing.financial_document_processor import FinancialDocumentProcessor

class TestFinancialDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FinancialDocumentProcessor()
        
    def test_extract_financial_entities(self):
        text = "The company reported an EBITDA of $25M and a debt-to-equity ratio of 1.5."
        entities = self.processor.extract_financial_entities(text)
        
        self.assertIn("monetary_values", entities)
        self.assertIn("$25M", entities["monetary_values"])
        self.assertIn("ratios", entities)
        self.assertIn("1.5", entities["ratios"])
```

### Integration Testing

Test the full document processing pipeline:

```python
# In tests/test_integration.py
import unittest
import tempfile
from pathlib import Path
from src.agents.orchestrator import AgentOrchestrator

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.orchestrator = AgentOrchestrator(collection_name="test_collection")
        
    def test_document_processing_pipeline(self):
        # Create a test document
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            # Write test content
            # ...
            
        try:
            # Process the document
            result = self.orchestrator.process_document(Path(f.name))
            
            # Verify processing results
            self.assertIsNotNone(result)
            # More assertions...
            
        finally:
            # Clean up
            Path(f.name).unlink(missing_ok=True)
```

## Deployment Considerations

### Docker Deployment

The system includes a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download spaCy model
RUN python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl

EXPOSE 8000

CMD ["python", "src/main.py"]
```

### Scaling Strategies

For production deployment:

1. **Horizontal Scaling**: Deploy multiple instances behind a load balancer
2. **Database Scaling**: Use a distributed vector database like Pinecone or Weaviate
3. **Async Processing**: Implement asynchronous processing with message queues
4. **Caching**: Implement result caching for frequent queries

### Monitoring and Logging

The system uses `loguru` for logging, but can be extended with:

1. **Centralized Logging**: Send logs to a centralized logging service
2. **Metrics Collection**: Implement metrics collection for performance monitoring
3. **Alerting**: Set up alerts for critical errors or performance issues

## Additional Resources

- [API Reference](./api-reference.md): Complete API documentation
- [LLM Integration](./llm-integration.md): Details on LLM configuration
- [Financial Document Processing](./financial-document-processing.md): Specialized document processing
- [Examples](./examples/README.md): Code examples and usage patterns
