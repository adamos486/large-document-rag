# Implementation Guide: Financial Due Diligence RAG System

This guide provides detailed implementation insights for developers working with the Financial Due Diligence RAG system, focusing on how the system was transformed from handling GIS/CAD files to processing financial documents for M&A due diligence.

## System Transformation Overview

The system underwent a significant transformation from processing GIS/CAD files to handling financial documents. This section outlines the key changes made during this transformation.

### Changes in Document Processing

| Original (GIS/CAD) | New (Financial Due Diligence) | Implementation Impact |
|-------------------|--------------------------|------------------------|
| GIS file processing (shapefiles, GeoJSON) | Financial document processing (PDF, XLSX, DOCX) | New file format handlers |
| Spatial chunking | Semantic & financial statement chunking | Changed chunking strategies |
| Coordinate system metadata | Financial entity metadata | New metadata extraction |
| Geographic feature extraction | Financial entity extraction | New entity recognition |
| Spatial querying | Financial term expansion | New query enhancement |

### Key Components Replaced

1. **GIS/CAD-specific processors → Financial document processors**
2. **Spatial indexing → Financial entity indexing**
3. **Feature-based chunking → Financial context-aware chunking**
4. **Geographic entity extraction → Financial entity extraction**
5. **Coordinate system normalization → Financial term normalization**

## Implementation Details

### 1. Document Processor Transformation

The original GIS/CAD processors were replaced with financial document processors:

#### Original GIS Processor:
```python
class GISProcessor(BaseDocumentProcessor):
    def load_document(self, file_path: Path):
        # GIS-specific loading with geopandas
        import geopandas as gpd
        gdf = gpd.read_file(file_path)
        return {"gdf": gdf, "type": "gis", "crs": gdf.crs}
    
    def chunk_document(self, document, chunk_size=1000):
        # Spatial chunking based on features or grid
        chunks = []
        gdf = document["gdf"]
        
        if len(gdf) <= chunk_size:
            return [{"content": gdf.to_json(), "metadata": {"feature_count": len(gdf)}}]
        
        # Grid-based or feature-based chunking
        # ...
        
        return chunks
```

#### New Financial Document Processor:
```python
class FinancialDocumentProcessor(BaseDocumentProcessor):
    def load_document(self, file_path: Path):
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            # PDF processing with PyPDF
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
                
            # Extract tables with tabula-py
            import tabula
            tables = tabula.read_pdf(file_path, pages='all')
            
            return {"text": text, "tables": tables, "type": "financial_report"}
            
        elif file_ext in ['.xlsx', '.xls']:
            # Excel processing
            import pandas as pd
            excel_data = pd.read_excel(file_path, sheet_name=None)
            return {"sheets": excel_data, "type": "financial_spreadsheet"}
            
        # Additional formats...
    
    def chunk_document(self, document, chunk_size=1024, chunk_overlap=128):
        # Financial-aware chunking
        if document["type"] == "financial_report":
            # Semantic chunking for reports
            return self._semantic_financial_chunking(document["text"], chunk_size, chunk_overlap)
        elif document["type"] == "financial_spreadsheet":
            # Sheet and table-based chunking
            return self._tabular_financial_chunking(document["sheets"])
```

### 2. Entity Extraction Transformation

The entity extraction system was completely transformed:

#### Original Geographic Entity Extraction:
```python
def extract_geographic_entities(self, gdf):
    """Extract geographic entities from GIS data."""
    entities = {
        "countries": [],
        "cities": [],
        "coordinates": [],
        "features": []
    }
    
    # Extract country/city names from attributes
    for idx, feature in gdf.iterrows():
        if "COUNTRY" in feature:
            entities["countries"].append(feature["COUNTRY"])
        # More geographic entity extraction...
    
    return entities
```

#### New Financial Entity Extraction:
```python
def extract_financial_entities(self, text):
    """Extract financial entities from text."""
    entities = {
        "monetary_values": [],
        "percentages": [],
        "dates": [],
        "companies": [],
        "financial_terms": []
    }
    
    # Extract monetary values with regex
    import re
    money_pattern = r'[$€£¥]\s?\d+(?:\.\d+)?(?:\s?(?:million|billion|m|b))?'
    entities["monetary_values"] = re.findall(money_pattern, text)
    
    # Extract percentages
    percent_pattern = r'\d+(?:\.\d+)?\s?%'
    entities["percentages"] = re.findall(percent_pattern, text)
    
    # Use spaCy for NER
    doc = self.nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["companies"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)
    
    # Financial term matching using custom dictionary
    for term in self.financial_terms_dict:
        if term.lower() in text.lower():
            entities["financial_terms"].append(term)
    
    return entities
```

### 3. Indexing System Transformation

The indexing system was transformed from spatial to financial:

#### Original Spatial Indexer:
```python
class SpatialIndexer:
    def __init__(self, index_path):
        self.index_path = index_path
        self.spatial_index = rtree.index.Index(index_path)
        
    def index_features(self, chunks):
        """Index spatial features for efficient retrieval."""
        for i, chunk in enumerate(chunks):
            # Extract bounding box
            minx, miny, maxx, maxy = chunk["metadata"]["bbox"]
            self.spatial_index.insert(i, (minx, miny, maxx, maxy), obj=chunk["id"])
    
    def query_by_location(self, bbox):
        """Query chunks by bounding box."""
        return list(self.spatial_index.intersection(bbox))
```

#### New Financial Indexer:
```python
class FinancialIndexer:
    def __init__(self, index_path):
        self.index_path = index_path
        self.entity_index = {}
        self.topic_model = None
        
    def index_chunks(self, chunks):
        """Index chunks by financial entities and topics."""
        # Create entity-based index
        for chunk in chunks:
            chunk_id = chunk.metadata["chunk_id"]
            financial_entities = chunk.metadata.get("financial_entities", {})
            
            for entity_type, values in financial_entities.items():
                if entity_type not in self.entity_index:
                    self.entity_index[entity_type] = {}
                    
                for value in values:
                    if value not in self.entity_index[entity_type]:
                        self.entity_index[entity_type][value] = []
                    self.entity_index[entity_type][value].append(chunk_id)
        
        # Create topic model
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        
        texts = [chunk.content for chunk in chunks]
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(texts)
        
        self.topic_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.topic_model.fit(dtm)
        self.vectorizer = vectorizer
    
    def query_by_entity(self, entity_type, value):
        """Query chunks by financial entity."""
        return self.entity_index.get(entity_type, {}).get(value, [])
    
    def enhance_query(self, query_text):
        """Enhance query with financial term expansion."""
        # Expand financial terms
        expanded_terms = []
        
        # Example financial term expansions
        expansions = {
            "profit": ["gross profit", "net profit", "operating profit", "profit margin"],
            "revenue": ["sales", "income", "turnover"],
            # More expansions...
        }
        
        for term, expansions_list in expansions.items():
            if term in query_text.lower():
                expanded_terms.extend(expansions_list)
        
        return expanded_terms
```

### 4. Query Agent Transformation

The query agent was transformed to handle financial queries:

#### Original GIS Query Logic:
```python
def query_gis_database(self, query, coordinates=None, region=None):
    """Query the GIS vector database with spatial context."""
    # Get embedding for text query
    query_embedding = self.embedding_model.get_embedding(query)
    
    # Apply spatial filter if provided
    filters = {}
    if coordinates:
        lat, lon = coordinates
        # Convert to bounding box
        bbox = (lon-0.1, lat-0.1, lon+0.1, lat+0.1)
        chunk_ids = self.spatial_indexer.query_by_location(bbox)
        filters["chunk_id"] = {"$in": chunk_ids}
    
    if region:
        filters["metadata.region"] = region
    
    # Query vector database
    results = self.vector_store.query(
        query_embedding=query_embedding,
        filters=filters
    )
    
    return results
```

#### New Financial Query Logic:
```python
def query_financial_database(self, query, filters=None):
    """Query the financial vector database with financial context."""
    # Get embedding for text query
    query_embedding = self.embedding_model.get_embedding(query)
    
    # Enhance query with financial term expansion
    if self.financial_indexer:
        expanded_terms = self.financial_indexer.enhance_query(query)
        if expanded_terms:
            # Get embeddings for expanded terms
            expanded_embeddings = [self.embedding_model.get_embedding(term) for term in expanded_terms]
            # Combine embeddings
            for exp_emb in expanded_embeddings:
                query_embedding = [0.8 * qe + 0.2 * ee for qe, ee in zip(query_embedding, exp_emb)]
    
    # Extract financial entities from query
    financial_entities = self.financial_document_processor.extract_financial_entities(query)
    
    # Apply financial entity filters
    query_filters = filters or {}
    if "monetary_values" in financial_entities and financial_entities["monetary_values"]:
        query_filters["metadata.financial_entities.monetary_values"] = {
            "$in": financial_entities["monetary_values"]
        }
    
    if "companies" in financial_entities and financial_entities["companies"]:
        query_filters["metadata.financial_entities.companies"] = {
            "$in": financial_entities["companies"]
        }
    
    # Query vector database
    results = self.vector_store.query(
        query_embedding=query_embedding,
        filters=query_filters
    )
    
    # Enhance results with financial entity matches
    if self.financial_indexer:
        entity_matches = []
        for entity_type, values in financial_entities.items():
            for value in values:
                chunk_ids = self.financial_indexer.query_by_entity(entity_type, value)
                if chunk_ids:
                    entity_chunks = self.vector_store.get_chunks_by_ids(chunk_ids)
                    entity_matches.extend(entity_chunks)
        
        # Merge vector results with entity-based results
        # ...
    
    return results
```

## Dependency Transformation

The system's dependencies were transformed to support financial document processing:

### Original GIS/CAD Dependencies:
```
# GIS/CAD dependencies
geopandas>=0.12.0
shapely>=2.0.0
fiona>=1.8.22
rtree>=1.0.0
pyproj>=3.4.0
ezdxf>=1.0.0
```

### New Financial Dependencies:
```
# Document processing
unstructured>=0.11.2
pypdf>=3.17.1  # For PDF files
python-docx>=1.0.1  # For Word documents
pdf2image>=1.16.3  # For converting PDFs to images
pytesseract>=0.3.10  # For OCR
openpyxl>=3.1.2  # For Excel files
lxml>=4.9.3  # For XML parsing
beautifulsoup4>=4.12.2  # For HTML parsing
tabula-py>=2.8.2  # For extracting tables from PDFs
python-pptx>=0.6.22  # For PowerPoint files
markdown>=3.5.1  # For Markdown files
json5>=0.9.14  # For JSON files

# Financial NLP/NER
spacy>=3.7.2
transformers-interpret>=0.10.0  # Alternative to finbert for financial NLP
spacytextblob>=4.0.0  # For sentiment analysis
```

## Maintaining Compatibility

When updating the system, it's important to maintain compatibility across components:

### 1. Vector Database Compatibility

The system uses ChromaDB for vector storage, which requires careful version handling:

```python
# In vector_db.py
import chromadb

class VectorStore:
    def __init__(self, collection_name="default"):
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        # Version 1.0.0 has different API than earlier versions
        self.db = chromadb.PersistentClient(path=str(settings.VECTOR_DB_PATH))
        
        # Create or get collection
        try:
            self.collection = self.db.get_collection(collection_name)
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.db.create_collection(
                name=collection_name,
                metadata={"description": f"Financial document collection: {collection_name}"}
            )
    
    def add_chunks(self, chunks):
        """Add document chunks to vector store."""
        # Format for ChromaDB 1.0.0 API
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
```

### 2. ML/NLP Package Compatibility

The system uses spaCy and transformers, which require careful version management:

```python
# In financial_document_processor.py
import spacy
from transformers import pipeline

class FinancialDocumentProcessor:
    def __init__(self):
        # Load spaCy model - ensure version compatibility
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            # Fallback for compatibility
            spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize sentiment analysis - with version check
        try:
            # Newer transformers API
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception as e:
            try:
                # Fallback to older API for compatibility
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            except Exception as e2:
                logger.warning(f"Could not initialize sentiment analysis: {e2}")
                self.sentiment_analyzer = None
```

### 3. Embedding Model Compatibility

Ensure consistent embedding dimensions across the system:

```python
# In embeddings.py
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        # Initialize embedding model with version check
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model {self.model_name} with dimension {self.embedding_dim}")
        except Exception as e:
            # Fallback to older compatible model
            fallback_model = "all-MiniLM-L6-v2"  # Very stable model
            logger.warning(f"Error loading {self.model_name}: {e}. Falling back to {fallback_model}")
            self.model = SentenceTransformer(fallback_model)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def get_embedding(self, text):
        # Ensure consistent output format regardless of version
        embedding = self.model.encode(text)
        
        # Return as Python list for consistent serialization
        return embedding.tolist()
```

## Testing the Transformation

Verify the financial document processing transformation works correctly:

### 1. Document Processing Test

```python
def test_financial_document_processor():
    """Test the financial document processor with various document types."""
    processor = FinancialDocumentProcessor()
    
    # Test PDF processing
    pdf_path = Path("tests/data/sample_financial_report.pdf")
    pdf_doc = processor.load_document(pdf_path)
    assert "text" in pdf_doc
    assert len(pdf_doc["text"]) > 0
    
    # Test Excel processing
    excel_path = Path("tests/data/sample_financial_data.xlsx")
    excel_doc = processor.load_document(excel_path)
    assert "sheets" in excel_doc
    assert len(excel_doc["sheets"]) > 0
    
    # Test chunking
    pdf_chunks = processor.chunk_document(pdf_doc)
    assert len(pdf_chunks) > 0
    assert "content" in pdf_chunks[0]
    assert "metadata" in pdf_chunks[0]
    
    # Test financial entity extraction
    entities = processor.extract_financial_entities(pdf_chunks[0].content)
    assert "monetary_values" in entities
    assert "percentages" in entities
    assert "companies" in entities
```

### 2. Vector Store Test

```python
def test_vector_store_compatibility():
    """Test vector store compatibility with financial document chunks."""
    processor = FinancialDocumentProcessor()
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(collection_name="test_financial")
    
    # Process a test document
    doc_path = Path("tests/data/sample_financial_report.pdf")
    document = processor.load_document(doc_path)
    chunks = processor.chunk_document(document)
    
    # Add embeddings
    for chunk in chunks:
        chunk.embedding = embedding_model.get_embedding(chunk.content)
    
    # Add to vector store
    vector_store.add_chunks(chunks)
    
    # Test query
    query = "What is the company's profit margin?"
    query_embedding = embedding_model.get_embedding(query)
    results = vector_store.query(query_embedding=query_embedding, n_results=3)
    
    assert len(results) > 0
```

## Migration Strategy

For projects transitioning from GIS/CAD to financial document processing:

### 1. Gradual Component Replacement

Replace components one at a time:

1. **First**: Update document processors while maintaining original indexing
2. **Second**: Add financial entity extraction while keeping original querying
3. **Third**: Update the indexing system
4. **Fourth**: Enhance the query logic
5. **Fifth**: Update the vector database integration

### 2. Parallel System Operation

During migration, run both systems in parallel:

```python
def process_document(self, file_path: Path):
    """Process a document with the appropriate processor."""
    file_ext = file_path.suffix.lower()
    
    # Route to appropriate processor
    if file_ext in ['.shp', '.geojson', '.dxf', '.dwg']:
        # Legacy GIS/CAD processing
        if not self.legacy_mode:
            logger.warning(f"GIS/CAD file detected, but system is now optimized for financial documents. Processing with legacy mode.")
        return self._process_legacy_document(file_path)
    elif file_ext in ['.pdf', '.docx', '.xlsx', '.txt']:
        # Financial document processing
        return self._process_financial_document(file_path)
    else:
        logger.error(f"Unsupported file type: {file_ext}")
        return None
```

### 3. Data Migration

Migrate existing vector data:

```python
def migrate_vector_data(legacy_collection, new_collection):
    """Migrate data from legacy collection to new collection with updated schema."""
    # Get all chunks from legacy collection
    legacy_chunks = legacy_collection.get_all()
    
    # Transform to new schema
    new_chunks = []
    for chunk in legacy_chunks:
        # Convert GIS metadata to financial placeholder metadata
        new_metadata = {
            "chunk_id": chunk.metadata.get("chunk_id", f"chunk_{uuid.uuid4()}"),
            "doc_id": chunk.metadata.get("doc_id", f"doc_{uuid.uuid4()}"),
            "source": chunk.metadata.get("source", "migrated"),
            "doc_category": "legacy_gis_data",
            "financial_entities": {
                "companies": [],
                "monetary_values": [],
                "percentages": []
            }
        }
        
        # Create new chunk
        new_chunk = DocumentChunk(
            content=chunk.content,
            metadata=new_metadata,
            embedding=chunk.embedding
        )
        new_chunks.append(new_chunk)
    
    # Add to new collection
    new_collection.add_chunks(new_chunks)
```

## Deployment Considerations

### 1. Environment Configuration

Update the .env file for financial processing:

```
# Vector DB Settings
VECTOR_DB_PATH=./data/financial_vector_store

# Financial Processing Settings
FINANCIAL_TERMS_DICT=./data/financial_terms.json
ENABLE_OCR=true
OCR_LANGUAGE=eng
TABLE_EXTRACTION=true

# LLM Settings for Financial Analysis
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LLM_PROVIDER=hybrid
```

### 2. Resource Requirements

The resource requirements change significantly when moving from GIS/CAD to financial processing:

| Resource | GIS/CAD Processing | Financial Processing | Reason for Change |
|----------|-------------------|----------------------|-------------------|
| CPU | Medium | Medium-High | OCR and table extraction |
| RAM | High (for large GIS) | Medium | Less spatial data in memory |
| Storage | Very High | Medium | Financial docs typically smaller |
| GPU | Optional | Recommended | For ML-based extraction |

### 3. Dependencies Installation

Install the new financial processing dependencies:

```bash
# Remove GIS/CAD dependencies if no longer needed
pip uninstall geopandas shapely fiona rtree pyproj ezdxf

# Install financial processing dependencies
pip install -r requirements.txt

# Install spaCy model separately
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl
```

## Best Practices for Financial RAG Systems

Based on our transformation experience, here are key best practices:

1. **Document Content Preservation**: Preserve the structure of financial documents, especially tables and statements
2. **Specialized Financial Chunking**: Use domain-specific chunking that respects financial document structure
3. **Rich Metadata**: Extract and store comprehensive financial metadata for enhanced retrieval
4. **Financial Entity Recognition**: Implement specialized extraction for financial entities
5. **Model Selection**: Choose embedding models that perform well on financial text
6. **Query Understanding**: Implement financial domain-specific query understanding
7. **Hybrid Retrieval**: Combine vector similarity with rule-based retrieval for financial entities
8. **LLM Provider Selection**: Use specialized LLMs for financial analysis tasks

## Conclusion

Transforming a RAG system from GIS/CAD to financial due diligence requires careful consideration of document structure, entity extraction, indexing strategies, and query processing. This guide provides a comprehensive overview of the implementation details and best practices for making this transformation while maintaining system compatibility and performance.

For hands-on implementation support, refer to the code examples and tests provided in this guide, and leverage the specialized financial document processing components included in the system.
