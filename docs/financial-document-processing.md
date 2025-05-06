# Financial Document Processing

This document explains how the Financial Due Diligence RAG system processes financial documents, including supported formats, chunking strategies, entity extraction, and metadata handling.

## Supported Document Formats

The system supports a wide range of financial document formats:

| Format | Extensions | Description |
|--------|------------|-------------|
| PDF | .pdf | Standard financial reports, contracts, filings |
| Microsoft Word | .docx, .doc | Text-based financial documents, memos, contracts |
| Microsoft Excel | .xlsx, .xls | Financial spreadsheets, financial models, data tables |
| Microsoft PowerPoint | .pptx, .ppt | Presentations, investor decks, financial summaries |
| CSV/TSV | .csv, .tsv | Tabular financial data, exported reports |
| Text | .txt | Plain text financial information |
| Markdown | .md | Documentation with financial information |
| HTML | .html | Web-based financial reports, disclosures |
| XML | .xml | Structured financial data, XBRL filings |
| JSON | .json | Financial data in JSON format |

## Financial Document Processor

The `FinancialDocumentProcessor` class handles the specialized processing of financial documents:

```python
class FinancialDocumentProcessor(BaseDocumentProcessor):
    """Processor for financial documents including statements, reports, and contracts."""
    
    def load_document(self, file_path: Path) -> Document:
        """Load document from file path with format-specific handling."""
        # Format-specific loading logic
        
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split document into chunks with financial context preservation."""
        # Financial-aware chunking strategy
        
    def extract_metadata(self, document: Document, chunk: str) -> Dict[str, Any]:
        """Extract financial metadata from document chunks."""
        # Financial metadata extraction
```

### Document Loading Process

The document loading process varies by file type:

1. **PDF Processing**:
   - Text extraction with PyPDF
   - OCR processing for scanned documents using Tesseract
   - Table extraction with Tabula

2. **Excel Processing**:
   - Sheet-based processing
   - Financial table recognition
   - Formula preservation
   - Cell type recognition (date, currency, percentage)

3. **Word Processing**:
   - Section-based extraction
   - Table parsing
   - Template recognition for common financial documents

4. **Other Formats**:
   - Format-specific handlers for HTML, XML, JSON, etc.
   - Structured data extraction

## Intelligent Chunking Strategies

The system employs specialized chunking strategies for financial documents:

### Semantic Boundary Chunking

Rather than splitting documents at arbitrary token limits, the system respects semantic boundaries in financial documents:

- Sections and subsections
- Financial statements (balance sheet, income statement, cash flow)
- Footnotes and disclosures
- Tables and financial data structures

```python
def semantic_boundary_chunking(self, text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> List[str]:
    """
    Chunk text while preserving semantic boundaries in financial documents.
    """
    # Identify section headers and boundaries
    section_patterns = [
        r"^\s*(?:ITEM|PART)\s+\d+[A-Z]?\.?\s+[A-Z\s]+",  # SEC filing sections
        r"^\s*(?:Note|Notes)\s+\d+[.:]\s+.+",  # Financial statement notes
        r"^\s*(?:Table|Figure)\s+\d+[.:]\s+.+",  # Tables and figures
        r"^\s*(?:Statement of|Balance Sheet|Income Statement|Cash Flow)",  # Financial statements
    ]
    
    # Find potential split points that respect semantic boundaries
    # ...
    
    return chunks
```

### Table-Aware Chunking

Financial tables are processed as cohesive units:

```python
def table_aware_chunking(self, document: Document) -> List[DocumentChunk]:
    """
    Extract and process tables as distinct chunks.
    """
    chunks = []
    
    # Extract tables from document
    tables = self._extract_tables(document)
    
    for table in tables:
        # Process each table as a separate chunk
        # Preserve table structure and relationships
        chunks.append(DocumentChunk(
            content=table.content,
            metadata={
                "is_table": True,
                "table_title": table.title,
                "table_columns": table.columns,
                # Other table metadata
            }
        ))
    
    # Process non-table content
    # ...
    
    return chunks
```

### Financial Statement Chunking

Special handling for structured financial statements:

```python
def financial_statement_chunking(self, document: Document) -> List[DocumentChunk]:
    """
    Process financial statements with specialized chunking.
    """
    # Identify financial statement sections
    statement_sections = [
        "Balance Sheet",
        "Income Statement",
        "Statement of Cash Flows",
        "Statement of Changes in Equity",
        # Other financial statement types
    ]
    
    # Extract and process each statement section
    # Preserve line items and financial hierarchies
    # ...
    
    return chunks
```

## Financial Entity Extraction

The system extracts various financial entities to enhance search and analysis:

### Entity Types

| Entity Type | Examples | Description |
|-------------|----------|-------------|
| Monetary Values | $10M, €50 million | Currency amounts with units |
| Percentages | 15%, 2.5 percent | Percentage values |
| Dates | Q3 2023, FY 2022 | Temporal financial references |
| Ratios | 1.5x, debt-to-equity, P/E | Financial ratios |
| Companies | Microsoft Corp., MSFT | Company names and tickers |
| Financial Terms | EBITDA, amortization | Specialized financial terminology |
| Risk Indicators | "material weakness", "going concern" | Terms indicating financial risks |

### Extraction Process

The extraction uses a combination of approaches:

1. **Named Entity Recognition (NER)**:
   - Custom spaCy models for financial entities
   - Rule-based pattern matching

2. **Regular Expression Patterns**:
   - Financial-specific regex patterns
   - Currency and number formats

3. **Financial Taxonomy Matching**:
   - Matching against financial term dictionaries
   - XBRL taxonomy integration

```python
def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
    """
    Extract financial entities from text.
    
    Returns:
        Dictionary of entity types to lists of entity values
    """
    entities = {
        "monetary_values": [],
        "percentages": [],
        "dates": [],
        "ratios": [],
        "companies": [],
        "financial_terms": [],
        "risk_indicators": []
    }
    
    # Apply NER model
    doc = self.nlp_model(text)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities["monetary_values"].append(ent.text)
        # Other entity mappings...
    
    # Apply regex patterns
    monetary_pattern = r'(?:[$€£¥])\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|m|b|k))?'
    for match in re.finditer(monetary_pattern, text):
        entities["monetary_values"].append(match.group())
    
    # More extraction logic...
    
    return entities
```

## Financial Metadata Enhancement

The system enriches document chunks with financial metadata:

### Metadata Fields

| Metadata Field | Description | Example |
|----------------|-------------|---------|
| doc_category | Type of financial document | "financial_statement", "contract", "regulatory_filing" |
| financial_period | Time period referenced | "Q3 2023", "FY 2022" |
| reporting_entity | Entity issuing the document | "Acme Corporation" |
| financial_entities | Extracted financial entities | {monetary_values: ["$10M"], ratios: ["2.3x"]} |
| key_metrics | Important financial metrics | {"revenue": "$10M", "profit_margin": "15%"} |
| statement_type | Type of financial statement | "balance_sheet", "income_statement" |
| risk_level | Assessed financial risk | "high", "medium", "low" |
| sentiment | Financial sentiment analysis | "negative", "positive", "neutral" |

### Metadata Enhancement Logic

```python
def enhance_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
    """
    Enhance chunk metadata with financial insights.
    """
    text = chunk.content
    metadata = chunk.metadata.copy()
    
    # Categorize document type
    metadata["doc_category"] = self._categorize_document(text)
    
    # Extract financial entities
    metadata["financial_entities"] = self.extract_financial_entities(text)
    
    # Identify financial period
    metadata["financial_period"] = self._extract_financial_period(text)
    
    # Extract key metrics
    metadata["key_metrics"] = self._extract_key_metrics(text)
    
    # Analyze financial sentiment
    metadata["sentiment"] = self._analyze_financial_sentiment(text)
    
    # Assess risk indicators
    risk_count = len(metadata["financial_entities"].get("risk_indicators", []))
    if risk_count > 5:
        metadata["risk_level"] = "high"
    elif risk_count > 2:
        metadata["risk_level"] = "medium"
    else:
        metadata["risk_level"] = "low"
    
    return metadata
```

## Financial Indexer

The `FinancialIndexer` enhances search capabilities specifically for financial documents:

### Topic Modeling

The system performs topic modeling on financial documents to identify key themes:

```python
def perform_topic_modeling(self, documents: List[str], num_topics: int = 10) -> Dict[int, List[str]]:
    """
    Perform topic modeling on a collection of financial documents.
    
    Returns:
        Dictionary mapping topic IDs to lists of representative terms
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95, min_df=2,
        stop_words='english'
    )
    dt_matrix = vectorizer.fit_transform(documents)
    
    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda.fit(dt_matrix)
    
    # Extract top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[topic_idx] = top_words
    
    return topics
```

### Financial Term Expansion

The system expands financial search terms to include related concepts:

```python
def expand_financial_terms(self, query: str) -> List[str]:
    """
    Expand financial terms in query with related concepts.
    
    For example, "profit margin" might expand to include
    "gross margin", "operating margin", "net margin", etc.
    """
    expansions = []
    
    # Financial term dictionaries
    financial_term_expansions = {
        "profit margin": ["gross margin", "operating margin", "net margin", "profit ratio"],
        "debt": ["loans", "bonds", "liabilities", "borrowings", "notes payable"],
        "revenue": ["sales", "income", "turnover", "top line"],
        # More financial term mappings...
    }
    
    # Expand terms found in the query
    for term, expansions_list in financial_term_expansions.items():
        if term in query.lower():
            expansions.extend(expansions_list)
    
    return expansions
```

### Entity-Based Indexing

The system creates specialized indices for financial entities:

```python
def index_financial_entities(self, chunks: List[DocumentChunk]) -> Dict[str, Dict[str, List[str]]]:
    """
    Create inverted indices for financial entities.
    
    Returns:
        Dictionary mapping entity types to dictionaries of entity values to chunk IDs
    """
    indices = {
        "monetary_values": {},
        "companies": {},
        "ratios": {},
        "dates": {},
        # Other entity types...
    }
    
    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        financial_entities = chunk.metadata.get("financial_entities", {})
        
        for entity_type, entity_values in financial_entities.items():
            if entity_type in indices:
                for value in entity_values:
                    if value not in indices[entity_type]:
                        indices[entity_type][value] = []
                    indices[entity_type][value].append(chunk_id)
    
    return indices
```

## Document Processing Pipeline

The complete financial document processing pipeline involves these steps:

1. **Document Loading**:
   - Input: Raw financial document file
   - Process: Format-specific loading and text extraction
   - Output: Document object with extracted text and basic metadata

2. **Financial Entity Extraction**:
   - Input: Document object
   - Process: NER, regex patterns, and taxonomy matching
   - Output: Extracted financial entities

3. **Intelligent Chunking**:
   - Input: Document with extracted entities
   - Process: Semantic, table-aware, and financial statement chunking
   - Output: Document chunks preserving financial context

4. **Metadata Enhancement**:
   - Input: Document chunks
   - Process: Financial metadata enhancement
   - Output: Chunks with rich financial metadata

5. **Embedding Generation**:
   - Input: Enhanced document chunks
   - Process: Vector embedding generation
   - Output: Vector representations of chunks

6. **Financial Indexing**:
   - Input: Document chunks with embeddings
   - Process: Topic modeling and entity-based indexing
   - Output: Financial topic and entity indices

7. **Vector Storage**:
   - Input: Chunks with embeddings and enhanced metadata
   - Process: Storage in vector database
   - Output: Indexed and queryable document collection

## Customization Options

### Custom Entity Recognition

You can extend the financial entity recognition with custom patterns:

```python
# Add custom financial entity patterns
additional_financial_patterns = [
    {"label": "FINANCIAL_METRIC", "pattern": [{"LOWER": "cagr"}, {"IS_DIGIT": True}, {"LOWER": "%"}]},
    {"LOWER": "arr"},
    {"TEXT": {"REGEX": "(?i)customer acquisition cost"}},
    # More custom patterns...
]

# Add to the NLP pipeline
nlp = spacy.load("en_core_web_lg")
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(additional_financial_patterns)
```

### Custom Document Categories

You can define custom financial document categories:

```python
# Define custom document categories
document_categories = {
    "term_sheet": ["term sheet", "termsheet", "terms and conditions"],
    "pitch_deck": ["pitch deck", "investor presentation", "funding presentation"],
    "valuation_report": ["valuation report", "valuation analysis", "business valuation"],
    # More categories...
}

# Implement the categorization logic
def _categorize_document(self, text: str) -> str:
    """Categorize financial document based on content."""
    text_lower = text.lower()
    
    for category, keywords in document_categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return "other"
```

## Best Practices

### Document Preparation

For optimal results with financial documents:

1. **Use OCR Pre-Processing** for scanned documents:
   - Ensure proper resolution (minimum 300 DPI)
   - Use image preprocessing to enhance image quality
   - Consider manual verification for critical financial data

2. **Standardize Financial Data Format**:
   - Use consistent number formatting
   - Include units with financial figures
   - Structure tables consistently

3. **Preserve Document Structure**:
   - Maintain clear section headings
   - Use proper formatting for financial statements
   - Include document metadata

### Processing Performance

1. **Batch Processing** for large document collections:
   - Group similar document types
   - Process in parallel
   - Monitor memory usage

2. **Optimize Entity Extraction**:
   - Focus on relevant entity types
   - Prioritize precision over recall for critical financial metrics
   - Cache extraction results for similar documents

3. **Chunking Strategy Selection**:
   - Use table-aware chunking for data-heavy documents
   - Use semantic chunking for narrative financial reports
   - Use financial statement chunking for formal financial statements

## Troubleshooting

### Common Issues

1. **Poor OCR Quality**:
   - Error: Extracted numbers are incorrect or garbled
   - Solution: Improve image quality, consider pre-processing, or use specialized financial OCR

2. **Table Structure Loss**:
   - Error: Financial tables are not preserved properly
   - Solution: Use table-aware chunking, consider custom table extraction

3. **Entity Extraction Misses**:
   - Error: Important financial entities are not extracted
   - Solution: Add custom patterns, adjust extraction thresholds

4. **Incorrect Document Categorization**:
   - Error: Documents assigned to wrong categories
   - Solution: Review categorization rules, add more specific keywords

5. **Large Document Handling**:
   - Error: Out of memory errors with very large financial documents
   - Solution: Adjust chunking parameters, implement streaming processing
