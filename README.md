# Financial Due Diligence RAG System for M&A

A Python-based multi-agent Retrieval Augmented Generation (RAG) system designed to process and query large volumes of financial documents for M&A due diligence purposes.

## Features

- Multi-agent architecture for parallel processing of large financial documents
- Specialized chunking strategies optimized for financial documents (PDFs, Excel, Word, etc.)
- Intelligent financial entity extraction and indexing
- Advanced semantic search with financial term expansion
- Topic modeling for document categorization
- Integration with LLMs for comprehensive analysis and summarization
- Vector database integration for efficient retrieval
- Distributed processing capabilities for handling large document collections
- REST API for interacting with the system

## Use Case: M&A Due Diligence

This system is specifically designed to assist financial analysts and investment bankers in the due diligence process for mergers and acquisitions. It can process and analyze:

- Financial statements and annual reports
- Legal contracts and agreements
- Regulatory filings
- Market analysis reports
- Valuation documents
- Tax documents
- Due diligence memos and reports

The system extracts key financial information, identifies risks and opportunities, and provides a comprehensive analysis to support M&A decision-making.

## Project Structure

```
financial-due-diligence-rag/
├── config/                 # Configuration files
├── data/                   # Data storage location
│   └── financial_indices/  # Intelligent indices for financial documents
├── docs/                   # Documentation
└── src/                    # Source code
    ├── agents/             # Multi-agent system components
    ├── api/                # API endpoints
    ├── document_processing/ # Document processors for financial documents
    ├── utils/              # Utility functions
    └── vector_store/       # Vector database integration
```

## Document Processing Pipeline

1. **Document Loading**: Supports various financial document formats (PDF, DOCX, XLSX, etc.)
2. **OCR Processing**: Handles scanned documents with OCR capabilities
3. **Financial Entity Extraction**: Identifies companies, monetary values, dates, percentages, etc.
4. **Intelligent Chunking**: Splits documents based on semantic boundaries
5. **Metadata Extraction**: Extracts key financial metrics and document categories
6. **Embedding Generation**: Creates vector representations of document chunks
7. **Intelligent Indexing**: Builds specialized indices for financial terms and entities
8. **Topic Modeling**: Categorizes documents for better organization and retrieval

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys (especially OpenAI for LLM integration)

## Usage

1. Start the system: `python src/main.py`
2. Use the API to upload financial documents and query the system

### API Endpoints

- `POST /api/upload`: Upload and process a financial document
- `POST /api/query`: Query the system with financial questions
- `POST /api/task/status`: Check the status of a document processing task
- `GET /api/collections`: List all document collections
- `GET /api/collections/{collection_name}/stats`: Get statistics for a collection

## Supported Financial Document Types

The system supports various financial document formats including:

- PDF (text and scanned via OCR)
- Microsoft Word (DOCX)
- Microsoft Excel (XLSX, XLS)
- Microsoft PowerPoint (PPTX, PPT)
- CSV and TSV files
- Plain text files
- HTML and XML documents
- Markdown files
- JSON files

## License

MIT
