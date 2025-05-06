# API Reference

This document provides a complete reference to the Financial Due Diligence RAG system API. It includes details on all available endpoints, request and response formats, and examples of how to use the API.

## Overview

The Financial Due Diligence RAG system API is designed for processing and analyzing financial documents for M&A activities. The API supports various financial document types, including financial statements, annual reports, due diligence reports, and contracts.

### Base URL

```
http://localhost:8000
```

### Authentication

API authentication is handled via API keys passed in the request header:

```
X-API-Key: your_api_key_here
```

### Financial Document Processing

The system specializes in processing financial documents with:
- Specialized financial document chunking strategies
- Financial entity extraction
- Financial metadata enhancement
- Financial sentiment analysis
- Multi-LLM vendor support (OpenAI and Anthropic)

## Base URL

All API endpoints are relative to the base URL:

```
http://{host}:{port}
```

By default: `http://0.0.0.0:8000`

## Authentication

Currently, the API does not implement authentication. For production deployments, we recommend implementing an authentication layer using API keys, OAuth, or other appropriate mechanisms.

## API Endpoints

### Upload Document

Uploads and processes a financial document, adding it to the specified collection.

**Endpoint:** `POST /upload`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `file`: (Required) The financial document file to upload
- `collection_name`: (Optional) The name of the collection to store the document in (default: "default")

**Response:**
```json
{
  "task_id": "upload_1712464589_financial_report.pdf",
  "file_name": "financial_report.pdf",
  "collection_name": "my_ma_deal",
  "status": "processing"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/financial_report.pdf" \
  -F "collection_name=my_ma_deal"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/upload"
files = {"file": open("/path/to/financial_report.pdf", "rb")}
data = {"collection_name": "my_ma_deal"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Process Batch

Uploads and processes multiple financial documents in a batch, adding them to the specified collection.

**Endpoint:** `POST /batch`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `files`: (Required) Multiple financial document files to upload
- `collection_name`: (Optional) The name of the collection to store the documents in (default: "default")
- `parallel_workers`: (Optional) Number of parallel workers to use for processing
- `use_ray`: (Optional) Whether to use Ray for distributed processing (default: false)

**Response:**
```json
{
  "task_id": "batch_1712464590",
  "file_count": 3,
  "collection_name": "my_ma_deal",
  "status": "processing"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/financial_report1.pdf" \
  -F "files=@/path/to/financial_report2.xlsx" \
  -F "files=@/path/to/financial_report3.docx" \
  -F "collection_name=my_ma_deal" \
  -F "parallel_workers=4"
```

### Query

Queries the system with a financial question, retrieving relevant documents and generating an analysis based on those documents.

**Endpoint:** `POST /query`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "query": "What are the key financial risks identified in the target company's balance sheet?",
  "collection_name": "my_ma_deal",
  "filters": {
    "doc_category": "financial_statement"
  },
  "n_results": 5
}
```

**Parameters:**
- `query`: (Required) The financial query text
- `collection_name`: (Optional) The name of the collection to query (default: "default")
- `filters`: (Optional) Metadata filters to apply to the query
- `n_results`: (Optional) Number of results to return (default: 5)

**Response:**
```json
{
  "query": "What are the key financial risks identified in the target company's balance sheet?",
  "llm_response": "Based on the financial documents, the key risks in the target company's balance sheet include:\n\n1. High debt-to-equity ratio (2.3) indicating significant leverage\n2. Declining current ratio from 1.8 to 1.2 over the past year\n3. $15M in contingent liabilities related to ongoing litigation\n4. Underfunded pension obligations of approximately $28M\n5. Significant concentration of accounts receivable with 40% from a single customer",
  "documents": [
    {
      "content": "...[document content]...",
      "metadata": {
        "source": "financial_report.pdf",
        "page_number": 32,
        "doc_category": "financial_statement",
        "doc_id": "doc_123",
        "chunk_id": "chunk_456",
        "financial_entities": {
          "monetary_values": ["$15M", "$28M"],
          "ratios": ["2.3", "1.8", "1.2", "40%"],
          "companies": ["TargetCorp"]
        }
      }
    },
    ...
  ],
  "processing_time": 1.45
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key financial risks identified in the target company'"'"'s balance sheet?",
    "collection_name": "my_ma_deal",
    "n_results": 5
  }'
```

**Python Example:**
```python
import requests
import json

url = "http://localhost:8000/query"
payload = {
    "query": "What are the key financial risks identified in the target company's balance sheet?",
    "collection_name": "my_ma_deal",
    "n_results": 5
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

### Task Status

Checks the status of a document processing task.

**Endpoint:** `POST /task/status`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "task_id": "upload_1712464589_financial_report.pdf"
}
```

**Response:**
```json
{
  "task_id": "upload_1712464589_financial_report.pdf",
  "status": "completed",
  "result": {
    "document_id": "doc_789",
    "chunks_created": 15,
    "entities_extracted": 127
  },
  "execution_time": 12.5
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/task/status" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "upload_1712464589_financial_report.pdf"}'
```

### List Collections

Lists all available document collections in the system.

**Endpoint:** `GET /collections`

**Response:**
```json
{
  "collections": [
    "default",
    "my_ma_deal",
    "acquisition_target_2023",
    "merger_documents"
  ]
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/collections" \
  -H "accept: application/json"
```

### Collection Statistics

Gets statistics for a specific document collection.

**Endpoint:** `GET /collections/{collection_name}/stats`

**Parameters:**
- `collection_name`: (Required) The name of the collection to get statistics for

**Response:**
```json
{
  "collection_name": "my_ma_deal",
  "document_count": 12,
  "chunk_count": 347,
  "metadata": {
    "last_updated": 1712464800,
    "document_types": {
      "financial_statement": 5,
      "contract": 2,
      "due_diligence_report": 3,
      "regulatory_filing": 2
    }
  }
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/collections/my_ma_deal/stats" \
  -H "accept: application/json"
```

## Error Handling

All API endpoints follow a consistent error response format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error

## Rate Limiting

Currently, there are no rate limits implemented. For production deployments, consider implementing rate limiting to prevent abuse.

## Versioning

The current API version is v1 (implicit in the paths). Future API versions will use explicit version prefixes in the path (e.g., `/v2/query`).
