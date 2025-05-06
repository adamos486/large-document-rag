# Getting Started with Financial Due Diligence RAG

This guide will help you set up and start using the Financial Due Diligence RAG system for M&A processes.

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended for parallel processing
- **RAM**: Minimum 8GB, 16GB+ recommended for large documents
- **Storage**: 10GB+ for code, dependencies, and vector database

### Software Requirements
- **Python**: 3.10+ (3.12 recommended)
- **Operating System**: Linux, macOS, or Windows
- **Dependencies**: See `requirements.txt` for the complete list

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/adamos486/large-document-rag.git
cd large-document-rag
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment
On macOS/Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Install spaCy Model
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl
```

### Step 6: Set Up Environment Variables
Create a `.env` file in the project root with the following variables:
```
# LLM Settings
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
LLM_PROVIDER=hybrid  # Options: openai, anthropic, hybrid

# API Settings
API_PORT=8000
API_HOST=0.0.0.0
```

## Configuration Options

The system is highly configurable through the settings in `src/config/config.py`. Here are the key settings you can adjust:

### Document Processing Settings
- `CHUNK_SIZE`: Size of document chunks (default: 1024)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 128)
- `EMBEDDING_MODEL`: Model to use for embeddings (default: "sentence-transformers/all-mpnet-base-v2")

### LLM Settings
- `LLM_PROVIDER`: The LLM provider to use ("openai", "anthropic", or "hybrid")
- `LLM_MODEL`: Default OpenAI model (default: "gpt-4")
- `ANTHROPIC_MODEL`: Default Anthropic model (default: "claude-3-opus-20240229")
- `TEMPERATURE`: Temperature for response generation (default: 0.0)
- `HYBRID_LLM_TASKS`: Task-specific LLM provider mappings (for hybrid mode)

### Multi-Agent Settings
- `MAX_WORKERS`: Maximum number of parallel workers (default: CPU count)
- `AGENT_TIMEOUT`: Timeout for agent tasks in seconds (default: 300)

### Vector Database Settings
- `VECTOR_DB_TYPE`: Vector database type (default: "chroma")
- `VECTOR_DB_PATH`: Path to store vector database (default: "data/vector_store")

## Quick Start Guide

### Step 1: Start the Server
```bash
python src/main.py
```

This will start the API server on the configured host and port (default: http://0.0.0.0:8000).

### Step 2: Upload Financial Documents
Use the API to upload financial documents for processing:

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/financial_document.pdf" \
  -F "collection_name=my_ma_deal"
```

### Step 3: Check Processing Status
The upload endpoint returns a task ID that you can use to check the processing status:

```bash
curl -X POST "http://localhost:8000/task/status" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "your_task_id_here"}'
```

### Step 4: Query the System
Once documents are processed, you can query the system:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key financial risks identified in the target company's balance sheet?",
    "collection_name": "my_ma_deal",
    "n_results": 5
  }'
```

## Next Steps

- Explore the [API Reference](./api-reference.md) for all available endpoints
- Learn about the [Financial Document Processing](./financial-document-processing.md) capabilities
- Read the [Developer Guide](./developer-guide.md) to understand the system architecture
