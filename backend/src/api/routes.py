import os
import tempfile
import shutil
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time

from fastapi import Form, APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query, status
from fastapi.responses import JSONResponse

from config.config import settings
from src.agents.orchestrator import AgentOrchestrator
from src.vector_store.vector_db import VectorStore
from src.exceptions import DocumentProcessingError, QueryProcessingError
from .models import (
    DocumentUploadRequest,
    DocumentProcessingResponse,
    QueryRequest,
    QueryResponse,
    TaskStatusRequest,
    TaskStatusResponse,
    CollectionListResponse,
    CollectionStatsResponse,
    ProcessBatchRequest,
    FinancialDocumentMetadata
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Create orchestrator instance for API
orchestrator = AgentOrchestrator()

# Cache for collection stats
_collection_cache = {}
_cache_timestamp = 0
_CACHE_TTL = 60  # seconds

def get_orchestrator(collection_name: str = "default", max_workers: Optional[int] = None):
    """Get an orchestrator instance with the specified collection name."""
    max_workers = max_workers or settings.MAX_WORKERS
    return AgentOrchestrator(collection_name=collection_name, max_workers=max_workers)

@router.post("/upload", response_model=DocumentProcessingResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
    custom_metadata: str = Form(default="{}")
):
    """Upload and process a document."""
    try:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp:
            # Copy uploaded file to temp file
            shutil.copyfileobj(file.file, temp)
            temp_path = Path(temp.name)
        
        # Parse custom metadata if provided
        metadata_dict = {}
        if custom_metadata:
            try:
                metadata_dict = json.loads(custom_metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in custom_metadata: {custom_metadata}")
        
        # Get orchestrator for this collection
        current_orchestrator = get_orchestrator(collection_name=collection_name)
        
        # Process document in background
        task_id = f"upload_{int(time.time())}_{file.filename}"
        
        def process_document_task():
            try:
                # Process the document
                current_orchestrator.process_document(temp_path)
                
                # Clean up temp file after processing
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        # Add task to background tasks
        background_tasks.add_task(process_document_task)
        
        return DocumentProcessingResponse(
            task_id=task_id,
            file_name=file.filename,
            collection_name=collection_name,
            status="processing"
        )
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the vector database."""
    try:
        # Get orchestrator for this collection
        current_orchestrator = get_orchestrator(collection_name=request.collection_name)
        
        # Execute query
        start_time = time.time()
        result = current_orchestrator.query(
            query_text=request.query,
            filters=request.filters,
            n_results=request.n_results
        )
        
        # Format response
        documents = []
        if "retrieved_documents" in result:
            documents = result["retrieved_documents"]
        
        return QueryResponse(
            query=request.query,
            llm_response=result.get("llm_response"),
            documents=documents,
            processing_time=time.time() - start_time
        )
    
    except Exception as e:
        logger.error(f"Error querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/task/status", response_model=TaskStatusResponse)
async def get_task_status(request: TaskStatusRequest):
    """Get the status of a task."""
    try:
        # Check status in global orchestrator
        task_status = orchestrator.get_task_status(request.task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
        
        return TaskStatusResponse(
            task_id=request.task_id,
            status=task_status.get("status", "unknown"),
            result=orchestrator.get_result(request.task_id),
            error=task_status.get("error"),
            execution_time=task_status.get("execution_time")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """List all collections in the vector database."""
    try:
        # Get all collections
        vector_store = VectorStore()
        collections = vector_store.get_collections()
        
        return CollectionListResponse(
            collections=collections
        )
    
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/{collection_name}/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(collection_name: str):
    """Get statistics for a collection."""
    global _collection_cache, _cache_timestamp
    
    try:
        # Check cache first
        current_time = time.time()
        if (collection_name in _collection_cache and 
            current_time - _cache_timestamp < _CACHE_TTL):
            return _collection_cache[collection_name]
        
        # Initialize vector store for the collection
        vector_store = VectorStore(collection_name=collection_name)
        
        # Get collection stats
        chunk_count = vector_store.count()
        
        # Get document count (approximate based on distinct doc_ids)
        document_count = 0
        try:
            # This is an approximation, as we need to fetch all chunks to count unique docs
            # For a large collection, this could be inefficient
            chunks = vector_store.get_chunks_by_filter({})
            if chunks:
                doc_ids = set()
                for chunk in chunks:
                    doc_id = chunk.metadata.get("doc_id")
                    if doc_id:
                        doc_ids.add(doc_id)
                document_count = len(doc_ids)
        except Exception as e:
            logger.warning(f"Could not determine exact document count: {e}")
        
        stats = CollectionStatsResponse(
            collection_name=collection_name,
            document_count=document_count,
            chunk_count=chunk_count,
            metadata={
                "last_updated": current_time
            }
        )
        
        # Update cache
        _collection_cache[collection_name] = stats
        _cache_timestamp = current_time
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/process", response_model=Dict[str, Any])
async def process_batch(request: ProcessBatchRequest, files: List[UploadFile] = File(...)):
    """Process multiple documents in batch."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create temp directory to store uploads
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            file_paths = []
            
            # Save uploaded files to temp directory
            for file in files:
                file_path = temp_dir_path / file.filename
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                file_paths.append(file_path)
            
            # Get orchestrator with specified parameters
            current_orchestrator = AgentOrchestrator(
                collection_name=request.collection_name,
                max_workers=request.parallel_workers,
                use_ray=request.use_ray
            )
            
            # Process documents
            result = current_orchestrator.process_documents(file_paths)
            
            return result
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))
