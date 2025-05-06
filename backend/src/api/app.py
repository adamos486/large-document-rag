import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from pathlib import Path

from .routes import router
from config.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Large Document RAG API",
    description="API for processing and querying large GIS and CAD files using a RAG approach",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Include API routes
app.include_router(router, prefix="/api")

# Add root endpoint
@app.get("/")
async def root():
    return {
        "message": "Large Document RAG API",
        "docs_url": "/docs",
        "version": "0.1.0"
    }

# Add health check endpoint
@app.get("/health", response_model=dict)
async def health() -> dict:
    """Check system health status and ensure required directories exist.
    
    Returns:
        dict: Health status information including directory checks and service status
    """
    health_data = {
        "status": "healthy",
        "services": {},
        "directories": {}
    }
    
    # Ensure data directories exist
    data_dir = settings.DATA_DIR
    vector_db_path = settings.VECTOR_DB_PATH
    
    try:
        # Check and create directories if needed
        directories = {
            "data_dir": data_dir,
            "vector_db_path": vector_db_path
        }
        
        for name, dir_path in directories.items():
            dir_exists = dir_path.exists()
            health_data["directories"][name] = {
                "path": str(dir_path),
                "exists": dir_exists,
                "writable": os.access(dir_path.parent, os.W_OK) if dir_path.parent.exists() else False
            }
            
            # Create if it doesn't exist
            if not dir_exists:
                dir_path.mkdir(parents=True, exist_ok=True)
                health_data["directories"][name]["created"] = True
        
        # Check vector store accessibility (if implemented)
        try:
            # Simple check to see if vector store is accessible
            from src.vector_db.vector_store import VectorStore
            vector_store = VectorStore()
            # Just testing connection, no need to query
            health_data["services"]["vector_store"] = {"status": "available"}
        except Exception as vs_error:
            logger.warning(f"Vector store health check failed: {vs_error}")
            health_data["services"]["vector_store"] = {
                "status": "unavailable",
                "error": str(vs_error)
            }
            health_data["status"] = "degraded"
        
        # Check LLM connectivity (only if needed for health check)
        try:
            from src.llm.llm_provider import LLMProviderFactory
            llm_factory = LLMProviderFactory()
            # Just checking initialization, not making actual API calls
            health_data["services"]["llm_provider"] = {"status": "initialized"}
        except Exception as llm_error:
            logger.warning(f"LLM provider health check failed: {llm_error}")
            health_data["services"]["llm_provider"] = {
                "status": "unavailable",
                "error": str(llm_error)
            }
            health_data["status"] = "degraded"
            
        return health_data
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
