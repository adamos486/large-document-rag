import os
import sys
import uvicorn
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file if it exists
load_dotenv()

# Import settings from config
from config.config import settings
from src.api.app import app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Large Document RAG System')
    parser.add_argument('--host', type=str, default=settings.API_HOST, help='Host to run the API server on')
    parser.add_argument('--port', type=int, default=settings.API_PORT, help='Port to run the API server on')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    # Run the API server
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Create necessary directories
    settings.DATA_DIR.mkdir(exist_ok=True, parents=True)
    settings.VECTOR_DB_PATH.mkdir(exist_ok=True, parents=True)
    
    # Start the application
    main()
