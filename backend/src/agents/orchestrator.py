import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import ray
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..config.config import settings
from .base_agent import Agent
from .document_processor_agent import DocumentProcessorAgent
from .query_agent import QueryAgent

# Set up logging
logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Coordinates the execution of multiple agents for parallel document processing."""
    
    def __init__(
        self, 
        collection_name: str = "default",
        max_workers: Optional[int] = None,
        use_ray: bool = False
    ):
        """Initialize the agent orchestrator.
        
        Args:
            collection_name: Name of the vector database collection to use.
            max_workers: Maximum number of worker processes/threads.
            use_ray: Whether to use Ray for distributed processing.
        """
        self.collection_name = collection_name
        self.max_workers = max_workers or settings.MAX_WORKERS
        self.use_ray = use_ray
        self.agents = {}
        self.results = {}
        self.processing_lock = threading.Lock()
        
        # Initialize Ray if using distributed processing
        if self.use_ray:
            try:
                if not ray.is_initialized():
                    ray.init()
                logger.info("Ray initialized for distributed processing")
                
                # Define Ray remote functions for document processing
                # This enables distributed processing across multiple machines
                @ray.remote
                def process_document_ray(file_path, collection_name):
                    agent = DocumentProcessorAgent(
                        file_path=file_path,
                        collection_name=collection_name
                    )
                    return agent.execute()
                
                self.process_document_ray = process_document_ray
                
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                logger.warning("Falling back to ThreadPoolExecutor")
                self.use_ray = False
    
    def process_documents(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Process multiple documents in parallel.
        
        Args:
            file_paths: List of paths to documents to process.
            
        Returns:
            Dictionary with processing results.
        """
        logger.info(f"Processing {len(file_paths)} documents with {self.max_workers} workers")
        
        start_time = time.time()
        task_ids = []
        
        # Convert all paths to Path objects
        file_paths = [Path(path) for path in file_paths]
        
        # Check if files exist
        for path in file_paths:
            if not path.exists():
                logger.warning(f"File does not exist: {path}")
                file_paths.remove(path)
        
        if not file_paths:
            return {"error": "No valid files to process"}
        
        if self.use_ray:
            # Use Ray for distributed processing
            try:
                # Submit tasks to Ray
                tasks = [
                    self.process_document_ray.remote(str(path), self.collection_name)
                    for path in file_paths
                ]
                
                # Get results as they complete
                for i, task_id in enumerate(tasks):
                    task_ids.append(f"task_{i}")
                    self.agents[f"task_{i}"] = {
                        "file_path": str(file_paths[i]),
                        "status": "running"
                    }
                
                # Wait for all tasks to complete
                results = ray.get(tasks)
                
                # Store results
                for i, result in enumerate(results):
                    self.results[f"task_{i}"] = result
                    self.agents[f"task_{i}"]["status"] = "completed"
                
            except Exception as e:
                logger.error(f"Error in Ray processing: {e}")
                return {"error": f"Ray processing failed: {str(e)}"}
        else:
            # Use ThreadPoolExecutor for local parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a future for each file
                future_to_path = {}
                
                for path in file_paths:
                    task_id = f"task_{uuid.uuid4().hex[:8]}"
                    task_ids.append(task_id)
                    
                    agent = DocumentProcessorAgent(
                        file_path=path,
                        collection_name=self.collection_name,
                        agent_id=task_id
                    )
                    
                    self.agents[task_id] = agent
                    
                    # Submit task to executor
                    future = executor.submit(agent.execute)
                    future_to_path[future] = task_id
                
                # Process results as they complete
                for future in as_completed(future_to_path):
                    task_id = future_to_path[future]
                    try:
                        result = future.result()
                        self.results[task_id] = result
                    except Exception as e:
                        logger.error(f"Agent execution failed: {e}")
                        self.results[task_id] = {"status": "failed", "error": str(e)}
        
        # Compile processing summary
        processing_time = time.time() - start_time
        
        summary = {
            "task_count": len(task_ids),
            "completed_count": sum(1 for task_id in task_ids if self.results.get(task_id, {}).get("status") == "completed"),
            "failed_count": sum(1 for task_id in task_ids if self.results.get(task_id, {}).get("status") == "failed"),
            "processing_time": processing_time,
            "task_ids": task_ids,
            "collection_name": self.collection_name
        }
        
        return summary
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single document.
        
        Args:
            file_path: Path to the document to process.
            
        Returns:
            Processing result.
        """
        try:
            file_path = Path(file_path)
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            agent = DocumentProcessorAgent(
                file_path=file_path,
                collection_name=self.collection_name,
                agent_id=task_id
            )
            
            self.agents[task_id] = agent
            result = agent.execute()
            self.results[task_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def query(
        self, 
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Query the vector database.
        
        Args:
            query_text: The query text.
            filters: Optional metadata filters.
            n_results: Number of results to retrieve.
            
        Returns:
            Query result.
        """
        try:
            task_id = f"query_{uuid.uuid4().hex[:8]}"
            
            agent = QueryAgent(
                collection_name=self.collection_name,
                n_results=n_results,
                agent_id=task_id
            )
            
            self.agents[task_id] = agent
            result = agent.run(query=query_text, filters=filters)
            self.results[task_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            Task status or None if task doesn't exist.
        """
        if task_id in self.agents:
            if isinstance(self.agents[task_id], Agent):
                return self.agents[task_id].get_status()
            else:
                return self.agents[task_id]
        
        return None
    
    def get_all_task_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all tasks.
        
        Returns:
            Dictionary mapping task IDs to task statuses.
        """
        statuses = {}
        
        for task_id, agent in self.agents.items():
            if isinstance(agent, Agent):
                statuses[task_id] = agent.get_status()
            else:
                statuses[task_id] = agent
        
        return statuses
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed task.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            Task result or None if task doesn't exist or hasn't completed.
        """
        return self.results.get(task_id)
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get the results of all completed tasks.
        
        Returns:
            Dictionary mapping task IDs to task results.
        """
        return self.results.copy()
