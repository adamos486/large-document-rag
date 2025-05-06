from abc import ABC, abstractmethod
import uuid
import logging
import time
from typing import List, Dict, Any, Optional, Union, Callable
import traceback

from ..config.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Agent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent. If None, a UUID will be generated.
            name: Human-readable name for this agent.
        """
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.name = name or self.__class__.__name__
        self.status = "initialized"
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the agent's task.
        
        Args:
            **kwargs: Additional arguments to pass to the run method.
        
        Returns:
            Dict containing results of agent execution.
        """
        self.status = "running"
        self.start_time = time.time()
        self.result = None
        self.error = None
        
        try:
            self.result = self.run(**kwargs)
            self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            logger.error(f"Agent {self.name} failed: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.end_time = time.time()
            
        return self.get_status()
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Run the agent's task. This method should be implemented by subclasses.
        
        Args:
            **kwargs: Additional arguments specific to the agent implementation.
            
        Returns:
            The result of the agent's task.
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of this agent.
        
        Returns:
            Dictionary with agent status information.
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "execution_time": (self.end_time - self.start_time) if self.end_time else None,
        }
