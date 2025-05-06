"""
LLM Provider Factory that supports multiple LLM backends including OpenAI and Anthropic.
This module allows for easy switching between different LLM providers and supports
a hybrid approach where different tasks can use different LLM providers.
"""

import os
import logging
from enum import Enum
from typing import Dict, Any, Optional, Union, Callable, List, Tuple

from ..config.config import settings
from .financial_task_detector import financial_task_detector, FinancialTaskType

# Set up logging
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HYBRID = "hybrid"

class LLMTask(str, Enum):
    """Different tasks that may benefit from specific LLM providers"""
    FINANCIAL_ANALYSIS = "financial_analysis"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    QUERY_RESPONSE = "query_response"
    DEFAULT = "default"

class LLMProviderFactory:
    """Factory class for creating LLM instances based on provider type"""
    
    def __init__(self):
        """Initialize the LLM provider factory"""
        self._provider_type = settings.DEFAULT_LLM_PROVIDER.lower()
        self._temperature = settings.TEMPERATURE
        self._openai_model = settings.OPENAI_MODEL
        self._anthropic_model = settings.ANTHROPIC_MODEL
        self._hybrid_tasks = settings.HYBRID_LLM_TASKS
        self._llm_instances: Dict[str, Any] = {}
        
        # Validate settings
        if self._provider_type not in [e.value for e in LLMProvider]:
            logger.warning(f"Unknown LLM provider: {self._provider_type}. Defaulting to OpenAI.")
            self._provider_type = LLMProvider.OPENAI
    
    def get_llm_for_task(self, task: str = LLMTask.DEFAULT, query: str = None, override_provider: str = None) -> Any:
        """Get the appropriate LLM for a specific task.
        
        Args:
            task: The task to get an LLM for (from LLMTask enum)
            query: Optional query text for more precise task detection
            override_provider: Optional override to force a specific provider
            
        Returns:
            An LLM instance appropriate for the task
        """
        if override_provider:
            provider = override_provider
        elif self._provider_type == LLMProvider.HYBRID and query:
            # In hybrid mode with a query, use the financial task detector for intelligent routing
            provider = financial_task_detector.get_llm_provider_for_task(query)
            logger.info(f"Financial task detector selected provider: {provider} for query")
        elif self._provider_type == LLMProvider.HYBRID:
            # In hybrid mode without a query, select provider based on task
            provider = self._hybrid_tasks.get(task, self._hybrid_tasks.get(LLMTask.DEFAULT))
        else:
            # Otherwise use the configured provider
            provider = self._provider_type
        
        # Check if we already have an instance for this provider
        if provider in self._llm_instances:
            return self._llm_instances[provider]
        
        # Otherwise create and cache a new instance
        llm = self._create_llm_instance(provider)
        if llm:
            self._llm_instances[provider] = llm
            return llm
        
        logger.error(f"Failed to create LLM instance for provider: {provider}")
        return None
        
    def analyze_financial_query(self, query: str) -> Tuple[str, Dict[str, float], Dict[str, List[str]]]:
        """Analyze a financial query to determine task type and extract entities.
        
        Args:
            query: The query text to analyze
            
        Returns:
            Tuple containing:
            - Detected task type
            - Confidence scores for each task type
            - Extracted financial entities
        """
        task_type, confidence_scores = financial_task_detector.detect_task_type(query)
        financial_entities = financial_task_detector.extract_financial_entities(query)
        
        return task_type, confidence_scores, financial_entities
    
    def _create_llm_instance(self, provider: str) -> Optional[Any]:
        """Create an LLM instance for the specified provider.
        
        Args:
            provider: The LLM provider to create an instance for
            
        Returns:
            An LLM instance or None if creation failed
        """
        if provider == LLMProvider.OPENAI:
            return self._create_openai_instance()
        elif provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic_instance()
        else:
            logger.error(f"Unsupported provider type: {provider}")
            return None
    
    def _create_openai_instance(self) -> Optional[Any]:
        """Create an OpenAI LLM instance"""
        if not os.environ.get("OPENAI_API_KEY") and not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set. OpenAI will not be available.")
            return None
        
        try:
            # Try to use LangChain's ChatOpenAI first, as it provides better streaming support
            from langchain_community.chat_models import ChatOpenAI
            return ChatOpenAI(
                model_name=self._openai_model,
                temperature=self._temperature
            )
        except ImportError:
            logger.warning("langchain_community.chat_models not available. Trying direct OpenAI.")
            try:
                # Fall back to the regular OpenAI model
                from langchain_community.llms import OpenAI
                return OpenAI(
                    model_name=self._openai_model,
                    temperature=self._temperature
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                return None
    
    def _create_anthropic_instance(self) -> Optional[Any]:
        """Create an Anthropic Claude LLM instance"""
        if not os.environ.get("ANTHROPIC_API_KEY") and not settings.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set. Anthropic Claude will not be available.")
            return None
        
        try:
            # Try to use LangChain's Anthropic integration
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=self._anthropic_model,
                temperature=self._temperature
            )
        except ImportError:
            logger.warning("langchain_anthropic not available. Trying direct Anthropic API.")
            try:
                # Fallback to direct Anthropic API if LangChain integration isn't available
                from anthropic import Anthropic
                
                # Create a simple wrapper class to match LangChain's interface
                class AnthropicWrapper:
                    def __init__(self, model_name, temperature):
                        self.client = Anthropic()
                        self.model_name = model_name
                        self.temperature = temperature
                    
                    def invoke(self, prompt, **kwargs):
                        message = self.client.messages.create(
                            model=self.model_name,
                            max_tokens=4096,
                            temperature=self.temperature,
                            system="You are a helpful AI assistant for financial due diligence tasks.",
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        return message.content[0].text
                
                return AnthropicWrapper(
                    model_name=self._anthropic_model,
                    temperature=self._temperature
                )
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
                return None

# Create a singleton instance
llm_provider_factory = LLMProviderFactory()
