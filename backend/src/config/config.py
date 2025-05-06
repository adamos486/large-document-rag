from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any, Union, Literal
import os
from enum import Enum

# Get the absolute path to the project root
ROOT_DIR = Path(__file__).parent.parent.absolute()

class FinancialEntityType(str, Enum):
    """Types of financial entities that can be extracted and analyzed."""
    COMPANY = "company"
    SUBSIDIARY = "subsidiary"
    METRIC = "metric"
    RATIO = "ratio"
    STATEMENT = "statement"
    ACCOUNT = "account"
    PERIOD = "period"
    CURRENCY = "currency"
    REGULATION = "regulation"
    RISK = "risk"


class FinancialTaskType(str, Enum):
    """Types of financial analysis tasks that can be performed."""
    RATIO_ANALYSIS = "ratio_analysis"
    TREND_ANALYSIS = "trend_analysis"
    VALUATION = "valuation"
    DUE_DILIGENCE = "due_diligence"
    RISK_ASSESSMENT = "risk_assessment"
    STATEMENT_ANALYSIS = "statement_analysis"
    SCENARIO_ANALYSIS = "scenario_analysis"
    FORECASTING = "forecasting"
    PEER_COMPARISON = "peer_comparison"
    CUSTOM = "custom"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HYBRID = "hybrid"


class HybridMode(str, Enum):
    """Modes for hybrid LLM provider selection."""
    TASK_SPECIFIC = "task_specific"  # Use specific provider for each task type
    ROUND_ROBIN = "round_robin"      # Alternate between providers
    FALLBACK = "fallback"            # Try primary, fall back to secondary if errors
    ENSEMBLE = "ensemble"            # Use multiple providers and combine results


class ChunkingStrategy(str, Enum):
    """Strategies for chunking financial documents."""
    # Legacy strategies (maintained for backward compatibility)
    SPATIAL = "spatial"              # Used for GIS documents
    ATTRIBUTE = "attribute"          # Used for GIS documents
    ENTITY = "entity"                # Used for CAD documents
    LAYER = "layer"                  # Used for CAD documents
    MIXED = "mixed"                  # Combination strategy
    
    # Financial document strategies
    STATEMENT_PRESERVING = "statement_preserving"  # Keep financial statements intact
    SECTION_BASED = "section_based"                # Chunk by document sections
    TABLE_AWARE = "table_aware"                    # Preserve tables within chunks
    METADATA_ENHANCED = "metadata_enhanced"        # Include document metadata
    ENTITY_CENTRIC = "entity_centric"              # Center chunks around entities


class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = ROOT_DIR
    DATA_DIR: Path = ROOT_DIR / "data"
    VECTOR_DB_PATH: Path = DATA_DIR / "vector_store"
    CACHE_DIR: Path = ROOT_DIR / "cache"
    MODEL_DIR: Path = ROOT_DIR / "models"
    
    # Financial-specific paths
    FINANCIAL_DATA_DIR: Path = DATA_DIR / "financial"
    FINANCIAL_CACHE_DIR: Path = CACHE_DIR / "financial"
    FINANCIAL_MODEL_DIR: Path = MODEL_DIR / "financial"
    
    # API settings
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    
    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Financial chunking strategy options: 
    # 'financial_statement', 'mda_risk', 'financial_notes', 'auto'
    FINANCIAL_CHUNK_STRATEGY: str = os.getenv("FINANCIAL_CHUNK_STRATEGY", "auto")

    # Financial document processing settings
    RESPECT_FINANCIAL_BOUNDARIES: bool = os.getenv("RESPECT_FINANCIAL_BOUNDARIES", "True").lower() == "true"
    FINANCIAL_ENTITY_EXTRACTION: bool = os.getenv("FINANCIAL_ENTITY_EXTRACTION", "True").lower() == "true"
    
    # Document processing
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Financial document processing
    FINANCIAL_CUSTOM_EMBEDDING: bool = True
    FINANCIAL_PROJECTION_LAYER: bool = True
    FINANCIAL_ENTITY_WEIGHTING: bool = True
    FINANCIAL_MAX_TABLE_SIZE: int = 100
    FINANCIAL_MIN_CONFIDENCE: float = 0.7
    FINANCIAL_USE_OCR: bool = True
    FINANCIAL_CHUNK_STRATEGY: str = ChunkingStrategy.STATEMENT_PRESERVING.value
    FINANCIAL_TABLE_EXTRACTION_MODE: str = "hybrid"
    
    # Financial causal analysis
    FINANCIAL_CAUSAL_MODELING: bool = True
    FINANCIAL_MONTE_CARLO_SIMULATIONS: int = 1000
    FINANCIAL_SCENARIO_DEPTH: int = 3
    
    # Vector database settings
    VECTOR_DB_TYPE: str = "sqlite"  # Changed from "postgres" to "sqlite"
    
    # ChromaDB server connection settings
    CHROMA_SERVER_HOST: str = "localhost"
    CHROMA_SERVER_PORT: int = 8010
    CHROMA_SERVER_SSL: bool = False
    
    # Multi-agent settings
    MAX_WORKERS: int = os.cpu_count() or 4
    AGENT_TIMEOUT: int = 300  # seconds
    
    # LLM settings
    DEFAULT_LLM_PROVIDER: str = LLMProvider.OPENAI.value
    OPENAI_MODEL: str = "gpt-4o"  # Default model for OpenAI
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"  # Default model for Anthropic
    TEMPERATURE: float = 0.0
    HYBRID_MODE: str = HybridMode.TASK_SPECIFIC.value
    TASK_DETECTION_THRESHOLD: float = 0.7
    
    # Control which LLM to use for specific tasks in hybrid mode
    # This allows us to leverage the strengths of different models
    # Format: {task_name: provider_name}
    HYBRID_LLM_TASKS: Dict[str, str] = {
        FinancialTaskType.RATIO_ANALYSIS.value: LLMProvider.ANTHROPIC.value,      # Claude excels at ratio analysis
        FinancialTaskType.TREND_ANALYSIS.value: LLMProvider.ANTHROPIC.value,     # Claude for trend analysis
        FinancialTaskType.VALUATION.value: LLMProvider.ANTHROPIC.value,          # Claude for valuation tasks
        FinancialTaskType.DUE_DILIGENCE.value: LLMProvider.ANTHROPIC.value,      # Claude for in-depth due diligence
        FinancialTaskType.RISK_ASSESSMENT.value: LLMProvider.ANTHROPIC.value,    # Claude for risk assessment
        FinancialTaskType.STATEMENT_ANALYSIS.value: LLMProvider.OPENAI.value,    # OpenAI for statement analysis
        FinancialTaskType.SCENARIO_ANALYSIS.value: LLMProvider.ANTHROPIC.value,  # Claude for complex scenarios
        FinancialTaskType.FORECASTING.value: LLMProvider.OPENAI.value,           # OpenAI for forecasting
        FinancialTaskType.PEER_COMPARISON.value: LLMProvider.OPENAI.value,       # OpenAI for comparisons
        "document_summarization": LLMProvider.ANTHROPIC.value,                  # Claude for large context windows
        "query_response": LLMProvider.OPENAI.value,                              # OpenAI for general queries
        "default": LLMProvider.OPENAI.value                                      # Default fallback
    }
    
    # API keys (to be loaded from .env)
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_chunking_strategy(self, document_type: str) -> str:
        """Get the appropriate chunking strategy based on document type."""
        # For backward compatibility
        if document_type.lower() == "gis":
            return self.GIS_CHUNK_STRATEGY
        elif document_type.lower() == "cad":
            return self.CAD_CHUNK_STRATEGY
        # For financial documents
        elif document_type.lower() in ["financial", "finance", "financial_document"]:
            return self.FINANCIAL_CHUNK_STRATEGY
        # Default to statement preserving for financial documents
        else:
            return ChunkingStrategy.STATEMENT_PRESERVING.value
    
    def get_llm_for_task(self, task: str) -> str:
        """Get the appropriate LLM provider for a specific task."""
        if self.DEFAULT_LLM_PROVIDER != LLMProvider.HYBRID.value:
            return self.DEFAULT_LLM_PROVIDER
            
        # For hybrid mode, look up task-specific provider
        return self.HYBRID_LLM_TASKS.get(task, self.HYBRID_LLM_TASKS["default"])


# Create a global settings instance
settings = Settings()

# Ensure necessary directories exist
settings.DATA_DIR.mkdir(exist_ok=True, parents=True)
settings.VECTOR_DB_PATH.mkdir(exist_ok=True, parents=True)
settings.CACHE_DIR.mkdir(exist_ok=True, parents=True)
settings.MODEL_DIR.mkdir(exist_ok=True, parents=True)
settings.FINANCIAL_DATA_DIR.mkdir(exist_ok=True, parents=True)
settings.FINANCIAL_CACHE_DIR.mkdir(exist_ok=True, parents=True)
settings.FINANCIAL_MODEL_DIR.mkdir(exist_ok=True, parents=True)
