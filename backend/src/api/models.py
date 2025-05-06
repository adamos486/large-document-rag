from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from datetime import date, datetime
from pathlib import Path

class FinancialDocumentType(str, Enum):
    """Enumeration of supported financial document types."""
    FINANCIAL_STATEMENT = "financial_statement"
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    TAX_DOCUMENT = "tax_document"
    CONTRACT = "contract"
    INVOICE = "invoice"
    BUDGET = "budget"
    AUDIT_REPORT = "audit_report"
    DUE_DILIGENCE = "due_diligence"
    VALUATION_REPORT = "valuation_report"
    FINANCIAL_MODEL = "financial_model"
    OTHER = "other"

class FinancialStatementType(str, Enum):
    """Enumeration of financial statement types."""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    STATEMENT_OF_EQUITY = "statement_of_equity"
    FOOTNOTES = "footnotes"
    COMBINED = "combined"

class FinancialPeriod(BaseModel):
    """Model representing a financial reporting period."""
    period_type: Literal["annual", "quarterly", "monthly", "custom"] = Field(..., description="Type of financial period")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year")
    quarter: Optional[int] = Field(None, description="Quarter number (1-4)")
    month: Optional[int] = Field(None, description="Month number (1-12)")
    start_date: Optional[date] = Field(None, description="Start date for custom periods")
    end_date: Optional[date] = Field(None, description="End date for custom periods")
    
    @validator('quarter')
    def validate_quarter(cls, v):
        if v is not None and (v < 1 or v > 4):
            raise ValueError('Quarter must be between 1 and 4')
        return v
    
    @validator('month')
    def validate_month(cls, v):
        if v is not None and (v < 1 or v > 12):
            raise ValueError('Month must be between 1 and 12')
        return v

class FinancialEntity(BaseModel):
    """Model representing a financial entity."""
    entity_type: Literal["company", "currency", "monetary_value", "percentage", "ratio", "date", "financial_term", "risk_indicator"] = Field(..., description="Type of financial entity")
    value: str = Field(..., description="Entity value")
    context: Optional[str] = Field(None, description="Surrounding context")
    confidence: Optional[float] = Field(None, description="Confidence score")

class FinancialDocumentMetadata(BaseModel):
    """Metadata model for financial documents."""
    document_type: FinancialDocumentType = Field(..., description="Type of financial document")
    statement_type: Optional[FinancialStatementType] = Field(None, description="Type of financial statement if applicable")
    reporting_entity: Optional[str] = Field(None, description="Entity issuing the document")
    financial_period: Optional[FinancialPeriod] = Field(None, description="Financial reporting period")
    currency: Optional[str] = Field(None, description="Primary currency used in the document")
    extracted_entities: Optional[List[FinancialEntity]] = Field(default_factory=list, description="Financial entities extracted from the document")
    key_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key financial metrics extracted from the document")
    risk_level: Optional[Literal["low", "medium", "high"]] = Field(None, description="Assessed financial risk level")
    sentiment: Optional[Literal["positive", "neutral", "negative"]] = Field(None, description="Overall financial sentiment")
    source_confidence: Optional[float] = Field(None, description="Confidence in the source data quality")
    processing_timestamp: Optional[datetime] = Field(None, description="When the document was processed")
    custom_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional custom metadata")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    collection_name: str = Field(default="default", description="Collection name to store the document in")
    document_type: Optional[FinancialDocumentType] = Field(None, description="Type of financial document being uploaded")
    statement_type: Optional[FinancialStatementType] = Field(None, description="Type of financial statement if applicable")
    reporting_entity: Optional[str] = Field(None, description="Entity issuing the document")
    financial_year: Optional[int] = Field(None, description="Financial year of the document")
    custom_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional custom metadata")

class DocumentProcessingResponse(BaseModel):
    """Response model for document processing."""
    task_id: str = Field(..., description="Task ID for tracking the processing status")
    file_name: str = Field(..., description="Name of the file being processed")
    collection_name: str = Field(..., description="Collection name where the document is stored")
    status: str = Field(..., description="Processing status")
    document_type: Optional[FinancialDocumentType] = Field(None, description="Detected type of financial document")
    document_metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata from the document")
    entity_count: Optional[int] = Field(None, description="Number of financial entities extracted")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

class QueryRequest(BaseModel):
    """Request model for vector store query."""
    query: str = Field(..., description="Query text")
    collection_name: str = Field(default="default", description="Collection name to query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for the query")
    n_results: int = Field(default=5, description="Number of results to return")
    document_types: Optional[List[FinancialDocumentType]] = Field(default=None, description="Filter by specific financial document types")
    statement_types: Optional[List[FinancialStatementType]] = Field(default=None, description="Filter by specific financial statement types")
    reporting_entity: Optional[str] = Field(default=None, description="Filter by reporting entity")
    date_range: Optional[Dict[str, date]] = Field(default=None, description="Filter by date range (start_date, end_date)")
    financial_year: Optional[int] = Field(default=None, description="Filter by financial year")
    financial_quarter: Optional[int] = Field(default=None, description="Filter by financial quarter")
    include_metadata: bool = Field(default=True, description="Whether to include document metadata in the response")
    llm_provider: Optional[Literal["openai", "anthropic", "hybrid"]] = Field(default=None, description="Specify which LLM provider to use for this query")

class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str = Field(..., description="Original query text")
    llm_response: Optional[str] = Field(default=None, description="Generated response from LLM")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    llm_provider_used: Optional[str] = Field(default=None, description="LLM provider that was used for the response")
    financial_entities: Optional[List[FinancialEntity]] = Field(default=None, description="Financial entities extracted from the query")
    monetary_values: Optional[List[Dict[str, Any]]] = Field(default=None, description="Monetary values extracted from results")
    percentages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Percentage values extracted from results")
    key_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Key financial metrics extracted from results")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score for the generated response")

class TaskStatusRequest(BaseModel):
    """Request model for task status."""
    task_id: str = Field(..., description="Task ID to check status for")

class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result if completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds if completed")

class CollectionListResponse(BaseModel):
    """Response model for collections list."""
    collections: List[str] = Field(..., description="List of available collections")

class CollectionStatsResponse(BaseModel):
    """Response model for collection statistics."""
    collection_name: str = Field(..., description="Collection name")
    document_count: int = Field(..., description="Number of documents in the collection")
    chunk_count: int = Field(..., description="Number of chunks in the collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")
    financial_document_types: Optional[Dict[str, int]] = Field(default=None, description="Count of each financial document type")
    financial_statement_types: Optional[Dict[str, int]] = Field(default=None, description="Count of each financial statement type")
    reporting_entities: Optional[List[str]] = Field(default=None, description="List of reporting entities in collection")
    financial_years: Optional[List[int]] = Field(default=None, description="List of financial years covered")
    total_financial_entities: Optional[int] = Field(default=None, description="Total number of financial entities extracted")
    average_entities_per_document: Optional[float] = Field(default=None, description="Average number of financial entities per document")

class ProcessBatchRequest(BaseModel):
    """Request model for batch document processing."""
    collection_name: str = Field(default="default", description="Collection name to store the documents in")
    parallel_workers: Optional[int] = Field(default=None, description="Number of parallel workers to use")
    use_ray: bool = Field(default=False, description="Whether to use Ray for distributed processing")
