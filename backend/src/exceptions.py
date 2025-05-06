"""
Custom exceptions for the Financial Due Diligence RAG system.

This module defines custom exceptions that provide more context and better error handling
for specific failure cases in the financial document processing workflow.
"""

class BaseRagException(Exception):
    """Base exception for all RAG system errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DocumentProcessingError(BaseRagException):
    """Exception raised when document processing fails."""
    def __init__(self, message: str, document_path: str = None, format_type: str = None, status_code: int = 500):
        self.document_path = document_path
        self.format_type = format_type
        detail = f"Failed to process document"
        if document_path:
            detail += f" at {document_path}"
        if format_type:
            detail += f" of type {format_type}"
        detail += f": {message}"
        super().__init__(detail, status_code)


class FinancialDataExtractionError(DocumentProcessingError):
    """Exception raised when financial data extraction fails."""
    def __init__(self, message: str, document_path: str = None, entity_type: str = None, status_code: int = 500):
        self.entity_type = entity_type
        detail = message
        if entity_type:
            detail = f"Failed to extract {entity_type}: {message}"
        super().__init__(detail, document_path, "financial", status_code)


class QueryProcessingError(BaseRagException):
    """Exception raised when query processing fails."""
    def __init__(self, message: str, query: str = None, status_code: int = 500):
        self.query = query
        detail = f"Query processing failed: {message}"
        if query:
            # Only include part of the query in the error message for privacy/security
            truncated_query = query[:30] + "..." if len(query) > 30 else query
            detail += f" (query: '{truncated_query}')"
        super().__init__(detail, status_code)


class LLMProviderError(BaseRagException):
    """Exception raised when there's an issue with the LLM provider."""
    def __init__(self, message: str, provider: str = None, status_code: int = 503):
        self.provider = provider
        detail = f"LLM provider error"
        if provider:
            detail += f" with {provider}"
        detail += f": {message}"
        super().__init__(detail, status_code)


class AnthropicProviderError(LLMProviderError):
    """Exception raised when there's an issue with the Anthropic provider."""
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message, "Anthropic", status_code)


class OpenAIProviderError(LLMProviderError):
    """Exception raised when there's an issue with the OpenAI provider."""
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message, "OpenAI", status_code)


class VectorDBError(BaseRagException):
    """Exception raised when there's an issue with the vector database."""
    def __init__(self, message: str, operation: str = None, collection: str = None, status_code: int = 500):
        self.operation = operation
        self.collection = collection
        detail = f"Vector database error"
        if operation:
            detail += f" during {operation}"
        if collection:
            detail += f" on collection '{collection}'"
        detail += f": {message}"
        super().__init__(detail, status_code)


class FinancialDataParsingError(BaseRagException):
    """Exception raised when parsing financial data fails."""
    def __init__(self, message: str, data_type: str = None, status_code: int = 400):
        self.data_type = data_type
        detail = f"Failed to parse financial data"
        if data_type:
            detail += f" of type {data_type}"
        detail += f": {message}"
        super().__init__(detail, status_code)


class UnsupportedFinancialDocumentError(DocumentProcessingError):
    """Exception raised when an unsupported financial document type is encountered."""
    def __init__(self, document_path: str = None, format_type: str = None):
        message = f"Unsupported financial document format: {format_type}"
        super().__init__(message, document_path, format_type, 415)
