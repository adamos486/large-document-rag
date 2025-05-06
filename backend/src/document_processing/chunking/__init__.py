"""
Financial document chunking module.

This module provides specialized chunking strategies for financial documents,
with a focus on preserving financial semantics and statement boundaries.
"""

from .base_chunker import BaseChunker, TextChunker
from .financial_chunkers import (
    FinancialStatementChunker,
    MDAndARiskChunker, 
    FinancialNotesChunker,
    FinancialStatementType,
    ChunkerFactory as FinancialChunkerFactory
)

# Unified factory that integrates all chunking strategies
class ChunkerFactory:
    """Factory for creating appropriate chunkers based on document types."""
    
    @staticmethod
    def create_chunker(document_type: str, statement_type=None, **kwargs) -> BaseChunker:
        """
        Create a chunker instance appropriate for the document type.
        
        Args:
            document_type: Type of document (e.g., "financial", "text", "tabular")
            statement_type: Type of financial statement (if known)
            **kwargs: Additional parameters to pass to the chunker
            
        Returns:
            An instance of a BaseChunker subclass
        """
        # For financial documents, use the financial chunker factory
        if document_type.lower() in [
            "financial", "finance", "10-k", "10-q", "annual_report", 
            "quarterly_report", "financial_statement", "income_statement", 
            "balance_sheet", "cash_flow"
        ]:
            return FinancialChunkerFactory.get_chunker(
                document_type, statement_type, **kwargs
            )
        
        # For regular text documents, use default chunker
        return TextChunker(**kwargs)


# Export the main interfaces
__all__ = [
    'BaseChunker',
    'TextChunker',
    'FinancialStatementChunker',
    'MDAndARiskChunker',
    'FinancialNotesChunker',
    'FinancialStatementType',
    'ChunkerFactory'
]
