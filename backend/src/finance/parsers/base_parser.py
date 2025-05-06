"""
Base Financial Parser

This module defines the base classes and interfaces for financial document parsing.
It provides a foundation for specialized parsers for different document types and formats.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from ..models.statement import (
    FinancialStatement,
    StatementType,
    StatementMetadata,
    TimePeriod,
    LineItem
)

logger = logging.getLogger(__name__)


class ParsingResult:
    """Container for the results of a parsing operation."""
    
    def __init__(self):
        self.statements: List[FinancialStatement] = []
        self.metadata: Dict[str, Any] = {}
        self.confidence_score: float = 0.0
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_statement(self, statement: FinancialStatement) -> None:
        """Add a financial statement to the results."""
        self.statements.append(statement)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error(f"Parsing error: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(f"Parsing warning: {warning}")
    
    def is_successful(self) -> bool:
        """Check if parsing was successful (no errors)."""
        return len(self.errors) == 0
    
    def get_statement_by_type(self, statement_type: StatementType) -> Optional[FinancialStatement]:
        """Get the first statement of the specified type."""
        for statement in self.statements:
            if statement.metadata.statement_type == statement_type:
                return statement
        return None
    
    def get_statements_by_type(self, statement_type: StatementType) -> List[FinancialStatement]:
        """Get all statements of the specified type."""
        return [s for s in self.statements if s.metadata.statement_type == statement_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary."""
        return {
            "statements": [s.to_dict() for s in self.statements],
            "metadata": self.metadata,
            "confidence_score": self.confidence_score,
            "errors": self.errors,
            "warnings": self.warnings,
            "successful": self.is_successful()
        }


class FinancialDocumentParser(ABC):
    """
    Abstract base class for financial document parsers.
    
    This class defines the interface that all financial document parsers must implement.
    """
    
    @abstractmethod
    def parse(self, document_path: Union[str, Path], **kwargs) -> ParsingResult:
        """
        Parse a financial document and extract structured data.
        
        Args:
            document_path: Path to the financial document
            **kwargs: Additional parser-specific parameters
            
        Returns:
            ParsingResult containing extracted financial statements and metadata
        """
        pass
    
    @abstractmethod
    def can_parse(self, document_path: Union[str, Path]) -> bool:
        """
        Check if this parser can handle the given document.
        
        Args:
            document_path: Path to the financial document
            
        Returns:
            True if the parser can handle this document, False otherwise
        """
        pass
    
    def _get_file_extension(self, document_path: Union[str, Path]) -> str:
        """Get the file extension from a document path."""
        return os.path.splitext(str(document_path))[1].lower()


class StatementDetector:
    """
    Detects the type of financial statement in a document or section.
    
    This class analyzes text content to determine what type of financial statement it contains.
    """
    
    def __init__(self):
        # Keywords that indicate different types of financial statements
        self.balance_sheet_keywords = [
            "balance sheet", "statement of financial position", "assets and liabilities",
            "total assets", "total liabilities", "stockholders' equity", "shareholders' equity",
            "current assets", "non-current assets", "long-term assets",
            "current liabilities", "non-current liabilities", "long-term liabilities"
        ]
        
        self.income_statement_keywords = [
            "income statement", "profit and loss", "statement of operations", 
            "statement of earnings", "statement of comprehensive income",
            "revenue", "sales", "gross profit", "operating income", "net income",
            "cost of goods sold", "cost of revenue", "operating expenses", "tax expense"
        ]
        
        self.cash_flow_keywords = [
            "cash flow statement", "statement of cash flows", "cash flows",
            "operating activities", "investing activities", "financing activities",
            "net cash provided by", "net cash used in", "cash and cash equivalents"
        ]
    
    def detect_statement_type(self, text: str) -> Tuple[StatementType, float]:
        """
        Detect the type of financial statement from text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (StatementType, confidence_score)
        """
        text = text.lower()
        
        # Count occurrences of keywords for each statement type
        bs_score = sum(1 for kw in self.balance_sheet_keywords if kw in text)
        is_score = sum(1 for kw in self.income_statement_keywords if kw in text)
        cf_score = sum(1 for kw in self.cash_flow_keywords if kw in text)
        
        # Get the maximum score
        max_score = max(bs_score, is_score, cf_score)
        
        # Normalize scores
        bs_confidence = bs_score / max(len(self.balance_sheet_keywords), 1) if bs_score > 0 else 0
        is_confidence = is_score / max(len(self.income_statement_keywords), 1) if is_score > 0 else 0
        cf_confidence = cf_score / max(len(self.cash_flow_keywords), 1) if cf_score > 0 else 0
        
        # Determine statement type based on highest score
        if max_score == 0:
            return StatementType.UNKNOWN, 0.0
        elif bs_score >= is_score and bs_score >= cf_score:
            return StatementType.BALANCE_SHEET, bs_confidence
        elif is_score >= bs_score and is_score >= cf_score:
            return StatementType.INCOME_STATEMENT, is_confidence
        else:
            return StatementType.CASH_FLOW, cf_confidence


class ParserRegistry:
    """
    Registry for financial document parsers.
    
    This class maintains a registry of available parsers and provides methods
    for selecting the appropriate parser for a given document.
    """
    
    def __init__(self):
        self.parsers: List[FinancialDocumentParser] = []
    
    def register_parser(self, parser: FinancialDocumentParser) -> None:
        """
        Register a parser in the registry.
        
        Args:
            parser: Parser to register
        """
        self.parsers.append(parser)
        logger.info(f"Registered parser: {parser.__class__.__name__}")
    
    def get_parser_for_document(self, document_path: Union[str, Path]) -> Optional[FinancialDocumentParser]:
        """
        Get the appropriate parser for a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            The first parser that can handle the document, or None if no suitable parser is found
        """
        for parser in self.parsers:
            if parser.can_parse(document_path):
                logger.info(f"Selected parser {parser.__class__.__name__} for {document_path}")
                return parser
        
        logger.warning(f"No suitable parser found for {document_path}")
        return None
    
    def parse_document(self, document_path: Union[str, Path], **kwargs) -> ParsingResult:
        """
        Parse a document using the appropriate parser.
        
        Args:
            document_path: Path to the document
            **kwargs: Additional parser-specific parameters
            
        Returns:
            ParsingResult from the selected parser, or an empty result if no parser is found
        """
        parser = self.get_parser_for_document(document_path)
        
        if parser:
            return parser.parse(document_path, **kwargs)
        else:
            result = ParsingResult()
            result.add_error(f"No suitable parser found for {document_path}")
            return result


# Singleton instance of the parser registry
parser_registry = ParserRegistry()
