"""
PDF Financial Statement Parser

This module provides specialized parsing capabilities for extracting financial statement
data from PDF documents. It handles the complexities of table extraction, layout analysis,
and financial data recognition in PDF financial statements.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import date, datetime

import pypdf
import tabula
import pandas as pd
import numpy as np
from unstructured.partition.pdf import partition_pdf

from .base_parser import (
    FinancialDocumentParser,
    ParsingResult,
    StatementDetector
)
from ..models.statement import (
    FinancialStatement,
    StatementType,
    StatementMetadata,
    TimePeriod,
    TimePeriodType,
    AccountingStandard,
    LineItem,
    create_financial_statement
)

logger = logging.getLogger(__name__)


class FinancialPeriodExtractor:
    """
    Extracts time period information from financial statements.
    
    This class analyzes text to identify reporting periods, fiscal years,
    and date ranges mentioned in financial statements.
    """
    
    def __init__(self):
        # Regular expressions for date detection
        self.year_pattern = r'(?:fiscal|year|fy|FY|Fiscal Year)?\s*(?:ended|ending)?\s*(?:in|on|,)?\s*(\d{4})'
        self.month_day_year_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        self.quarter_pattern = r'(?:Q|q)(\d)\s*(?:FY|fiscal year|fy|year)?[\s,]*(\d{4})'
        self.period_pattern = r'(?:three|six|nine|twelve|3|6|9|12)\s*months?\s*(?:ended|ending)\s*(?:in|on)?\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})'
    
    def extract_periods(self, text: str) -> List[TimePeriod]:
        """
        Extract time periods from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected time periods
        """
        periods = []
        
        # Try to extract fiscal years
        fiscal_years = re.findall(self.year_pattern, text)
        for year_str in fiscal_years:
            try:
                year = int(year_str)
                if 1900 <= year <= 2100:  # Sanity check
                    periods.append(TimePeriod(
                        period_type=TimePeriodType.ANNUAL,
                        fiscal_year=year
                    ))
            except ValueError:
                continue
        
        # Try to extract quarters
        quarters = re.findall(self.quarter_pattern, text)
        for quarter, year_str in quarters:
            try:
                quarter = int(quarter)
                year = int(year_str)
                if 1 <= quarter <= 4 and 1900 <= year <= 2100:
                    periods.append(TimePeriod(
                        period_type=TimePeriodType.QUARTERLY,
                        fiscal_year=year,
                        fiscal_quarter=quarter
                    ))
            except ValueError:
                continue
        
        # Try to extract month periods
        month_periods = re.findall(self.period_pattern, text)
        for date_str in month_periods:
            try:
                end_date = datetime.strptime(date_str, "%B %d, %Y").date()
                # Infer period type from context
                if "three months" in text or "3 months" in text:
                    period_type = TimePeriodType.QUARTERLY
                elif "six months" in text or "6 months" in text:
                    period_type = TimePeriodType.SEMI_ANNUAL
                elif "twelve months" in text or "12 months" in text:
                    period_type = TimePeriodType.ANNUAL
                else:
                    period_type = TimePeriodType.CUSTOM
                
                periods.append(TimePeriod(
                    period_type=period_type,
                    end_date=end_date
                ))
            except ValueError:
                continue
        
        return periods


class CompanyExtractor:
    """
    Extracts company information from financial statements.
    
    This class identifies company names, stock tickers, and related information
    from financial statement text.
    """
    
    def __init__(self):
        # Patterns for company name extraction
        self.company_patterns = [
            r'((?:[A-Z][a-z]*\s*)+(?:Inc\.|Corporation|Corp\.|Company|Co\.|Ltd\.|Limited|LLC|LLP|LP|plc|PLC|Group|Holdings|Holding|N\.V\.|S\.A\.|SE|NV|SA))',
            r'((?:[A-Z][a-z]*\s*)+)(?:\((?:NASDAQ|NYSE|OTCBB|OTCQB|OTCQX|LSE|TSX|CSE|HKEX|SSE|SZSE):?\s*([A-Z]+(?:\.[A-Z])?)\))'
        ]
        
    def extract_company_info(self, text: str) -> Dict[str, Any]:
        """
        Extract company information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with company name and ticker if found
        """
        company_info = {"name": None, "ticker": None}
        
        # Look for company name with ticker
        for pattern in self.company_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if len(matches[0]) == 2 and matches[0][1]:  # This means we matched the pattern with a ticker
                    company_info["name"] = matches[0][0].strip()
                    company_info["ticker"] = matches[0][1].strip()
                    break
                else:
                    company_info["name"] = matches[0].strip() if isinstance(matches[0], str) else matches[0][0].strip()
        
        return company_info


class PDFTableExtractor:
    """
    Extracts tables from PDF documents.
    
    This class handles the extraction of tabular data from PDF files,
    with specialized handling for financial tables.
    """
    
    def __init__(self):
        self.default_options = {
            "pages": "all",
            "lattice": True,  # Use lattice mode for structured tables
            "stream": True,   # Also try stream mode if lattice fails
            "guess": False,   # Don't guess table structure
            "multiple_tables": True
        }
    
    def extract_tables(self, pdf_path: Union[str, Path], **kwargs) -> List[pd.DataFrame]:
        """
        Extract tables from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            **kwargs: Additional options for tabula
            
        Returns:
            List of DataFrames containing table data
        """
        options = {**self.default_options, **kwargs}
        tables = []
        
        try:
            # First try lattice mode
            lattice_tables = tabula.read_pdf(
                str(pdf_path),
                pages=options["pages"],
                lattice=True,
                multiple_tables=options["multiple_tables"]
            )
            tables.extend(lattice_tables)
            
            # If no tables found with lattice, try stream mode
            if len(tables) == 0 and options["stream"]:
                stream_tables = tabula.read_pdf(
                    str(pdf_path),
                    pages=options["pages"],
                    stream=True,
                    multiple_tables=options["multiple_tables"]
                )
                tables.extend(stream_tables)
            
            # Filter out empty or invalid tables
            valid_tables = []
            for table in tables:
                if isinstance(table, pd.DataFrame) and not table.empty and table.shape[0] > 1 and table.shape[1] > 1:
                    valid_tables.append(table)
            
            return valid_tables
            
        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {e}")
            return []
    
    def is_financial_table(self, table: pd.DataFrame) -> bool:
        """
        Check if a table looks like a financial statement table.
        
        Args:
            table: DataFrame to check
            
        Returns:
            True if the table appears to be a financial statement, False otherwise
        """
        # Convert all values to strings for easier analysis
        table_str = table.astype(str)
        
        # Check for financial keywords in the table
        financial_keywords = [
            "assets", "liabilities", "equity", "revenue", "income", "expense",
            "cash", "profit", "loss", "total", "net", "gross", "operating",
            "current", "non-current", "long-term", "short-term"
        ]
        
        keyword_count = 0
        for keyword in financial_keywords:
            if any(table_str.stack().str.lower().str.contains(keyword)):
                keyword_count += 1
        
        # Check for monetary values ($ or numbers with commas or decimals)
        money_pattern = r'(\$\s*[\d,]+\.?\d*|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)'
        has_monetary_values = False
        
        for col in table.columns:
            col_values = table[col].astype(str)
            if any(col_values.str.match(money_pattern)):
                has_monetary_values = True
                break
        
        # Consider it a financial table if it has enough keywords and monetary values
        return keyword_count >= 3 and has_monetary_values


class PDFFinancialStatementParser(FinancialDocumentParser):
    """
    Parser for financial statements in PDF format.
    
    This class extracts structured financial data from PDF financial statements,
    including balance sheets, income statements, and cash flow statements.
    """
    
    def __init__(self):
        self.statement_detector = StatementDetector()
        self.period_extractor = FinancialPeriodExtractor()
        self.company_extractor = CompanyExtractor()
        self.table_extractor = PDFTableExtractor()
        
        # Extensions that this parser can handle
        self.supported_extensions = {".pdf"}
    
    def can_parse(self, document_path: Union[str, Path]) -> bool:
        """
        Check if this parser can handle the given document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            True if this parser can handle the document, False otherwise
        """
        ext = self._get_file_extension(document_path)
        return ext in self.supported_extensions and os.path.exists(document_path)
    
    def parse(self, document_path: Union[str, Path], **kwargs) -> ParsingResult:
        """
        Parse a PDF financial document and extract structured data.
        
        Args:
            document_path: Path to the PDF document
            **kwargs: Additional parser-specific parameters
            
        Returns:
            ParsingResult containing extracted financial statements and metadata
        """
        document_path = Path(document_path)
        result = ParsingResult()
        
        if not self.can_parse(document_path):
            result.add_error(f"Cannot parse {document_path} with PDFFinancialStatementParser")
            return result
        
        try:
            # Extract text and metadata from PDF
            text, metadata = self._extract_pdf_content(document_path)
            result.metadata = metadata
            
            # Detect company information
            company_info = self.company_extractor.extract_company_info(text[:5000])  # Check first few pages
            company_name = company_info.get("name", "Unknown Company")
            
            # Extract time periods
            periods = self.period_extractor.extract_periods(text[:5000])
            if not periods:
                # If no specific periods detected, create a default one
                periods = [TimePeriod(
                    period_type=TimePeriodType.ANNUAL,
                    label="Unknown Period"
                )]
            
            # Extract tables from the PDF
            tables = self.table_extractor.extract_tables(document_path)
            financial_tables = [table for table in tables if self.table_extractor.is_financial_table(table)]
            
            if not financial_tables:
                result.add_warning(f"No financial tables detected in {document_path}")
            
            # Process each financial table
            for i, table in enumerate(financial_tables):
                # Get table context (surrounding text)
                table_context = self._get_table_context(text, i, tables)
                
                # Detect statement type from context
                statement_type, confidence = self.statement_detector.detect_statement_type(table_context)
                
                if statement_type != StatementType.UNKNOWN:
                    # Create metadata for the statement
                    statement_metadata = StatementMetadata(
                        company_name=company_name,
                        statement_type=statement_type,
                        period=periods[0] if periods else TimePeriod(period_type=TimePeriodType.ANNUAL, label="Unknown Period"),
                        accounting_standard=AccountingStandard.UNKNOWN,
                        source_document=str(document_path),
                        confidence_score=confidence
                    )
                    
                    # Parse table into line items
                    line_items = self._parse_table_to_line_items(table)
                    
                    # Create appropriate statement type
                    statement = create_financial_statement(
                        statement_type=statement_type,
                        metadata=statement_metadata,
                        line_items=line_items
                    )
                    
                    result.add_statement(statement)
            
            # Set overall confidence score
            if result.statements:
                result.confidence_score = sum(s.metadata.confidence_score for s in result.statements) / len(result.statements)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error parsing {document_path}: {e}")
            result.add_error(f"Error parsing {document_path}: {str(e)}")
            return result
    
    def _extract_pdf_content(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (text_content, metadata_dict)
        """
        metadata = {}
        text = ""
        
        try:
            # Extract metadata using PyPDF
            with open(pdf_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                
                if pdf.metadata:
                    # Extract standard PDF metadata
                    if pdf.metadata.get("/Title"):
                        metadata["title"] = pdf.metadata.get("/Title")
                    if pdf.metadata.get("/Author"):
                        metadata["author"] = pdf.metadata.get("/Author")
                    if pdf.metadata.get("/CreationDate"):
                        metadata["creation_date"] = pdf.metadata.get("/CreationDate")
                
                # Extract text from all pages
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
                
                # Add basic PDF info
                metadata["page_count"] = len(pdf.pages)
        
        except Exception as e:
            logger.warning(f"Error extracting content from PDF {pdf_path}: {e}")
            
            # Try alternate extraction method if primary fails
            try:
                elements = partition_pdf(str(pdf_path))
                text = "\n".join(str(element) for element in elements)
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {pdf_path}: {e2}")
                text = ""
        
        return text, metadata
    
    def _get_table_context(self, text: str, table_index: int, tables: List[pd.DataFrame]) -> str:
        """
        Extract the text context surrounding a table.
        
        Args:
            text: Full document text
            table_index: Index of the current table
            tables: List of all tables
            
        Returns:
            Text context around the table
        """
        # This is an approximation since we don't have precise table positions
        # In a real implementation, we would use PDF layout analysis to get exact positions
        
        # For simplicity, we'll just split the text into chunks based on table count
        if not tables:
            return text
        
        chunks = text.split('\n\n')
        chunk_per_table = max(1, len(chunks) // len(tables))
        
        start_idx = max(0, table_index * chunk_per_table - 2)
        end_idx = min(len(chunks), (table_index + 1) * chunk_per_table + 2)
        
        context = '\n\n'.join(chunks[start_idx:end_idx])
        return context
    
    def _parse_table_to_line_items(self, table: pd.DataFrame) -> Dict[str, LineItem]:
        """
        Parse a DataFrame table into financial statement line items.
        
        Args:
            table: DataFrame containing table data
            
        Returns:
            Dictionary of line items extracted from the table
        """
        line_items = {}
        
        # Clean column names
        table.columns = [str(col).strip() for col in table.columns]
        
        # Identify label column (usually first column)
        label_col = table.columns[0]
        
        # Process each row to create line items
        for idx, row in table.iterrows():
            # Skip rows with no label
            label = row[label_col]
            if not isinstance(label, str) or not label.strip():
                continue
            
            label = label.strip()
            
            # Look for value columns (typically contain numeric values or - for zero)
            for col in table.columns[1:]:
                # Skip non-numeric and empty values
                value_str = str(row[col]).strip()
                if not value_str or value_str == '-' or value_str.lower() == 'nan':
                    continue
                
                # Remove currency symbols, commas, and handle parentheses for negative values
                value_str = value_str.replace('$', '').replace(',', '')
                # Handle negative values in parentheses
                if value_str.startswith('(') and value_str.endswith(')'):
                    value_str = '-' + value_str[1:-1]
                
                try:
                    value = float(value_str)
                    
                    # Create a unique name using label and column
                    item_name = f"{label} ({col})" if col != table.columns[1] else label
                    
                    # Create line item
                    line_items[item_name] = LineItem(
                        name=item_name,
                        value=value,
                        raw_value=value_str,
                        unit="USD",  # Default to USD, should be detected in actual implementation
                        scaling_factor=1.0,  # Should be detected based on statement terminology
                        is_calculated=False
                    )
                except ValueError:
                    # Skip values that can't be converted to float
                    continue
        
        return line_items
