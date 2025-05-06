"""
Financial-specific document chunking strategies.

This module provides specialized chunkers that understand financial document structure
and preserve financial semantics during the chunking process.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Pattern
from enum import Enum, auto

from ...config.config import settings

from ..base_processor import DocumentChunk, Document
from .base_chunker import BaseChunker, TextChunker

logger = logging.getLogger(__name__)


class FinancialStatementType(Enum):
    """Types of financial statements for classification."""
    INCOME_STATEMENT = auto()
    BALANCE_SHEET = auto()
    CASH_FLOW = auto()
    STATEMENT_OF_EQUITY = auto()
    NOTES = auto()
    MANAGEMENT_DISCUSSION = auto()
    RISK_FACTORS = auto()
    UNKNOWN = auto()


class FinancialChunkerBase(TextChunker):
    """Base class for all financial document chunkers."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 100,
                 respect_financial_boundaries: bool = True):
        """
        Initialize the financial chunker.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            respect_financial_boundaries: Whether to respect financial statement boundaries
        """
        super().__init__(chunk_size, chunk_overlap, respect_semantics=True)
        self.respect_financial_boundaries = respect_financial_boundaries
        
        # Financial statement type detection patterns
        self._init_financial_patterns()
    
    def _init_financial_patterns(self):
        """Initialize regex patterns for financial statement detection."""
        self.statement_patterns = {
            FinancialStatementType.INCOME_STATEMENT: re.compile(
                r"(?i)(consolidated\s+)?(statements?\s+of\s+|)(income|profit\s+and\s+loss|operations|earnings)",
                re.IGNORECASE
            ),
            FinancialStatementType.BALANCE_SHEET: re.compile(
                r"(?i)(consolidated\s+)?(statements?\s+of\s+|)(financial\s+position|balance\s+sheets?)",
                re.IGNORECASE
            ),
            FinancialStatementType.CASH_FLOW: re.compile(
                r"(?i)(consolidated\s+)?(statements?\s+of\s+|)cash\s+flows?",
                re.IGNORECASE
            ),
            FinancialStatementType.STATEMENT_OF_EQUITY: re.compile(
                r"(?i)(consolidated\s+)?(statements?\s+of\s+|)(stockholders'|shareholders'|equity|changes\s+in\s+equity)",
                re.IGNORECASE
            ),
            FinancialStatementType.NOTES: re.compile(
                r"(?i)notes\s+to\s+(consolidated\s+|)financial\s+statements",
                re.IGNORECASE
            ),
            FinancialStatementType.MANAGEMENT_DISCUSSION: re.compile(
                r"(?i)(item\s+7\.?\s*)?management'?s?\s+discussion\s+and\s+analysis",
                re.IGNORECASE
            ),
            FinancialStatementType.RISK_FACTORS: re.compile(
                r"(?i)(item\s+1A\.?\s*)?risk\s+factors",
                re.IGNORECASE
            )
        }
        
        # Financial table detection
        self.table_pattern = re.compile(r"(?:\s*\d+[\s,]*)(?:\s*\d+[\s,]*)(?:\s*\d+[\s,]*)")
        
        # Financial period patterns
        self.period_patterns = [
            re.compile(r"(?i)(?:year(?:s)?|quarter(?:s)?)?\s*ended\s+(?:(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}|(?:\d{1,2}/\d{1,2}/\d{2,4}))"),
            re.compile(r"(?i)(?:fiscal\s+years?|quarters?)\s+\d{4}(?:\s*-\s*\d{4})?"),
            re.compile(r"(?i)(?:q[1-4]|first|second|third|fourth)\s+quarter(?:\s+of|\s+\d{4})")
        ]
    
    def detect_statement_type(self, text: str) -> FinancialStatementType:
        """
        Detect the financial statement type from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            The detected statement type or UNKNOWN
        """
        for stmt_type, pattern in self.statement_patterns.items():
            if pattern.search(text):
                return stmt_type
        
        # Check for tables with numeric data which may be financial statements
        if self.table_pattern.search(text):
            # We can't determine exactly which statement type, but it's financial
            return FinancialStatementType.UNKNOWN
            
        return FinancialStatementType.UNKNOWN
    
    def detect_financial_period(self, text: str) -> Optional[str]:
        """
        Extract the financial period information from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Financial period string or None if not found
        """
        for pattern in self.period_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None
    
    def has_table(self, text: str) -> bool:
        """
        Detect if the text contains a financial table.
        
        Args:
            text: The text to analyze
            
        Returns:
            True if text contains a table
        """
        return bool(self.table_pattern.search(text))
    
    def extract_financial_entities(self, text: str) -> List[str]:
        """
        Extract financial entities from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of financial entities found
        """
        # Simple implementation - in production this would use NER/ML
        entities = []
        
        # Common financial terms
        financial_terms = [
            "revenue", "income", "profit", "loss", "ebitda", "assets", 
            "liabilities", "equity", "cash flow", "balance", "earnings",
            "eps", "dividend", "depreciation", "amortization", "tax",
            "interest", "debt", "capital", "expense", "margin"
        ]
        
        # Search for terms
        for term in financial_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                entities.append(term)
                
        return entities
    
    def create_financial_chunk(self, 
                              content: str, 
                              metadata: Dict[str, Any], 
                              doc_id: str, 
                              chunk_index: int) -> DocumentChunk:
        """
        Create a document chunk with financial metadata.
        
        Args:
            content: The chunk content
            metadata: The document metadata to extend
            doc_id: Document identifier
            chunk_index: Index of this chunk within the document
            
        Returns:
            A DocumentChunk instance with financial metadata
        """
        # Get the basic chunk
        chunk = super().create_chunk(content, metadata, doc_id, chunk_index)
        
        # Add financial-specific metadata
        statement_type = self.detect_statement_type(content)
        financial_period = self.detect_financial_period(content)
        has_table = self.has_table(content)
        financial_entities = self.extract_financial_entities(content)
        
        # Update metadata
        chunk.metadata.update({
            "statement_type": statement_type.name.lower(),
            "contains_table": has_table,
            "financial_entities": financial_entities
        })
        
        if financial_period:
            chunk.metadata["financial_period"] = financial_period
            
        return chunk


class FinancialStatementChunker(FinancialChunkerBase):
    """
    Chunker specialized for financial statements that preserves table structures
    and statement boundaries.
    """
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a financial document respecting statement boundaries.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        metadata = document.metadata
        doc_id = document.doc_id
        
        # First identify statement sections
        sections = self._identify_sections(content)
        
        chunks = []
        chunk_index = 0
        
        for section_type, section_text in sections:
            # Set the statement type in metadata
            section_metadata = metadata.copy() if metadata else {}
            section_metadata["statement_type"] = section_type.name.lower()
            
            # For tables, keep them whole if possible
            if self.has_table(section_text):
                if len(section_text) <= self.chunk_size * 2:  # Allow tables to be bigger
                    chunks.append(self.create_financial_chunk(
                        section_text, section_metadata, doc_id, chunk_index
                    ))
                    chunk_index += 1
                    continue
            
            # For longer sections, chunk while respecting table rows
            if len(section_text) > self.chunk_size:
                section_chunks = self._chunk_section(section_text, section_type)
                
                for sc in section_chunks:
                    chunks.append(self.create_financial_chunk(
                        sc, section_metadata, doc_id, chunk_index
                    ))
                    chunk_index += 1
            else:
                # For short sections, keep them whole
                chunks.append(self.create_financial_chunk(
                    section_text, section_metadata, doc_id, chunk_index
                ))
                chunk_index += 1
                
        return chunks
    
    def _identify_sections(self, content: str) -> List[Tuple[FinancialStatementType, str]]:
        """
        Identify different statement sections in the document.
        
        Args:
            content: Document content
            
        Returns:
            List of (statement_type, section_text) tuples
        """
        # Split content on potential statement boundaries
        boundaries = []
        
        for stmt_type, pattern in self.statement_patterns.items():
            for match in pattern.finditer(content):
                boundaries.append((match.start(), stmt_type))
        
        # Sort boundaries by position
        boundaries.sort()
        
        # If no boundaries found, treat as a single section
        if not boundaries:
            stmt_type = self.detect_statement_type(content[:1000])  # Check the beginning
            return [(stmt_type, content)]
        
        # Create sections from boundaries
        sections = []
        current_pos = 0
        
        for pos, stmt_type in boundaries:
            # If there's text before this boundary
            if pos > current_pos:
                # Determine type of previous section
                prev_text = content[current_pos:pos]
                prev_type = self.detect_statement_type(prev_text)
                sections.append((prev_type, prev_text))
            
            # Find the end of this section
            next_pos = len(content)
            for npos, _ in boundaries:
                if npos > pos:
                    next_pos = npos
                    break
                    
            # Add this section
            section_text = content[pos:next_pos]
            sections.append((stmt_type, section_text))
            current_pos = next_pos
        
        # Add final section if needed
        if current_pos < len(content):
            final_text = content[current_pos:]
            final_type = self.detect_statement_type(final_text)
            sections.append((final_type, final_text))
            
        return sections
    
    def _chunk_section(self, section_text: str, section_type: FinancialStatementType) -> List[str]:
        """
        Chunk a section of text while respecting table structures.
        
        Args:
            section_text: The section text to chunk
            section_type: The type of financial statement
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # For tables, try to split on row boundaries
        if section_type in [FinancialStatementType.INCOME_STATEMENT, 
                           FinancialStatementType.BALANCE_SHEET,
                           FinancialStatementType.CASH_FLOW,
                           FinancialStatementType.STATEMENT_OF_EQUITY]:
            
            # Split on newlines to get rows
            rows = section_text.split('\n')
            
            current_chunk = ""
            for row in rows:
                # If adding this row would exceed chunk size and we have content
                if len(current_chunk) + len(row) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = row + '\n'
                else:
                    current_chunk += row + '\n'
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk)
                
        else:
            # For narrative sections, use paragraph boundaries
            paragraphs = re.split(r'(\n\s*\n)', section_text)
            
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk += para
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks


class MDAndARiskChunker(FinancialChunkerBase):
    """
    Specialized chunker for Management Discussion & Analysis and Risk Factors sections.
    Preserves logical discussion units while maintaining proper context.
    """
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a MD&A or Risk Factors section with semantic awareness.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        metadata = document.metadata
        doc_id = document.doc_id
        
        # Identify subsections based on headers
        subsections = self._identify_subsections(content)
        
        chunks = []
        chunk_index = 0
        
        for header, subsection_text in subsections:
            # Set subsection in metadata
            subsection_metadata = metadata.copy() if metadata else {}
            if header:
                subsection_metadata["subsection"] = header
            
            # Determine the statement type
            statement_type = self.detect_statement_type(subsection_text[:1000])
            subsection_metadata["statement_type"] = statement_type.name.lower()
            
            # For short subsections, keep them whole
            if len(subsection_text) <= self.chunk_size:
                chunks.append(self.create_financial_chunk(
                    subsection_text, subsection_metadata, doc_id, chunk_index
                ))
                chunk_index += 1
                continue
            
            # For longer subsections, chunk by paragraphs
            paragraphs = re.split(r'(\n\s*\n)', subsection_text)
            
            current_chunk = ""
            # If we have a header, start with it for context
            if header:
                current_chunk = header + "\n\n"
                
            for para in paragraphs:
                # Skip empty paragraphs
                if not para.strip():
                    continue
                    
                # If adding this paragraph would exceed the chunk size
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    chunks.append(self.create_financial_chunk(
                        current_chunk, subsection_metadata, doc_id, chunk_index
                    ))
                    chunk_index += 1
                    
                    # Reset chunk, potentially with header context
                    if header and self.chunk_overlap < len(header) + 10:
                        current_chunk = header + " (continued):\n\n" + para
                    else:
                        current_chunk = para
                else:
                    current_chunk += para
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(self.create_financial_chunk(
                    current_chunk, subsection_metadata, doc_id, chunk_index
                ))
                chunk_index += 1
                
        return chunks
    
    def _identify_subsections(self, content: str) -> List[Tuple[str, str]]:
        """
        Identify subsections based on headers.
        
        Args:
            content: Document content
            
        Returns:
            List of (header, subsection_text) tuples
        """
        # Look for headers (capitalized lines, numbered sections, etc.)
        header_pattern = re.compile(
            r'(?:^|\n)(?:'
            r'(?:[A-Z][A-Z\s]+[A-Z]\.?)|'  # ALL CAPS HEADERS
            r'(?:\d+\.\s+[A-Z][a-zA-Z\s]+)|'  # Numbered headers
            r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+:)'  # Title Case Headers with colon
            r')'
        )
        
        # Find all potential headers
        matches = list(header_pattern.finditer(content))
        
        # If no headers found, return the whole content
        if not matches:
            return [("", content)]
        
        subsections = []
        current_pos = 0
        
        for match in matches:
            header_start = match.start()
            header_end = match.end()
            header = content[header_start:header_end].strip()
            
            # If there's content before this header
            if header_start > current_pos:
                previous_text = content[current_pos:header_start]
                previous_header = subsections[-1][0] if subsections else ""
                subsections.append((previous_header + " (continued)", previous_text))
            
            # Find the end of this subsection
            next_pos = len(content)
            for next_match in matches:
                if next_match.start() > header_end:
                    next_pos = next_match.start()
                    break
            
            # Add this subsection
            subsection_text = content[header_start:next_pos]
            subsections.append((header, subsection_text))
            current_pos = next_pos
        
        # Add final subsection if needed
        if current_pos < len(content):
            final_text = content[current_pos:]
            final_header = subsections[-1][0] + " (continued)" if subsections else ""
            subsections.append((final_header, final_text))
        
        return subsections


class FinancialNotesChunker(FinancialChunkerBase):
    """
    Specialized chunker for financial statement notes.
    Preserves note boundaries and ensures each note stays coherent.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 100,
                 respect_financial_boundaries: bool = True):
        """Initialize the notes chunker."""
        super().__init__(chunk_size, chunk_overlap, respect_financial_boundaries)
        
        # Pattern to identify note numbers
        self.note_pattern = re.compile(
            r'(?:^|\n)(?:NOTE|Note)\s+(\d+)[\.:]?\s+([A-Z][A-Za-z\s]+)',
            re.MULTILINE
        )
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk financial notes while preserving note boundaries.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        metadata = document.metadata
        doc_id = document.doc_id
        
        # Identify individual notes
        notes = self._identify_notes(content)
        
        chunks = []
        chunk_index = 0
        
        for note_num, note_title, note_text in notes:
            # Set note metadata
            note_metadata = metadata.copy() if metadata else {}
            note_metadata["statement_type"] = FinancialStatementType.NOTES.name.lower()
            note_metadata["note_number"] = note_num
            note_metadata["note_title"] = note_title
            
            # For short notes, keep them whole
            if len(note_text) <= self.chunk_size:
                chunks.append(self.create_financial_chunk(
                    note_text, note_metadata, doc_id, chunk_index
                ))
                chunk_index += 1
                continue
            
            # For longer notes, chunk by paragraphs but keep the note title
            paragraphs = re.split(r'(\n\s*\n)', note_text)
            
            current_chunk = ""
            # Always start with the note title for context
            note_header = f"NOTE {note_num}: {note_title}"
            current_chunk = note_header + "\n\n"
            
            for para in paragraphs:
                # Skip empty paragraphs
                if not para.strip():
                    continue
                
                # If adding this paragraph would exceed the chunk size
                if len(current_chunk) + len(para) > self.chunk_size and len(current_chunk) > len(note_header) + 10:
                    chunks.append(self.create_financial_chunk(
                        current_chunk, note_metadata, doc_id, chunk_index
                    ))
                    chunk_index += 1
                    
                    # Reset chunk with note title for context
                    current_chunk = f"{note_header} (continued):\n\n{para}"
                else:
                    current_chunk += para
            
            # Add the last chunk
            if current_chunk.strip() and len(current_chunk) > len(note_header) + 10:
                chunks.append(self.create_financial_chunk(
                    current_chunk, note_metadata, doc_id, chunk_index
                ))
                chunk_index += 1
                
        return chunks
    
    def _identify_notes(self, content: str) -> List[Tuple[str, str, str]]:
        """
        Identify individual notes in the financial statements.
        
        Args:
            content: Document content
            
        Returns:
            List of (note_number, note_title, note_text) tuples
        """
        # Find all note markers
        matches = list(self.note_pattern.finditer(content))
        
        # If no notes found, process as a single section
        if not matches:
            return [("0", "General", content)]
        
        notes = []
        current_pos = 0
        
        for i, match in enumerate(matches):
            note_num = match.group(1)
            note_title = match.group(2).strip()
            note_start = match.start()
            
            # If there's content before the first note
            if i == 0 and note_start > 0:
                intro_text = content[:note_start]
                if intro_text.strip():
                    notes.append(("0", "Introduction to Notes", intro_text))
            
            # Find the end of this note
            if i < len(matches) - 1:
                note_end = matches[i+1].start()
            else:
                note_end = len(content)
            
            # Add this note
            note_text = content[note_start:note_end]
            notes.append((note_num, note_title, note_text))
            current_pos = note_end
        
        return notes


class ChunkerFactory:
    """Factory for creating appropriate financial chunkers."""
    
    @staticmethod
    def get_chunker(document_type: str, statement_type: Optional[str] = None, **kwargs) -> BaseChunker:
        """
        Get a financial chunker appropriate for the document and statement type.
        
        Args:
            document_type: Type of document (e.g., "financial", "10-k", "annual_report")
            statement_type: Type of financial statement if known
            **kwargs: Additional parameters to pass to the chunker
            
        Returns:
            An instance of a financial chunker
        """
        # Default chunker parameters
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 100)
        respect_boundaries = kwargs.get('respect_financial_boundaries', True)
        
        # Determine chunker based on statement type
        if statement_type:
            statement_type = statement_type.lower()
            
            if statement_type in ["income_statement", "balance_sheet", "cash_flow", "statement_of_equity"]:
                return FinancialStatementChunker(chunk_size, chunk_overlap, respect_boundaries)
                
            elif statement_type in ["management_discussion", "md&a", "risk_factors", "risk"]:
                return MDAndARiskChunker(chunk_size, chunk_overlap, respect_boundaries)
                
            elif statement_type in ["notes", "financial_notes"]:
                return FinancialNotesChunker(chunk_size, chunk_overlap, respect_boundaries)
        
        # If statement type not specified, determine based on document type
        if document_type.lower() in ["10-k", "10-q", "annual_report", "quarterly_report"]:
            # For full reports, default to MD&A chunker as it's more general
            return MDAndARiskChunker(chunk_size, chunk_overlap, respect_boundaries)
            
        # Default to the financial statement chunker
        return FinancialStatementChunker(chunk_size, chunk_overlap, respect_boundaries)
