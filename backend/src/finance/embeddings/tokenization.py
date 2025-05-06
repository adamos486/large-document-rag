"""
Financial-specific tokenization for text preprocessing.

This module provides specialized tokenization capabilities for financial text,
with enhanced handling of financial terms, numerical values, and document structures.
"""

import re
import logging
from typing import List, Dict, Optional, Union, Any, Set
from pathlib import Path
import json

from src.config.config import settings
from .config import EmbeddingModelConfig

logger = logging.getLogger(__name__)


class FinancialTokenizer:
    """
    Specialized tokenizer for financial text preprocessing.
    
    This tokenizer enhances standard tokenization with:
    1. Special handling for financial terms and entities
    2. Numeric value normalization
    3. Table structure preservation
    4. Financial statement section marking
    """
    
    def __init__(self, config: Optional[EmbeddingModelConfig] = None):
        """
        Initialize the financial tokenizer.
        
        Args:
            config: Configuration for the tokenizer
        """
        self.config = config or EmbeddingModelConfig()
        
        # Load financial terms for specialized handling
        self.financial_terms_path = self.config.financial_terms_path
        self._load_financial_terms()
        
        # Load statement structure for section recognition
        self.statement_structure_path = self.config.statement_structure_path
        self._load_statement_structure()
        
        # Compile regex patterns for preprocessing
        self._compile_patterns()
    
    def _load_financial_terms(self):
        """Load financial terms from file."""
        if self.financial_terms_path.exists():
            try:
                with open(self.financial_terms_path, 'r') as f:
                    self.financial_terms = json.load(f)
                
                # Create sets for faster lookup
                self.metrics = set(self.financial_terms.get("metrics", []))
                self.ratios = set(self.financial_terms.get("ratios", []))
                self.statements = set(self.financial_terms.get("statements", []))
                self.regulations = set(self.financial_terms.get("regulations", []))
                
                # Create a combined set of all terms
                self.all_terms = set()
                for category, terms in self.financial_terms.items():
                    self.all_terms.update(terms)
                
                logger.info(f"Loaded {len(self.all_terms)} financial terms")
            except Exception as e:
                logger.warning(f"Error loading financial terms: {e}")
                self._initialize_default_terms()
        else:
            logger.warning(f"Financial terms file not found at {self.financial_terms_path}")
            self._initialize_default_terms()
    
    def _initialize_default_terms(self):
        """Initialize default financial terms."""
        self.financial_terms = {
            "metrics": [
                "Revenue", "Net Income", "EBITDA", "Gross Profit", "Operating Income",
                "Total Assets", "Total Liabilities", "Shareholders' Equity"
            ],
            "ratios": [
                "P/E Ratio", "EPS", "ROI", "ROE", "ROA", "Current Ratio", "Quick Ratio",
                "Debt-to-Equity", "Gross Margin", "Operating Margin", "Net Profit Margin"
            ],
            "statements": [
                "Balance Sheet", "Income Statement", "Cash Flow Statement"
            ],
            "regulations": [
                "GAAP", "IFRS", "SOX", "FASB", "SEC"
            ]
        }
        
        # Create sets for faster lookup
        self.metrics = set(self.financial_terms.get("metrics", []))
        self.ratios = set(self.financial_terms.get("ratios", []))
        self.statements = set(self.financial_terms.get("statements", []))
        self.regulations = set(self.financial_terms.get("regulations", []))
        
        # Create a combined set of all terms
        self.all_terms = set()
        for category, terms in self.financial_terms.items():
            self.all_terms.update(terms)
    
    def _load_statement_structure(self):
        """Load statement structure from file."""
        if self.statement_structure_path.exists():
            try:
                with open(self.statement_structure_path, 'r') as f:
                    self.statement_structure = json.load(f)
                
                # Extract all account names for recognition
                self.all_accounts = set()
                
                # Process balance sheet
                if "balance_sheet" in self.statement_structure:
                    balance_sheet = self.statement_structure["balance_sheet"]
                    
                    # Process assets
                    if "assets" in balance_sheet:
                        assets = balance_sheet["assets"]
                        if "current_assets" in assets:
                            self.all_accounts.update(assets["current_assets"])
                        if "non_current_assets" in assets:
                            self.all_accounts.update(assets["non_current_assets"])
                    
                    # Process liabilities
                    if "liabilities" in balance_sheet:
                        liabilities = balance_sheet["liabilities"]
                        if "current_liabilities" in liabilities:
                            self.all_accounts.update(liabilities["current_liabilities"])
                        if "non_current_liabilities" in liabilities:
                            self.all_accounts.update(liabilities["non_current_liabilities"])
                    
                    # Process equity
                    if "equity" in balance_sheet:
                        self.all_accounts.update(balance_sheet["equity"])
                
                # Process income statement
                if "income_statement" in self.statement_structure:
                    self.all_accounts.update(self.statement_structure["income_statement"])
                
                # Process cash flow statement
                if "cash_flow_statement" in self.statement_structure:
                    cash_flow = self.statement_structure["cash_flow_statement"]
                    
                    for section, accounts in cash_flow.items():
                        self.all_accounts.update(accounts)
                
                # Convert underscores to spaces for better matching
                self.all_accounts_normalized = {account.replace("_", " ") for account in self.all_accounts}
                
                logger.info(f"Loaded {len(self.all_accounts)} financial accounts")
            except Exception as e:
                logger.warning(f"Error loading statement structure: {e}")
                self.statement_structure = {}
                self.all_accounts = set()
                self.all_accounts_normalized = set()
        else:
            logger.warning(f"Statement structure file not found at {self.statement_structure_path}")
            self.statement_structure = {}
            self.all_accounts = set()
            self.all_accounts_normalized = set()
    
    def _compile_patterns(self):
        """Compile regex patterns for preprocessing."""
        # Money pattern (matches various currency formats)
        self.money_pattern = re.compile(
            r'(?:[$€£¥]|\b(?:USD|EUR|GBP|JPY|CAD|AUD|CHF)\b)\s*'
            r'(?:\d{1,3}(?:,\d{3})+\.\d+|\d+\.\d+|\.\d+|\d{1,3}(?:,\d{3})+|\d+)'
            r'(?:\s*(?:million|billion|trillion|m|b|t))?',
            re.IGNORECASE
        )
        
        # Percentage pattern
        self.percentage_pattern = re.compile(r'\d+(?:\.\d+)?%')
        
        # Date pattern (matches various date formats)
        self.date_pattern = re.compile(
            r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}|'
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b',
            re.IGNORECASE
        )
        
        # Table pattern (matches table-like structures)
        self.table_pattern = re.compile(
            r'(?:^\s*[^\n|]+(?:\|[^\n|]+)+\s*$)|'  # | separated tables
            r'(?:^\s*[^\n,]+(?:,[^\n,]+)+\s*$)',   # CSV-like tables
            re.MULTILINE
        )
        
        # Section heading pattern
        self.section_pattern = re.compile(
            r'^\s*(?:\d+\.(?:\d+\.?)*)?\s*([A-Z][A-Za-z\s]+)(?:\:|\.\s|\s\-\s)?\s*$',
            re.MULTILINE
        )
        
        # Financial statement type pattern
        self.statement_type_pattern = re.compile(
            r'\b(?:balance\s+sheet|statement\s+of\s+financial\s+position|'
            r'income\s+statement|profit\s+and\s+loss|statement\s+of\s+operations|'
            r'cash\s+flow\s+statement|statement\s+of\s+cash\s+flows|'
            r'statement\s+of\s+changes\s+in\s+equity|'
            r'statement\s+of\s+comprehensive\s+income)\b',
            re.IGNORECASE
        )
        
        # Financial period pattern
        self.period_pattern = re.compile(
            r'\b(?:year(?:s)?\s+end(?:ed|ing)?|quarter(?:s)?\s+end(?:ed|ing)?|'
            r'month(?:s)?\s+end(?:ed|ing)?|period\s+end(?:ed|ing)?|fiscal\s+year|'
            r'(?:q[1-4]|fy\d{2,4}))\b',
            re.IGNORECASE
        )

        # Numerical value with units pattern
        self.units_pattern = re.compile(
            r'\b\d+(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion|percent|'
            r'k|m|b|t|%))\b',
            re.IGNORECASE
        )
    
    def _normalize_money(self, text: str) -> str:
        """
        Normalize money expressions to a standard format.
        
        Examples:
            $1,234,567.89 -> $1234567.89
            $1.2M -> $1200000
            EUR 5.6 billion -> EUR 5600000000
        """
        def replace_money(match):
            money_str = match.group(0)
            
            # Extract the currency symbol/code
            currency = re.search(r'[$€£¥]|(?:USD|EUR|GBP|JPY|CAD|AUD|CHF)', money_str, re.IGNORECASE)
            currency_symbol = currency.group(0) if currency else ""
            
            # Extract the numeric value
            numeric_part = re.sub(r'[$€£¥]|(?:USD|EUR|GBP|JPY|CAD|AUD|CHF)', '', money_str, flags=re.IGNORECASE)
            numeric_part = numeric_part.strip()
            
            # Remove commas
            numeric_part = numeric_part.replace(',', '')
            
            # Handle million/billion/trillion
            multiplier = 1
            if re.search(r'million|m\b', numeric_part, re.IGNORECASE):
                multiplier = 1_000_000
                numeric_part = re.sub(r'million|m\b', '', numeric_part, flags=re.IGNORECASE).strip()
            elif re.search(r'billion|b\b', numeric_part, re.IGNORECASE):
                multiplier = 1_000_000_000
                numeric_part = re.sub(r'billion|b\b', '', numeric_part, flags=re.IGNORECASE).strip()
            elif re.search(r'trillion|t\b', numeric_part, re.IGNORECASE):
                multiplier = 1_000_000_000_000
                numeric_part = re.sub(r'trillion|t\b', '', numeric_part, flags=re.IGNORECASE).strip()
            
            # Convert to float and apply multiplier
            try:
                value = float(numeric_part) * multiplier
                # Return with currency symbol
                return f"{currency_symbol}{value:.2f}"
            except ValueError:
                # If conversion fails, return the original string
                return money_str
        
        return re.sub(self.money_pattern, replace_money, text)
    
    def _mark_financial_terms(self, text: str) -> str:
        """
        Mark financial terms with special tags.
        
        Examples:
            "EBITDA increased" -> "[METRIC]EBITDA[/METRIC] increased"
            "P/E Ratio of 15" -> "[RATIO]P/E Ratio[/RATIO] of 15"
        """
        # Create a copy to modify
        modified_text = text
        
        # Sort terms by length (longest first) to avoid partial matches
        sorted_terms = sorted(self.all_terms, key=len, reverse=True)
        
        # Dictionary to track inserted tags and their positions
        tags = {}
        
        # Process each term
        for term in sorted_terms:
            # Case-insensitive search (but preserve the original term case)
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
            # Find all matches
            for match in pattern.finditer(modified_text):
                start, end = match.span()
                
                # Skip if this position is already tagged
                skip = False
                for tag_start, tag_end in tags.values():
                    if (start >= tag_start and start < tag_end) or (end > tag_start and end <= tag_end):
                        skip = True
                        break
                
                if skip:
                    continue
                
                # Determine tag type
                tag_type = None
                if term in self.metrics:
                    tag_type = "METRIC"
                elif term in self.ratios:
                    tag_type = "RATIO"
                elif term in self.statements:
                    tag_type = "STATEMENT"
                elif term in self.regulations:
                    tag_type = "REGULATION"
                else:
                    # Default to a generic tag for other terms
                    tag_type = "FINANCIAL_TERM"
                
                # Store tag position
                tags[len(tags)] = (start, end)
        
        # Apply tags from right to left to maintain correct positions
        sorted_positions = sorted(tags.items(), key=lambda x: x[1][0], reverse=True)
        
        for _, (start, end) in sorted_positions:
            # Get the matched term
            term = modified_text[start:end]
            
            # Determine tag type
            tag_type = None
            if term.lower() in {t.lower() for t in self.metrics}:
                tag_type = "METRIC"
            elif term.lower() in {t.lower() for t in self.ratios}:
                tag_type = "RATIO"
            elif term.lower() in {t.lower() for t in self.statements}:
                tag_type = "STATEMENT"
            elif term.lower() in {t.lower() for t in self.regulations}:
                tag_type = "REGULATION"
            elif term.lower() in {t.lower() for t in self.all_accounts_normalized}:
                tag_type = "ACCOUNT"
            else:
                # Skip if no tag type determined
                continue
            
            # Insert tags
            modified_text = modified_text[:start] + f"[{tag_type}]{term}[/{tag_type}]" + modified_text[end:]
        
        return modified_text
    
    def _mark_statement_types(self, text: str) -> str:
        """
        Mark financial statement types.
        
        Examples:
            "Balance Sheet" -> "[BALANCE_SHEET]Balance Sheet[/BALANCE_SHEET]"
            "Income Statement" -> "[INCOME_STATEMENT]Income Statement[/INCOME_STATEMENT]"
        """
        def replace_statement(match):
            statement_text = match.group(0)
            statement_type = statement_text.lower()
            
            if "balance sheet" in statement_type or "financial position" in statement_type:
                return f"[BALANCE_SHEET]{statement_text}[/BALANCE_SHEET]"
            elif "income statement" in statement_type or "profit and loss" in statement_type or "operations" in statement_type:
                return f"[INCOME_STATEMENT]{statement_text}[/INCOME_STATEMENT]"
            elif "cash flow" in statement_type:
                return f"[CASH_FLOW]{statement_text}[/CASH_FLOW]"
            elif "changes in equity" in statement_type:
                return f"[EQUITY_STATEMENT]{statement_text}[/EQUITY_STATEMENT]"
            elif "comprehensive income" in statement_type:
                return f"[COMPREHENSIVE_INCOME]{statement_text}[/COMPREHENSIVE_INCOME]"
            else:
                return f"[STATEMENT]{statement_text}[/STATEMENT]"
        
        return re.sub(self.statement_type_pattern, replace_statement, text)
    
    def _mark_periods(self, text: str) -> str:
        """
        Mark financial periods.
        
        Examples:
            "Year ended December 31, 2022" -> "[PERIOD]Year ended December 31, 2022[/PERIOD]"
            "Q1 2023" -> "[PERIOD]Q1 2023[/PERIOD]"
        """
        def replace_period(match):
            period_text = match.group(0)
            return f"[PERIOD]{period_text}[/PERIOD]"
        
        # First, find date matches
        date_matches = list(self.date_pattern.finditer(text))
        
        # Now find period matches, but exclude portions that are dates
        period_positions = []
        for period_match in self.period_pattern.finditer(text):
            period_start, period_end = period_match.span()
            
            # Check if this period overlaps with any date
            overlap = False
            for date_match in date_matches:
                date_start, date_end = date_match.span()
                if (period_start >= date_start and period_start < date_end) or \
                   (period_end > date_start and period_end <= date_end):
                    overlap = True
                    break
            
            if not overlap:
                period_positions.append((period_start, period_end))
        
        # Apply tags from right to left
        period_positions.sort(reverse=True)
        result = text
        
        for start, end in period_positions:
            period_text = result[start:end]
            result = result[:start] + f"[PERIOD]{period_text}[/PERIOD]" + result[end:]
        
        return result
    
    def _mark_tables(self, text: str) -> str:
        """
        Mark table structures.
        
        Examples:
            "Item | Value | Change\nRevenue | $100M | +5%" ->
            "[TABLE]Item | Value | Change\nRevenue | $100M | +5%[/TABLE]"
        """
        def replace_table(match):
            table_text = match.group(0)
            
            # Process the table structure
            processed_table = table_text
            
            # Mark rows
            rows = processed_table.strip().split('\n')
            processed_rows = []
            
            for row in rows:
                # For pipe-separated tables
                if '|' in row:
                    cells = row.split('|')
                    processed_cells = [f"[CELL]{cell.strip()}[/CELL]" for cell in cells]
                    processed_row = "[ROW]" + "|".join(processed_cells) + "[/ROW]"
                # For comma-separated tables
                elif ',' in row:
                    cells = row.split(',')
                    processed_cells = [f"[CELL]{cell.strip()}[/CELL]" for cell in cells]
                    processed_row = "[ROW]" + ",".join(processed_cells) + "[/ROW]"
                else:
                    processed_row = f"[ROW]{row}[/ROW]"
                
                processed_rows.append(processed_row)
            
            processed_table = "\n".join(processed_rows)
            
            return f"[TABLE]{processed_table}[/TABLE]"
        
        return re.sub(self.table_pattern, replace_table, text)
    
    def _normalize_numerical_values(self, text: str) -> str:
        """Normalize numerical values with units."""
        def replace_units(match):
            value_str = match.group(0)
            
            # Extract the numeric part
            numeric_part = re.search(r'\d+(?:\.\d+)?', value_str).group(0)
            
            # Extract the unit
            unit_match = re.search(r'(?:thousand|million|billion|trillion|percent|k|m|b|t|%)', 
                                  value_str, re.IGNORECASE)
            
            if not unit_match:
                return value_str
                
            unit = unit_match.group(0).lower()
            
            # Convert to standard form
            try:
                value = float(numeric_part)
                
                if unit in ('thousand', 'k'):
                    value *= 1_000
                elif unit in ('million', 'm'):
                    value *= 1_000_000
                elif unit in ('billion', 'b'):
                    value *= 1_000_000_000
                elif unit in ('trillion', 't'):
                    value *= 1_000_000_000_000
                
                # Format appropriately
                if unit in ('percent', '%'):
                    return f"{value}%"
                else:
                    return f"{value:.2f}"
            except ValueError:
                return value_str
        
        return re.sub(self.units_pattern, replace_units, text)
    
    def process_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to the text.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text with financial-specific enhancements
        """
        # Normalize whitespace
        processed_text = re.sub(r'\s+', ' ', text).strip()
        
        # Mark financial terms
        processed_text = self._mark_financial_terms(processed_text)
        
        # Mark statement types
        processed_text = self._mark_statement_types(processed_text)
        
        # Mark periods
        processed_text = self._mark_periods(processed_text)
        
        # Mark tables
        processed_text = self._mark_tables(processed_text)
        
        # Normalize money values
        processed_text = self._normalize_money(processed_text)
        
        # Normalize numerical values
        processed_text = self._normalize_numerical_values(processed_text)
        
        return processed_text
