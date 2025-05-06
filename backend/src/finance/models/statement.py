"""
Data models for financial statements.

This module defines structured data classes for representing various types of financial statements,
including balance sheets, income statements, and cash flow statements.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import uuid


class StatementType(str, Enum):
    """Types of financial statements."""
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    STATEMENT_OF_CHANGES = "statement_of_changes"
    NOTES = "notes"
    UNKNOWN = "unknown"


class TimePeriodType(str, Enum):
    """Types of financial reporting periods."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    MONTHLY = "monthly"
    TTM = "trailing_twelve_months"
    YTD = "year_to_date"
    CUSTOM = "custom"


class AccountingStandard(str, Enum):
    """Types of accounting standards."""
    GAAP = "gaap"  # Generally Accepted Accounting Principles
    IFRS = "ifrs"  # International Financial Reporting Standards
    LOCAL = "local"  # Local accounting standards
    UNKNOWN = "unknown"


@dataclass
class TimePeriod:
    """Represents a financial reporting period."""
    
    period_type: TimePeriodType
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    label: Optional[str] = None
    
    def __post_init__(self):
        """Generate a label if not provided."""
        if self.label is None:
            if self.period_type == TimePeriodType.ANNUAL and self.fiscal_year:
                self.label = f"FY{self.fiscal_year}"
            elif self.period_type == TimePeriodType.QUARTERLY and self.fiscal_year and self.fiscal_quarter:
                self.label = f"Q{self.fiscal_quarter} FY{self.fiscal_year}"
            elif self.start_date and self.end_date:
                self.label = f"{self.start_date.isoformat()} to {self.end_date.isoformat()}"
            else:
                self.label = "Undefined Period"


@dataclass
class LineItem:
    """Represents a single line item in a financial statement."""
    
    name: str
    value: Optional[float] = None
    raw_value: Optional[str] = None
    unit: Optional[str] = "USD"
    scaling_factor: Optional[float] = 1.0  # e.g., 1000 for thousands, 1000000 for millions
    is_calculated: bool = False
    parent_name: Optional[str] = None
    notes_ref: Optional[List[str]] = field(default_factory=list)
    confidence: Optional[float] = 1.0  # For extraction confidence
    
    @property
    def scaled_value(self) -> Optional[float]:
        """Get the value adjusted by the scaling factor."""
        if self.value is not None:
            return self.value * self.scaling_factor
        return None


@dataclass
class StatementMetadata:
    """Metadata for a financial statement."""
    
    company_name: str
    statement_type: StatementType
    period: TimePeriod
    accounting_standard: AccountingStandard = AccountingStandard.UNKNOWN
    currency: str = "USD"
    audit_status: Optional[str] = None  # e.g., "Audited", "Unaudited", "Reviewed"
    prepared_by: Optional[str] = None
    prepared_date: Optional[date] = None
    source_document: Optional[str] = None
    source_page_numbers: Optional[List[int]] = field(default_factory=list)
    extraction_date: Optional[date] = field(default_factory=date.today)
    confidence_score: float = 1.0  # Overall confidence in extraction accuracy


@dataclass
class FinancialStatement:
    """
    Base class for financial statements.
    
    This provides the common structure for all financial statement types.
    """
    
    metadata: StatementMetadata
    line_items: Dict[str, LineItem] = field(default_factory=dict)
    section_structure: Dict[str, List[str]] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def add_line_item(self, line_item: LineItem) -> None:
        """Add a line item to the statement."""
        self.line_items[line_item.name] = line_item
    
    def get_line_item(self, name: str) -> Optional[LineItem]:
        """Get a line item by name."""
        return self.line_items.get(name)
    
    def get_section_items(self, section_name: str) -> List[LineItem]:
        """Get all line items in a section."""
        item_names = self.section_structure.get(section_name, [])
        return [self.line_items[name] for name in item_names if name in self.line_items]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the statement to a dictionary representation."""
        return {
            "id": self.id,
            "company_name": self.metadata.company_name,
            "statement_type": self.metadata.statement_type.value,
            "period": {
                "type": self.metadata.period.period_type.value,
                "label": self.metadata.period.label,
                "fiscal_year": self.metadata.period.fiscal_year,
                "fiscal_quarter": self.metadata.period.fiscal_quarter
            },
            "accounting_standard": self.metadata.accounting_standard.value,
            "currency": self.metadata.currency,
            "line_items": {
                name: {
                    "value": item.value,
                    "scaled_value": item.scaled_value,
                    "unit": item.unit,
                    "scaling_factor": item.scaling_factor
                } for name, item in self.line_items.items()
            }
        }


@dataclass
class BalanceSheet(FinancialStatement):
    """Represents a balance sheet financial statement."""
    
    def __post_init__(self):
        """Initialize default section structure if not provided."""
        if not self.section_structure:
            self.section_structure = {
                "Assets": [],
                "Current Assets": [],
                "Non-Current Assets": [],
                "Liabilities": [],
                "Current Liabilities": [],
                "Non-Current Liabilities": [],
                "Equity": [],
                "Total": []
            }
    
    @property
    def total_assets(self) -> Optional[float]:
        """Get the total assets value."""
        asset_item = self.get_line_item("Total Assets")
        if asset_item:
            return asset_item.scaled_value
        return None
    
    @property
    def total_liabilities(self) -> Optional[float]:
        """Get the total liabilities value."""
        liabilities_item = self.get_line_item("Total Liabilities")
        if liabilities_item:
            return liabilities_item.scaled_value
        return None
    
    @property
    def total_equity(self) -> Optional[float]:
        """Get the total equity value."""
        equity_item = self.get_line_item("Total Equity")
        if equity_item:
            return equity_item.scaled_value
        return None
    
    def validate_balance(self) -> bool:
        """Validate that assets = liabilities + equity."""
        assets = self.total_assets
        liabilities = self.total_liabilities
        equity = self.total_equity
        
        if None in (assets, liabilities, equity):
            return False
        
        # Allow small rounding differences (0.1%)
        difference = abs(assets - (liabilities + equity))
        tolerance = assets * 0.001
        
        return difference <= tolerance


@dataclass
class IncomeStatement(FinancialStatement):
    """Represents an income statement financial statement."""
    
    def __post_init__(self):
        """Initialize default section structure if not provided."""
        if not self.section_structure:
            self.section_structure = {
                "Revenue": [],
                "Cost of Revenue": [],
                "Gross Profit": [],
                "Operating Expenses": [],
                "Operating Income": [],
                "Non-Operating Items": [],
                "Income Before Tax": [],
                "Income Tax": [],
                "Net Income": []
            }
    
    @property
    def revenue(self) -> Optional[float]:
        """Get the revenue value."""
        revenue_item = self.get_line_item("Revenue") or self.get_line_item("Total Revenue")
        if revenue_item:
            return revenue_item.scaled_value
        return None
    
    @property
    def net_income(self) -> Optional[float]:
        """Get the net income value."""
        net_income_item = self.get_line_item("Net Income")
        if net_income_item:
            return net_income_item.scaled_value
        return None
    
    @property
    def gross_profit(self) -> Optional[float]:
        """Get the gross profit value."""
        gross_profit_item = self.get_line_item("Gross Profit")
        if gross_profit_item:
            return gross_profit_item.scaled_value
        return None
    
    @property
    def operating_income(self) -> Optional[float]:
        """Get the operating income value."""
        operating_income_item = (
            self.get_line_item("Operating Income") or 
            self.get_line_item("Operating Profit")
        )
        if operating_income_item:
            return operating_income_item.scaled_value
        return None


@dataclass
class CashFlowStatement(FinancialStatement):
    """Represents a cash flow statement."""
    
    def __post_init__(self):
        """Initialize default section structure if not provided."""
        if not self.section_structure:
            self.section_structure = {
                "Operating Activities": [],
                "Investing Activities": [],
                "Financing Activities": [],
                "Net Change in Cash": [],
                "Beginning Cash Balance": [],
                "Ending Cash Balance": []
            }
    
    @property
    def operating_cash_flow(self) -> Optional[float]:
        """Get the operating cash flow value."""
        ocf_item = (
            self.get_line_item("Net Cash from Operating Activities") or
            self.get_line_item("Net Cash Provided by Operating Activities")
        )
        if ocf_item:
            return ocf_item.scaled_value
        return None
    
    @property
    def investing_cash_flow(self) -> Optional[float]:
        """Get the investing cash flow value."""
        icf_item = (
            self.get_line_item("Net Cash from Investing Activities") or
            self.get_line_item("Net Cash Used in Investing Activities")
        )
        if icf_item:
            return icf_item.scaled_value
        return None
    
    @property
    def financing_cash_flow(self) -> Optional[float]:
        """Get the financing cash flow value."""
        fcf_item = (
            self.get_line_item("Net Cash from Financing Activities") or
            self.get_line_item("Net Cash Used in Financing Activities")
        )
        if fcf_item:
            return fcf_item.scaled_value
        return None
    
    @property
    def free_cash_flow(self) -> Optional[float]:
        """Calculate free cash flow if possible."""
        ocf = self.operating_cash_flow
        capex_item = self.get_line_item("Capital Expenditures")
        
        if ocf is None:
            return None
        
        # If we have capex, use it to calculate FCF
        if capex_item and capex_item.scaled_value is not None:
            capex = capex_item.scaled_value
            # Ensure capex is negative for calculation
            if capex > 0:
                capex = -capex
            return ocf + capex
        
        # If no explicit capex, try inferring from investing activities
        ppnt_item = self.get_line_item("Purchase of Property and Equipment")
        if ppnt_item and ppnt_item.scaled_value is not None:
            ppnt = ppnt_item.scaled_value
            # Ensure ppnt is negative for calculation
            if ppnt > 0:
                ppnt = -ppnt
            return ocf + ppnt
        
        return None


# Factory function to create appropriate statement type
def create_financial_statement(
    statement_type: StatementType, 
    metadata: StatementMetadata, 
    line_items: Dict[str, LineItem] = None,
    section_structure: Dict[str, List[str]] = None
) -> FinancialStatement:
    """
    Factory function to create the appropriate financial statement type.
    
    Args:
        statement_type: The type of statement to create
        metadata: Metadata for the statement
        line_items: Optional dictionary of line items
        section_structure: Optional dictionary of section structures
        
    Returns:
        An appropriate subclass of FinancialStatement
    """
    if line_items is None:
        line_items = {}
    if section_structure is None:
        section_structure = {}
    
    if statement_type == StatementType.BALANCE_SHEET:
        return BalanceSheet(
            metadata=metadata,
            line_items=line_items,
            section_structure=section_structure
        )
    elif statement_type == StatementType.INCOME_STATEMENT:
        return IncomeStatement(
            metadata=metadata,
            line_items=line_items,
            section_structure=section_structure
        )
    elif statement_type == StatementType.CASH_FLOW:
        return CashFlowStatement(
            metadata=metadata,
            line_items=line_items,
            section_structure=section_structure
        )
    else:
        return FinancialStatement(
            metadata=metadata,
            line_items=line_items,
            section_structure=section_structure
        )
