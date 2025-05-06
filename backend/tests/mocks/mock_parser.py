"""
Mock implementations of parser components for testing purposes.

These mocks allow tests to run without requiring all the external dependencies
that the actual implementations need.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.finance.parsers.base_parser import ParsingResult
from src.finance.models.statement import (
    FinancialStatement, 
    BalanceSheet, 
    IncomeStatement, 
    CashFlowStatement
)


class MockFinancialStatementParser:
    """Mock implementation of a financial statement parser for testing."""
    
    def __init__(self):
        """Initialize the mock parser."""
        pass
    
    def parse(self, file_path: Union[str, Path], **kwargs) -> ParsingResult:
        """
        Mock implementation that returns sample financial statements.
        
        Args:
            file_path: Path to the file to parse
            **kwargs: Additional parsing parameters
        
        Returns:
            ParsingResult containing sample financial statements
        """
        file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
        
        # Create sample statements based on the file name
        sample_statements = []
        
        # Balance sheet
        if "balance" in file_path.name.lower() or "annual" in file_path.name.lower():
            sample_statements.append(self._create_sample_balance_sheet())
        
        # Income statement
        if "income" in file_path.name.lower() or "annual" in file_path.name.lower():
            sample_statements.append(self._create_sample_income_statement())
        
        # Cash flow statement
        if "cash" in file_path.name.lower() or "annual" in file_path.name.lower():
            sample_statements.append(self._create_sample_cash_flow_statement())
        
        # Always return at least one statement
        if not sample_statements:
            sample_statements.append(self._create_sample_balance_sheet())
        
        return ParsingResult(
            statements=sample_statements,
            metadata={
                "company_name": "Test Company Inc.",
                "document_type": "annual_report",
                "document_date": "2023-12-31",
                "fiscal_year": 2023,
                "currency": "USD"
            },
            raw_text=f"Sample financial report text for {file_path.name}",
            source_path=str(file_path)
        )
    
    def _create_sample_balance_sheet(self) -> BalanceSheet:
        """Create a sample balance sheet for testing."""
        balance_sheet_data = {
            "statement_type": "balance_sheet",
            "period_end_date": "2023-12-31",
            "company_name": "Test Company Inc.",
            "currency": "USD",
            "assets": {
                "current_assets": {
                    "cash_and_equivalents": 1000000,
                    "short_term_investments": 500000,
                    "accounts_receivable": 750000,
                    "inventory": 600000,
                    "prepaid_expenses": 100000,
                    "total_current_assets": 2950000
                },
                "non_current_assets": {
                    "property_plant_equipment": 3000000,
                    "goodwill": 500000,
                    "intangible_assets": 250000,
                    "long_term_investments": 1000000,
                    "total_non_current_assets": 4750000
                },
                "total_assets": 7700000
            },
            "liabilities": {
                "current_liabilities": {
                    "accounts_payable": 400000,
                    "short_term_debt": 200000,
                    "accrued_expenses": 150000,
                    "total_current_liabilities": 750000
                },
                "non_current_liabilities": {
                    "long_term_debt": 2000000,
                    "deferred_tax_liabilities": 250000,
                    "total_non_current_liabilities": 2250000
                },
                "total_liabilities": 3000000
            },
            "equity": {
                "common_stock": 1000000,
                "retained_earnings": 3500000,
                "treasury_stock": -100000,
                "accumulated_other_comprehensive_income": 300000,
                "total_equity": 4700000
            }
        }
        
        return BalanceSheet.from_dict(balance_sheet_data)
    
    def _create_sample_income_statement(self) -> IncomeStatement:
        """Create a sample income statement for testing."""
        income_statement_data = {
            "statement_type": "income_statement",
            "period_end_date": "2023-12-31",
            "company_name": "Test Company Inc.",
            "currency": "USD",
            "period_length_months": 12,
            "revenue": 5000000,
            "cost_of_goods_sold": 2500000,
            "gross_profit": 2500000,
            "operating_expenses": {
                "research_and_development": 500000,
                "selling_general_administrative": 750000,
                "depreciation_amortization": 250000,
                "total_operating_expenses": 1500000
            },
            "operating_income": 1000000,
            "non_operating_income": 100000,
            "interest_expense": 150000,
            "income_before_tax": 950000,
            "income_tax_expense": 200000,
            "net_income": 750000,
            "earnings_per_share": {
                "basic": 1.50,
                "diluted": 1.45
            }
        }
        
        return IncomeStatement.from_dict(income_statement_data)
    
    def _create_sample_cash_flow_statement(self) -> CashFlowStatement:
        """Create a sample cash flow statement for testing."""
        cash_flow_statement_data = {
            "statement_type": "cash_flow_statement",
            "period_end_date": "2023-12-31",
            "company_name": "Test Company Inc.",
            "currency": "USD",
            "period_length_months": 12,
            "operating_activities": {
                "net_income": 750000,
                "adjustments": {
                    "depreciation_amortization": 250000,
                    "stock_based_compensation": 50000,
                    "deferred_taxes": 25000,
                    "total_adjustments": 325000
                },
                "changes_in_working_capital": {
                    "accounts_receivable": -100000,
                    "inventory": -50000,
                    "accounts_payable": 75000,
                    "total_changes_in_working_capital": -75000
                },
                "net_cash_from_operating_activities": 1000000
            },
            "investing_activities": {
                "capital_expenditures": -400000,
                "acquisitions": -200000,
                "short_term_investments": -100000,
                "net_cash_from_investing_activities": -700000
            },
            "financing_activities": {
                "debt_issuance": 500000,
                "debt_repayment": -300000,
                "dividends_paid": -200000,
                "share_repurchases": -100000,
                "net_cash_from_financing_activities": -100000
            },
            "effect_of_exchange_rates": -25000,
            "net_change_in_cash": 175000,
            "beginning_cash_balance": 825000,
            "ending_cash_balance": 1000000
        }
        
        return CashFlowStatement.from_dict(cash_flow_statement_data)
