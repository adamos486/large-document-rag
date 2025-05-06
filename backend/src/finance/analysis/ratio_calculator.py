"""
Financial Ratio Calculator

This module provides utilities for calculating standard and custom financial ratios
from financial statement data. It implements comprehensive ratio analysis with proper
handling of special cases and exceptions.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ..models.statement import (
    BalanceSheet, 
    IncomeStatement, 
    CashFlowStatement, 
    FinancialStatement
)

logger = logging.getLogger(__name__)


@dataclass
class FinancialRatio:
    """Represents a calculated financial ratio."""
    
    name: str
    value: Optional[float]
    formula: str
    category: str
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    is_percentage: bool = False
    is_multiple: bool = False
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    industry_average: Optional[float] = None
    description: Optional[str] = None
    
    def format_value(self, decimal_places: int = 2) -> str:
        """Format the ratio value for display."""
        if self.value is None:
            return "N/A"
        
        if self.is_percentage:
            return f"{self.value * 100:.{decimal_places}f}%"
        elif self.is_multiple:
            return f"{self.value:.{decimal_places}f}x"
        else:
            return f"{self.value:.{decimal_places}f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ratio to a dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "formula": self.formula,
            "category": self.category,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "is_percentage": self.is_percentage,
            "is_multiple": self.is_multiple,
            "trend": self.trend,
            "industry_average": self.industry_average,
            "description": self.description,
            "formatted_value": self.format_value()
        }


class BaseRatioCalculator:
    """Base class for ratio calculators providing common functionality."""
    
    def _safe_divide(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Safely divide two numbers, handling None and zero denominators."""
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return numerator / denominator


class LiquidityRatioCalculator(BaseRatioCalculator):
    """Calculates liquidity ratios from balance sheet data."""
    
    def __init__(self, balance_sheet: BalanceSheet):
        """
        Initialize with a balance sheet.
        
        Args:
            balance_sheet: Balance sheet data for ratio calculation
        """
        self.balance_sheet = balance_sheet
    
    def current_ratio(self) -> FinancialRatio:
        """
        Calculate the current ratio (Current Assets / Current Liabilities).
        
        This ratio measures a company's ability to pay short-term obligations.
        """
        current_assets = None
        current_liabilities = None
        
        # Try to get values from balance sheet
        ca_item = self.balance_sheet.get_line_item("Current Assets") or self.balance_sheet.get_line_item("Total Current Assets")
        if ca_item:
            current_assets = ca_item.scaled_value
            
        cl_item = self.balance_sheet.get_line_item("Current Liabilities") or self.balance_sheet.get_line_item("Total Current Liabilities")
        if cl_item:
            current_liabilities = cl_item.scaled_value
        
        value = self._safe_divide(current_assets, current_liabilities)
        
        return FinancialRatio(
            name="Current Ratio",
            value=value,
            formula="Current Assets / Current Liabilities",
            category="Liquidity",
            numerator=current_assets,
            denominator=current_liabilities,
            is_multiple=True,
            description="Measures a company's ability to pay short-term obligations."
        )
    
    def quick_ratio(self) -> FinancialRatio:
        """
        Calculate the quick ratio ((Current Assets - Inventory) / Current Liabilities).
        
        This ratio measures a company's ability to meet short-term obligations with
        its most liquid assets.
        """
        current_assets = None
        inventory = None
        current_liabilities = None
        
        # Get values from balance sheet
        ca_item = self.balance_sheet.get_line_item("Current Assets") or self.balance_sheet.get_line_item("Total Current Assets")
        if ca_item:
            current_assets = ca_item.scaled_value
            
        inv_item = self.balance_sheet.get_line_item("Inventory") or self.balance_sheet.get_line_item("Inventories")
        if inv_item:
            inventory = inv_item.scaled_value
        else:
            inventory = 0  # Assume zero if not found
            
        cl_item = self.balance_sheet.get_line_item("Current Liabilities") or self.balance_sheet.get_line_item("Total Current Liabilities")
        if cl_item:
            current_liabilities = cl_item.scaled_value
        
        numerator = None
        if current_assets is not None and inventory is not None:
            numerator = current_assets - inventory
            
        value = self._safe_divide(numerator, current_liabilities)
        
        return FinancialRatio(
            name="Quick Ratio",
            value=value,
            formula="(Current Assets - Inventory) / Current Liabilities",
            category="Liquidity",
            numerator=numerator,
            denominator=current_liabilities,
            is_multiple=True,
            description="Measures a company's ability to meet short-term obligations with its most liquid assets."
        )
    
    def cash_ratio(self) -> FinancialRatio:
        """
        Calculate the cash ratio (Cash & Equivalents / Current Liabilities).
        
        This ratio measures a company's ability to cover short-term liabilities with cash.
        """
        cash = None
        current_liabilities = None
        
        # Get values from balance sheet
        cash_item = (
            self.balance_sheet.get_line_item("Cash and Cash Equivalents") or
            self.balance_sheet.get_line_item("Cash") or
            self.balance_sheet.get_line_item("Cash & Equivalents")
        )
        if cash_item:
            cash = cash_item.scaled_value
            
        cl_item = self.balance_sheet.get_line_item("Current Liabilities") or self.balance_sheet.get_line_item("Total Current Liabilities")
        if cl_item:
            current_liabilities = cl_item.scaled_value
        
        value = self._safe_divide(cash, current_liabilities)
        
        return FinancialRatio(
            name="Cash Ratio",
            value=value,
            formula="Cash & Equivalents / Current Liabilities",
            category="Liquidity",
            numerator=cash,
            denominator=current_liabilities,
            is_multiple=True,
            description="Measures a company's ability to cover short-term liabilities with its cash reserves."
        )
    
    def calculate_all_liquidity_ratios(self) -> Dict[str, FinancialRatio]:
        """Calculate all liquidity ratios."""
        return {
            "current_ratio": self.current_ratio(),
            "quick_ratio": self.quick_ratio(),
            "cash_ratio": self.cash_ratio()
        }


class ProfitabilityRatioCalculator(BaseRatioCalculator):
    """Calculates profitability ratios from income statement and balance sheet data."""
    
    def __init__(self, income_statement: IncomeStatement, balance_sheet: Optional[BalanceSheet] = None):
        """
        Initialize with income statement and optional balance sheet.
        
        Args:
            income_statement: Income statement data for ratio calculation
            balance_sheet: Optional balance sheet data for ratio calculation
        """
        self.income_statement = income_statement
        self.balance_sheet = balance_sheet
    
    def gross_margin(self) -> FinancialRatio:
        """
        Calculate the gross margin (Gross Profit / Revenue).
        
        This ratio measures a company's manufacturing and distribution efficiency.
        """
        gross_profit = self.income_statement.gross_profit
        revenue = self.income_statement.revenue
        
        value = self._safe_divide(gross_profit, revenue)
        
        return FinancialRatio(
            name="Gross Margin",
            value=value,
            formula="Gross Profit / Revenue",
            category="Profitability",
            numerator=gross_profit,
            denominator=revenue,
            is_percentage=True,
            description="Measures a company's manufacturing and distribution efficiency."
        )
    
    def operating_margin(self) -> FinancialRatio:
        """
        Calculate the operating margin (Operating Income / Revenue).
        
        This ratio measures a company's operating efficiency and pricing strategy.
        """
        operating_income = self.income_statement.operating_income
        revenue = self.income_statement.revenue
        
        value = self._safe_divide(operating_income, revenue)
        
        return FinancialRatio(
            name="Operating Margin",
            value=value,
            formula="Operating Income / Revenue",
            category="Profitability",
            numerator=operating_income,
            denominator=revenue,
            is_percentage=True,
            description="Measures a company's operating efficiency and pricing strategy."
        )
    
    def net_profit_margin(self) -> FinancialRatio:
        """
        Calculate the net profit margin (Net Income / Revenue).
        
        This ratio measures how much of each dollar of revenue is kept as profit.
        """
        net_income = self.income_statement.net_income
        revenue = self.income_statement.revenue
        
        value = self._safe_divide(net_income, revenue)
        
        return FinancialRatio(
            name="Net Profit Margin",
            value=value,
            formula="Net Income / Revenue",
            category="Profitability",
            numerator=net_income,
            denominator=revenue,
            is_percentage=True,
            description="Measures how much of each dollar of revenue is kept as profit."
        )
    
    def return_on_assets(self) -> FinancialRatio:
        """
        Calculate the return on assets (Net Income / Total Assets).
        
        This ratio measures how efficiently a company uses its assets to generate profit.
        """
        if not self.balance_sheet:
            return FinancialRatio(
                name="Return on Assets",
                value=None,
                formula="Net Income / Total Assets",
                category="Profitability",
                is_percentage=True,
                description="Measures how efficiently a company uses its assets to generate profit."
            )
            
        net_income = self.income_statement.net_income
        total_assets = self.balance_sheet.total_assets
        
        value = self._safe_divide(net_income, total_assets)
        
        return FinancialRatio(
            name="Return on Assets",
            value=value,
            formula="Net Income / Total Assets",
            category="Profitability",
            numerator=net_income,
            denominator=total_assets,
            is_percentage=True,
            description="Measures how efficiently a company uses its assets to generate profit."
        )
    
    def return_on_equity(self) -> FinancialRatio:
        """
        Calculate the return on equity (Net Income / Total Equity).
        
        This ratio measures a company's profitability by revealing how much profit
        it generates with the money shareholders have invested.
        """
        if not self.balance_sheet:
            return FinancialRatio(
                name="Return on Equity",
                value=None,
                formula="Net Income / Total Equity",
                category="Profitability",
                is_percentage=True,
                description="Measures a company's profitability relative to shareholder equity."
            )
            
        net_income = self.income_statement.net_income
        total_equity = self.balance_sheet.total_equity
        
        value = self._safe_divide(net_income, total_equity)
        
        return FinancialRatio(
            name="Return on Equity",
            value=value,
            formula="Net Income / Total Equity",
            category="Profitability",
            numerator=net_income,
            denominator=total_equity,
            is_percentage=True,
            description="Measures a company's profitability relative to shareholder equity."
        )
    
    def calculate_all_profitability_ratios(self) -> Dict[str, FinancialRatio]:
        """Calculate all profitability ratios."""
        ratios = {
            "gross_margin": self.gross_margin(),
            "operating_margin": self.operating_margin(),
            "net_profit_margin": self.net_profit_margin()
        }
        
        # Add ratios that require balance sheet if available
        if self.balance_sheet:
            ratios.update({
                "return_on_assets": self.return_on_assets(),
                "return_on_equity": self.return_on_equity()
            })
            
        return ratios


class SolvencyRatioCalculator(BaseRatioCalculator):
    """Calculates solvency ratios from balance sheet and income statement data."""
    
    def __init__(
        self, 
        balance_sheet: BalanceSheet, 
        income_statement: Optional[IncomeStatement] = None
    ):
        """
        Initialize with balance sheet and optional income statement.
        
        Args:
            balance_sheet: Balance sheet data for ratio calculation
            income_statement: Optional income statement data for ratio calculation
        """
        self.balance_sheet = balance_sheet
        self.income_statement = income_statement
    
    def debt_to_equity(self) -> FinancialRatio:
        """
        Calculate the debt-to-equity ratio (Total Debt / Total Equity).
        
        This ratio measures a company's financial leverage.
        """
        # Try different variations of debt terminology
        debt_item = (
            self.balance_sheet.get_line_item("Total Debt") or
            self.balance_sheet.get_line_item("Long-term Debt") or
            self.balance_sheet.get_line_item("Total Liabilities")
        )
        
        if debt_item:
            total_debt = debt_item.scaled_value
        else:
            # If no specific debt line item, use total liabilities
            total_debt = self.balance_sheet.total_liabilities
            
        total_equity = self.balance_sheet.total_equity
        
        value = self._safe_divide(total_debt, total_equity)
        
        return FinancialRatio(
            name="Debt-to-Equity Ratio",
            value=value,
            formula="Total Debt / Total Equity",
            category="Solvency",
            numerator=total_debt,
            denominator=total_equity,
            is_multiple=True,
            description="Measures a company's financial leverage and risk."
        )
    
    def debt_ratio(self) -> FinancialRatio:
        """
        Calculate the debt ratio (Total Liabilities / Total Assets).
        
        This ratio measures the percentage of assets financed by debt.
        """
        total_liabilities = self.balance_sheet.total_liabilities
        total_assets = self.balance_sheet.total_assets
        
        value = self._safe_divide(total_liabilities, total_assets)
        
        return FinancialRatio(
            name="Debt Ratio",
            value=value,
            formula="Total Liabilities / Total Assets",
            category="Solvency",
            numerator=total_liabilities,
            denominator=total_assets,
            is_percentage=True,
            description="Measures the percentage of assets financed by debt."
        )
    
    def interest_coverage_ratio(self) -> FinancialRatio:
        """
        Calculate the interest coverage ratio (EBIT / Interest Expense).
        
        This ratio measures a company's ability to pay interest on its debt.
        """
        if not self.income_statement:
            return FinancialRatio(
                name="Interest Coverage Ratio",
                value=None,
                formula="EBIT / Interest Expense",
                category="Solvency",
                is_multiple=True,
                description="Measures a company's ability to pay interest on its debt."
            )
        
        # Try to get EBIT or Operating Income
        ebit_item = (
            self.income_statement.get_line_item("EBIT") or 
            self.income_statement.get_line_item("Operating Income") or
            self.income_statement.get_line_item("Operating Profit")
        )
        
        interest_item = (
            self.income_statement.get_line_item("Interest Expense") or
            self.income_statement.get_line_item("Interest Paid") or
            self.income_statement.get_line_item("Finance Costs")
        )
        
        if ebit_item and interest_item:
            ebit = ebit_item.scaled_value
            interest_expense = interest_item.scaled_value
            value = self._safe_divide(ebit, interest_expense)
        else:
            ebit = None 
            interest_expense = None
            value = None
        
        return FinancialRatio(
            name="Interest Coverage Ratio",
            value=value,
            formula="EBIT / Interest Expense",
            category="Solvency",
            numerator=ebit,
            denominator=interest_expense,
            is_multiple=True,
            description="Measures a company's ability to pay interest on its debt."
        )
    
    def calculate_all_solvency_ratios(self) -> Dict[str, FinancialRatio]:
        """Calculate all solvency ratios."""
        ratios = {
            "debt_to_equity": self.debt_to_equity(),
            "debt_ratio": self.debt_ratio()
        }
        
        # Add ratios that require income statement if available
        if self.income_statement:
            ratios["interest_coverage_ratio"] = self.interest_coverage_ratio()
            
        return ratios


class FinancialRatioCalculator:
    """
    Comprehensive calculator for financial ratios.
    
    This calculator encapsulates the specialized calculators for different ratio categories
    and provides a unified interface for calculating all financial ratios.
    """
    
    def __init__(
        self,
        balance_sheet: Optional[BalanceSheet] = None,
        income_statement: Optional[IncomeStatement] = None,
        cash_flow_statement: Optional[CashFlowStatement] = None
    ):
        """
        Initialize with financial statements.
        
        Args:
            balance_sheet: Balance sheet data
            income_statement: Income statement data
            cash_flow_statement: Cash flow statement data
        """
        self.balance_sheet = balance_sheet
        self.income_statement = income_statement
        self.cash_flow_statement = cash_flow_statement
        
        # Initialize specialized calculators as needed
        self._liquidity_calculator = None
        self._profitability_calculator = None
        self._solvency_calculator = None
    
    @property
    def liquidity_calculator(self) -> Optional[LiquidityRatioCalculator]:
        """Get the liquidity ratio calculator, initializing if necessary."""
        if self._liquidity_calculator is None and self.balance_sheet is not None:
            self._liquidity_calculator = LiquidityRatioCalculator(self.balance_sheet)
        return self._liquidity_calculator
    
    @property
    def profitability_calculator(self) -> Optional[ProfitabilityRatioCalculator]:
        """Get the profitability ratio calculator, initializing if necessary."""
        if self._profitability_calculator is None and self.income_statement is not None:
            self._profitability_calculator = ProfitabilityRatioCalculator(
                self.income_statement, self.balance_sheet
            )
        return self._profitability_calculator
    
    @property
    def solvency_calculator(self) -> Optional[SolvencyRatioCalculator]:
        """Get the solvency ratio calculator, initializing if necessary."""
        if self._solvency_calculator is None and self.balance_sheet is not None:
            self._solvency_calculator = SolvencyRatioCalculator(
                self.balance_sheet, self.income_statement
            )
        return self._solvency_calculator
    
    def calculate_all_ratios(self) -> Dict[str, Dict[str, FinancialRatio]]:
        """
        Calculate all available ratios based on provided financial statements.
        
        Returns:
            Dictionary of ratio categories containing calculated ratios
        """
        result = {}
        
        # Calculate liquidity ratios if balance sheet is available
        if self.liquidity_calculator:
            result["liquidity"] = self.liquidity_calculator.calculate_all_liquidity_ratios()
        
        # Calculate profitability ratios if income statement is available
        if self.profitability_calculator:
            result["profitability"] = self.profitability_calculator.calculate_all_profitability_ratios()
        
        # Calculate solvency ratios if balance sheet is available
        if self.solvency_calculator:
            result["solvency"] = self.solvency_calculator.calculate_all_solvency_ratios()
        
        return result
