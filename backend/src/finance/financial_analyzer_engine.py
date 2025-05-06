"""
Financial Analyzer Engine

This module provides the main integration point for the financial analysis system,
combining statement parsing, ratio analysis, time series analysis, and insight generation
into a cohesive workflow for financial document processing and analysis.
"""

import os
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set

from .models.statement import (
    FinancialStatement,
    StatementType,
    StatementMetadata,
    TimePeriod,
    AccountingStandard
)
from .parsers.base_parser import ParsingResult, parser_registry
from .parsers.pdf_statement_parser import PDFFinancialStatementParser
from .analysis.ratio_calculator import (
    FinancialRatio,
    FinancialRatioCalculator
)
from .analysis.time_series_analyzer import (
    FinancialTimeSeriesAnalyzer,
    TimeSeriesMetric,
    TrendAnalysisResult,
    TrendDirection
)

logger = logging.getLogger(__name__)


class FinancialAnalyzerEngine:
    """
    Main engine for financial document analysis.
    
    This class orchestrates the financial analysis process, from document parsing
    to detailed financial analysis and insight generation.
    """
    
    def __init__(self):
        """Initialize the financial analyzer engine."""
        # Register document parsers
        self._register_parsers()
        
        # Initialize analyzers
        self.ratio_calculator = FinancialRatioCalculator()
        self.time_series_analyzer = FinancialTimeSeriesAnalyzer()
        
        # Storage for processed statements
        self.statements: Dict[str, List[FinancialStatement]] = {
            StatementType.BALANCE_SHEET.value: [],
            StatementType.INCOME_STATEMENT.value: [],
            StatementType.CASH_FLOW.value: []
        }
        
        # Analysis results
        self.analysis_results: Dict[str, Any] = {}
    
    def _register_parsers(self):
        """Register all available financial document parsers."""
        # Register PDF parser
        parser_registry.register_parser(PDFFinancialStatementParser())
        
        # Additional parsers would be registered here
        # e.g., Excel, HTML, Word, etc.
    
    def process_document(self, document_path: Union[str, Path], **kwargs) -> ParsingResult:
        """
        Process a financial document and extract structured data.
        
        Args:
            document_path: Path to the financial document
            **kwargs: Additional processing parameters
            
        Returns:
            ParsingResult containing extracted financial statements
        """
        document_path = Path(document_path)
        logger.info(f"Processing financial document: {document_path}")
        
        # Parse document using appropriate parser
        result = parser_registry.parse_document(document_path, **kwargs)
        
        if result.is_successful():
            # Store extracted statements by type
            for statement in result.statements:
                statement_type = statement.metadata.statement_type.value
                if statement_type in self.statements:
                    self.statements[statement_type].append(statement)
            
            logger.info(f"Successfully extracted {len(result.statements)} statements from {document_path}")
        else:
            logger.warning(f"Failed to extract statements from {document_path}: {', '.join(result.errors)}")
        
        return result
    
    def process_document_batch(self, document_paths: List[Union[str, Path]], **kwargs) -> List[ParsingResult]:
        """
        Process a batch of financial documents.
        
        Args:
            document_paths: List of paths to financial documents
            **kwargs: Additional processing parameters
            
        Returns:
            List of ParsingResult objects
        """
        results = []
        
        for document_path in document_paths:
            result = self.process_document(document_path, **kwargs)
            results.append(result)
        
        return results
    
    def calculate_financial_ratios(self, company_name: Optional[str] = None) -> Dict[str, Dict[str, FinancialRatio]]:
        """
        Calculate financial ratios for loaded statements.
        
        Args:
            company_name: Optional company name to filter statements
            
        Returns:
            Dictionary of ratio categories containing calculated ratios
        """
        # Filter statements by company if specified
        balance_sheets = self._filter_statements_by_company(
            self.statements[StatementType.BALANCE_SHEET.value], company_name
        )
        income_statements = self._filter_statements_by_company(
            self.statements[StatementType.INCOME_STATEMENT.value], company_name
        )
        cash_flow_statements = self._filter_statements_by_company(
            self.statements[StatementType.CASH_FLOW.value], company_name
        )
        
        # Use the most recent statements for ratio calculation
        balance_sheet = self._get_most_recent_statement(balance_sheets)
        income_statement = self._get_most_recent_statement(income_statements)
        cash_flow_statement = self._get_most_recent_statement(cash_flow_statements)
        
        # Calculate ratios using available statements
        calculator = FinancialRatioCalculator(
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            cash_flow_statement=cash_flow_statement
        )
        
        ratios = calculator.calculate_all_ratios()
        
        # Store results
        self.analysis_results["ratios"] = ratios
        
        return ratios
    
    def analyze_financial_trends(
        self, 
        company_name: Optional[str] = None
    ) -> Dict[str, Optional[TrendAnalysisResult]]:
        """
        Analyze trends in financial metrics over time.
        
        Args:
            company_name: Optional company name to filter statements
            
        Returns:
            Dictionary of metric names to trend analysis results
        """
        # Filter statements by company if specified
        balance_sheets = self._filter_statements_by_company(
            self.statements[StatementType.BALANCE_SHEET.value], company_name
        )
        income_statements = self._filter_statements_by_company(
            self.statements[StatementType.INCOME_STATEMENT.value], company_name
        )
        cash_flow_statements = self._filter_statements_by_company(
            self.statements[StatementType.CASH_FLOW.value], company_name
        )
        
        # Sort statements by time
        balance_sheets = self._sort_statements_by_time(balance_sheets)
        income_statements = self._sort_statements_by_time(income_statements)
        cash_flow_statements = self._sort_statements_by_time(cash_flow_statements)
        
        # Create time series for key metrics
        time_series_metrics = {}
        
        # Balance sheet metrics
        if len(balance_sheets) >= 3:
            time_series_metrics["total_assets"] = self.time_series_analyzer.create_time_series(
                balance_sheets, "total_assets"
            )
            time_series_metrics["total_liabilities"] = self.time_series_analyzer.create_time_series(
                balance_sheets, "total_liabilities"
            )
            time_series_metrics["total_equity"] = self.time_series_analyzer.create_time_series(
                balance_sheets, "total_equity"
            )
        
        # Income statement metrics
        if len(income_statements) >= 3:
            time_series_metrics["revenue"] = self.time_series_analyzer.create_time_series(
                income_statements, "revenue"
            )
            time_series_metrics["net_income"] = self.time_series_analyzer.create_time_series(
                income_statements, "net_income"
            )
            time_series_metrics["gross_profit"] = self.time_series_analyzer.create_time_series(
                income_statements, "gross_profit"
            )
            time_series_metrics["operating_income"] = self.time_series_analyzer.create_time_series(
                income_statements, "operating_income"
            )
        
        # Cash flow metrics
        if len(cash_flow_statements) >= 3:
            time_series_metrics["operating_cash_flow"] = self.time_series_analyzer.create_time_series(
                cash_flow_statements, "operating_cash_flow"
            )
            time_series_metrics["free_cash_flow"] = self.time_series_analyzer.create_time_series(
                cash_flow_statements, "free_cash_flow"
            )
        
        # Analyze trends for all metrics
        trend_results = self.time_series_analyzer.analyze_multiple_metrics(time_series_metrics)
        
        # Store results
        self.analysis_results["trends"] = trend_results
        self.analysis_results["time_series"] = time_series_metrics
        
        return trend_results
    
    def generate_financial_insights(self) -> Dict[str, Any]:
        """
        Generate insights from financial analysis results.
        
        Returns:
            Dictionary of financial insights by category
        """
        insights = {
            "strengths": [],
            "weaknesses": [],
            "trends": [],
            "risks": [],
            "opportunities": []
        }
        
        # Extract ratios if available
        ratios = self.analysis_results.get("ratios", {})
        
        # Extract trends if available
        trends = self.analysis_results.get("trends", {})
        
        # Generate liquidity insights
        liquidity_ratios = ratios.get("liquidity", {})
        if liquidity_ratios:
            self._analyze_liquidity_ratios(liquidity_ratios, insights)
        
        # Generate profitability insights
        profitability_ratios = ratios.get("profitability", {})
        if profitability_ratios:
            self._analyze_profitability_ratios(profitability_ratios, insights)
        
        # Generate solvency insights
        solvency_ratios = ratios.get("solvency", {})
        if solvency_ratios:
            self._analyze_solvency_ratios(solvency_ratios, insights)
        
        # Generate trend insights
        if trends:
            self._analyze_trends(trends, insights)
        
        # Store insights
        self.analysis_results["insights"] = insights
        
        return insights
    
    def _filter_statements_by_company(
        self, 
        statements: List[FinancialStatement], 
        company_name: Optional[str]
    ) -> List[FinancialStatement]:
        """Filter statements by company name."""
        if not company_name:
            return statements
        
        return [s for s in statements if s.metadata.company_name == company_name]
    
    def _get_most_recent_statement(
        self, 
        statements: List[FinancialStatement]
    ) -> Optional[FinancialStatement]:
        """Get the most recent statement from a list."""
        if not statements:
            return None
        
        # Sort statements by date
        sorted_statements = self._sort_statements_by_time(statements)
        
        # Return the most recent one
        return sorted_statements[-1] if sorted_statements else None
    
    def _sort_statements_by_time(
        self, 
        statements: List[FinancialStatement]
    ) -> List[FinancialStatement]:
        """Sort statements by time (oldest to newest)."""
        def get_sort_key(statement):
            period = statement.metadata.period
            if period.end_date:
                return period.end_date
            elif period.fiscal_year:
                # Create approximate date
                quarter = period.fiscal_quarter or 4
                month = min(quarter * 3, 12)
                try:
                    return date(period.fiscal_year, month, 28)
                except ValueError:
                    return date(2000, 1, 1)  # Fallback
            else:
                return date(2000, 1, 1)  # Fallback for unknown dates
        
        return sorted(statements, key=get_sort_key)
    
    def _analyze_liquidity_ratios(
        self, 
        liquidity_ratios: Dict[str, FinancialRatio], 
        insights: Dict[str, List[str]]
    ):
        """Analyze liquidity ratios and generate insights."""
        current_ratio = liquidity_ratios.get("current_ratio")
        if current_ratio and current_ratio.value is not None:
            if current_ratio.value >= 2.0:
                insights["strengths"].append(
                    f"Strong liquidity position with current ratio of {current_ratio.format_value()}"
                )
            elif current_ratio.value >= 1.5:
                insights["strengths"].append(
                    f"Adequate liquidity with current ratio of {current_ratio.format_value()}"
                )
            elif current_ratio.value >= 1.0:
                insights["risks"].append(
                    f"Limited liquidity buffer with current ratio of {current_ratio.format_value()}"
                )
            else:
                insights["weaknesses"].append(
                    f"Potential liquidity issues with current ratio of {current_ratio.format_value()}, " 
                    f"which is below 1.0"
                )
        
        quick_ratio = liquidity_ratios.get("quick_ratio")
        if quick_ratio and quick_ratio.value is not None:
            if quick_ratio.value >= 1.0:
                insights["strengths"].append(
                    f"Strong acid-test ratio of {quick_ratio.format_value()}, indicating ability to "
                    f"cover short-term obligations without selling inventory"
                )
            elif quick_ratio.value < 0.7:
                insights["risks"].append(
                    f"Low quick ratio of {quick_ratio.format_value()}, suggesting potential "
                    f"difficulty meeting short-term obligations without selling inventory"
                )
    
    def _analyze_profitability_ratios(
        self, 
        profitability_ratios: Dict[str, FinancialRatio], 
        insights: Dict[str, List[str]]
    ):
        """Analyze profitability ratios and generate insights."""
        net_margin = profitability_ratios.get("net_profit_margin")
        if net_margin and net_margin.value is not None:
            if net_margin.value >= 0.15:
                insights["strengths"].append(
                    f"Excellent profitability with net margin of {net_margin.format_value()}"
                )
            elif net_margin.value >= 0.08:
                insights["strengths"].append(
                    f"Good profitability with net margin of {net_margin.format_value()}"
                )
            elif net_margin.value <= 0:
                insights["weaknesses"].append(
                    f"Negative net margin of {net_margin.format_value()}, indicating losses"
                )
        
        roe = profitability_ratios.get("return_on_equity")
        if roe and roe.value is not None:
            if roe.value >= 0.20:
                insights["strengths"].append(
                    f"Strong return on equity of {roe.format_value()}, indicating efficient use of capital"
                )
            elif roe.value <= 0.05 and roe.value > 0:
                insights["weaknesses"].append(
                    f"Low return on equity of {roe.format_value()}, suggesting inefficient use of capital"
                )
            elif roe.value <= 0:
                insights["weaknesses"].append(
                    f"Negative return on equity of {roe.format_value()}"
                )
    
    def _analyze_solvency_ratios(
        self, 
        solvency_ratios: Dict[str, FinancialRatio], 
        insights: Dict[str, List[str]]
    ):
        """Analyze solvency ratios and generate insights."""
        debt_to_equity = solvency_ratios.get("debt_to_equity")
        if debt_to_equity and debt_to_equity.value is not None:
            if debt_to_equity.value >= 2.0:
                insights["risks"].append(
                    f"High leverage with debt-to-equity ratio of {debt_to_equity.format_value()}"
                )
            elif debt_to_equity.value <= 0.5:
                insights["strengths"].append(
                    f"Conservative capital structure with debt-to-equity ratio of {debt_to_equity.format_value()}"
                )
        
        interest_coverage = solvency_ratios.get("interest_coverage_ratio")
        if interest_coverage and interest_coverage.value is not None:
            if interest_coverage.value >= 5.0:
                insights["strengths"].append(
                    f"Strong interest coverage of {interest_coverage.format_value()}, indicating low debt service risk"
                )
            elif interest_coverage.value <= 2.0:
                insights["risks"].append(
                    f"Low interest coverage of {interest_coverage.format_value()}, suggesting potential "
                    f"difficulty meeting interest obligations"
                )
    
    def _analyze_trends(
        self, 
        trends: Dict[str, Optional[TrendAnalysisResult]], 
        insights: Dict[str, List[str]]
    ):
        """Analyze trends and generate insights."""
        # Revenue trend
        revenue_trend = trends.get("revenue")
        if revenue_trend:
            if revenue_trend.direction == TrendDirection.INCREASING and revenue_trend.is_statistically_significant:
                insights["trends"].append(
                    f"Revenue shows a statistically significant increasing trend with {revenue_trend.growth_rate:.1%} growth rate"
                )
            elif revenue_trend.direction == TrendDirection.DECREASING and revenue_trend.is_statistically_significant:
                insights["risks"].append(
                    f"Revenue shows a statistically significant decreasing trend with {abs(revenue_trend.growth_rate):.1%} decline rate"
                )
        
        # Net income trend
        net_income_trend = trends.get("net_income")
        if net_income_trend:
            if net_income_trend.direction == TrendDirection.INCREASING and net_income_trend.is_statistically_significant:
                insights["trends"].append(
                    f"Net income shows a statistically significant increasing trend"
                )
                if revenue_trend and revenue_trend.direction != TrendDirection.INCREASING:
                    insights["strengths"].append(
                        "Improving profitability despite stable/declining revenue, indicating efficiency improvements"
                    )
            elif net_income_trend.direction == TrendDirection.DECREASING and net_income_trend.is_statistically_significant:
                insights["risks"].append(
                    f"Net income shows a statistically significant decreasing trend"
                )
        
        # Operating cash flow trend
        ocf_trend = trends.get("operating_cash_flow")
        if ocf_trend:
            if ocf_trend.direction == TrendDirection.INCREASING and ocf_trend.is_statistically_significant:
                insights["strengths"].append(
                    f"Operating cash flow shows a positive trend, indicating strong cash generation"
                )
            elif ocf_trend.direction == TrendDirection.DECREASING and ocf_trend.is_statistically_significant:
                insights["risks"].append(
                    f"Operating cash flow shows a declining trend, indicating potential cash flow issues"
                )
        
        # Compare net income and operating cash flow trends
        if net_income_trend and ocf_trend:
            if (net_income_trend.direction == TrendDirection.INCREASING and 
                ocf_trend.direction != TrendDirection.INCREASING):
                insights["risks"].append(
                    "Net income is increasing, but operating cash flow is not, suggesting potential earnings quality issues"
                )
            elif (net_income_trend.direction != TrendDirection.INCREASING and 
                  ocf_trend.direction == TrendDirection.INCREASING):
                insights["opportunities"].append(
                    "Operating cash flow is improving despite flat/declining net income, indicating potential for future profit improvement"
                )


# Initialize the financial analyzer engine
financial_analyzer = FinancialAnalyzerEngine()
