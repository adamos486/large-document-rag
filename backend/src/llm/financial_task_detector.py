"""
Financial Task Detector for identifying the type of financial task in a query.

This module provides specialized detection of financial task types to enable
more intelligent routing between OpenAI and Anthropic models based on task complexity,
financial domain specificity, and other characteristics.
"""

import re
import logging
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class FinancialTaskType(str, Enum):
    """Detailed financial task types for specialized LLM routing."""
    # Analysis tasks
    RATIO_ANALYSIS = "ratio_analysis"
    TREND_ANALYSIS = "trend_analysis"
    VALUATION = "valuation"
    RISK_ASSESSMENT = "risk_assessment"
    FORECASTING = "forecasting"
    
    # Document handling tasks
    DOCUMENT_EXTRACTION = "document_extraction"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    TABLE_INTERPRETATION = "table_interpretation"
    
    # Financial statement tasks
    INCOME_STATEMENT_ANALYSIS = "income_statement_analysis"
    BALANCE_SHEET_ANALYSIS = "balance_sheet_analysis"
    CASH_FLOW_ANALYSIS = "cash_flow_analysis"
    
    # Due diligence tasks
    DUE_DILIGENCE = "due_diligence"
    RED_FLAG_IDENTIFICATION = "red_flag_identification"
    COMPLIANCE_CHECK = "compliance_check"
    
    # General tasks
    QUESTION_ANSWERING = "question_answering"
    GENERAL_FINANCIAL = "general_financial"
    NON_FINANCIAL = "non_financial"


class FinancialTaskDetector:
    """Detector for identifying financial task types from queries."""
    
    def __init__(self):
        """Initialize the financial task detector with task patterns."""
        # Task pattern dictionaries
        self.task_patterns: Dict[str, List[str]] = {
            FinancialTaskType.RATIO_ANALYSIS: [
                r'\bratio analysis\b', r'\bfinancial ratio', r'\bprofitability ratio', r'\bliquidity ratio',
                r'\bdebt to equity\b', r'\bcurrent ratio\b', r'\bquick ratio\b', r'\broi\b', r'\broe\b',
                r'\bebitda margin\b', r'\bgross margin\b', r'\boperating margin\b'
            ],
            FinancialTaskType.TREND_ANALYSIS: [
                r'\btrend analysis\b', r'\bhistorical trend', r'\bfinancial trend', r'\btrend over time\b',
                r'\bgrowth rate', r'\byear over year\b', r'\bquarter over quarter\b', r'\bcagr\b'
            ],
            FinancialTaskType.VALUATION: [
                r'\bvaluation\b', r'\bcompany value\b', r'\bvaluation model', r'\bdcf\b', r'\bdiscounted cash flow\b',
                r'\bprice to earnings\b', r'\bp/e\b', r'\bebitda multiple\b', r'\bev/ebitda\b', r'\bfair value\b',
                r'\bintrinsic value\b', r'\bcomparable companies\b', r'\bcomps\b'
            ],
            FinancialTaskType.RISK_ASSESSMENT: [
                r'\brisk assessment\b', r'\brisk analysis\b', r'\brisk exposure\b', r'\bfinancial risk\b',
                r'\bcredit risk\b', r'\bmarket risk\b', r'\bliquidity risk\b', r'\boperational risk\b',
                r'\brisk factors\b', r'\brisk mitigation\b'
            ],
            FinancialTaskType.FORECASTING: [
                r'\bforecast\b', r'\bprojection\b', r'\bprojected\b', r'\bestimate future\b', r'\bfuture performance\b',
                r'\bfinancial model\b', r'\bprediction\b', r'\bgrowth projection\b', r'\bestimated revenue\b'
            ],
            FinancialTaskType.DOCUMENT_EXTRACTION: [
                r'\bextract\b', r'\bpull\b', r'\bfind\b.*\bfrom.*\bdocument', r'\bdata extraction\b',
                r'\bextract information\b', r'\bpull data\b', r'\bget.*\bfrom.*\breport'
            ],
            FinancialTaskType.DOCUMENT_SUMMARIZATION: [
                r'\bsummarize\b', r'\bsummary\b', r'\bkey points\b', r'\bmain findings\b', r'\boverview\b',
                r'\bhighlight\b', r'\bcondense\b', r'\bbrief\b'
            ],
            FinancialTaskType.TABLE_INTERPRETATION: [
                r'\btable\b', r'\bfinancial table\b', r'\bspreadsheet\b', r'\bfinanical data\b', r'\brows?\b.*\bcolumns?\b',
                r'\bdata in the table\b', r'\binterpret the table\b', r'\btabular data\b', r'\bexcel\b'
            ],
            FinancialTaskType.INCOME_STATEMENT_ANALYSIS: [
                r'\bincome statement\b', r'\bprofit and loss\b', r'\bp&l\b', r'\brevenue\b', r'\bsales\b',
                r'\bgross profit\b', r'\boperating expense\b', r'\bnet income\b', r'\bebitda\b', r'\boperating income\b',
                r'\bcost of goods sold\b', r'\bcogs\b', r'\bexpense\b'
            ],
            FinancialTaskType.BALANCE_SHEET_ANALYSIS: [
                r'\bbalance sheet\b', r'\bassets?\b', r'\bliabilities?\b', r'\bequity\b', r'\bdebt\b',
                r'\bcurrent assets?\b', r'\bcurrent liabilities?\b', r'\binventory\b', r'\baccounts receivable\b',
                r'\baccounts payable\b', r'\blong-term debt\b', r'\bshareholders equity\b', r'\bbook value\b'
            ],
            FinancialTaskType.CASH_FLOW_ANALYSIS: [
                r'\bcash flow\b', r'\bcash from operations\b', r'\boperating cash flow\b', r'\bfree cash flow\b',
                r'\binvesting activities\b', r'\bfinancing activities\b', r'\bcapital expenditure\b', r'\bcapex\b',
                r'\bdividend\b', r'\brepurchase\b', r'\bcash on hand\b'
            ],
            FinancialTaskType.DUE_DILIGENCE: [
                r'\bdue diligence\b', r'\bm&a\b', r'\bmerger and acquisition\b', r'\btarget company\b',
                r'\bacquisition target\b', r'\bcomprehensive review\b', r'\bfinancial due diligence\b',
                r'\blegal due diligence\b', r'\boperational due diligence\b'
            ],
            FinancialTaskType.RED_FLAG_IDENTIFICATION: [
                r'\bred flags?\b', r'\bwarning signs?\b', r'\bfinanical irregularities\b', r'\baccounting issues\b',
                r'\bfraud\b', r'\bmisstatement\b', r'\baccounting manipulation\b', r'\bfinancial fraud\b',
                r'\birreguarities\b', r'\baudit issues\b', r'\bmaterial weakness\b'
            ],
            FinancialTaskType.COMPLIANCE_CHECK: [
                r'\bcompliance\b', r'\bregulatory\b', r'\bregulation\b', r'\bfinancial regulations\b', r'\bcompliant\b',
                r'\baccounting standards\b', r'\bgaap\b', r'\bifrs\b', r'\bsec\b', r'\bsarbanes-oxley\b', r'\bsox\b',
                r'\blegal requirement\b'
            ],
            FinancialTaskType.QUESTION_ANSWERING: [
                r'\bwhat\b.*\?', r'\bhow\b.*\?', r'\bwhen\b.*\?', r'\bwhere\b.*\?', r'\bwhy\b.*\?',
                r'\bcan you tell me\b', r'\bexplain\b', r'\bcould you provide\b'
            ],
            FinancialTaskType.GENERAL_FINANCIAL: [
                r'\bfinance\b', r'\bfinancial\b', r'\bmoney\b', r'\bcapital\b', r'\binvestment\b',
                r'\bdollar\b', r'\beuro\b', r'\bcurrency\b', r'\beconomic\b', r'\bmarket\b'
            ]
        }
        
        # Compiled regex patterns for faster matching
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {
            task_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for task_type, patterns in self.task_patterns.items()
        }
        
        # Task complexity indicators
        self.complex_task_indicators = [
            r'\bcompare\b', r'\banalyze\b', r'\bcorrelation\b', r'\bcausation\b',
            r'\bimpact of\b', r'\brelationship between\b', r'\btradeoff\b', r'\bscenario analysis\b',
            r'\bsensitivity analysis\b', r'\bwhat-if\b', r'\bcomplex\b', r'\badvanced\b',
            r'\bsophisticated\b', r'\bdetailed\b', r'\bnuanced\b', r'\bmultifaceted\b'
        ]
        self.compiled_complexity_indicators = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.complex_task_indicators
        ]
        
        # Financial entity keywords that suggest financial context
        self.financial_entities = [
            r'\b\$\d+\b', r'\bdollars\b', r'\beuros\b', r'\byuans?\b', r'\byens?\b',
            r'\bpounds\b', r'\bcurrency\b', r'\bpercentage\b', r'\bcompany\b', r'\bcorporation\b',
            r'\benterprise\b', r'\bfirm\b', r'\borganization\b', r'\bbusiness\b'
        ]
        self.compiled_financial_entities = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.financial_entities
        ]
        
        # Map task types to preferred LLM providers
        # This could also be loaded from configuration
        self.task_llm_mapping = {
            # Tasks better suited for Anthropic Claude
            FinancialTaskType.RATIO_ANALYSIS: "anthropic",
            FinancialTaskType.VALUATION: "anthropic",
            FinancialTaskType.RISK_ASSESSMENT: "anthropic",
            FinancialTaskType.DUE_DILIGENCE: "anthropic",
            FinancialTaskType.RED_FLAG_IDENTIFICATION: "anthropic",
            FinancialTaskType.COMPLIANCE_CHECK: "anthropic",
            
            # Tasks better suited for OpenAI
            FinancialTaskType.DOCUMENT_EXTRACTION: "openai",
            FinancialTaskType.DOCUMENT_SUMMARIZATION: "openai",
            FinancialTaskType.TABLE_INTERPRETATION: "openai",
            FinancialTaskType.QUESTION_ANSWERING: "openai",
            
            # Tasks that could go either way
            FinancialTaskType.TREND_ANALYSIS: "hybrid",
            FinancialTaskType.FORECASTING: "hybrid",
            FinancialTaskType.INCOME_STATEMENT_ANALYSIS: "hybrid",
            FinancialTaskType.BALANCE_SHEET_ANALYSIS: "hybrid",
            FinancialTaskType.CASH_FLOW_ANALYSIS: "hybrid",
            FinancialTaskType.GENERAL_FINANCIAL: "openai",
            FinancialTaskType.NON_FINANCIAL: "openai"
        }
    
    def detect_task_type(self, query: str) -> Tuple[str, Dict[str, float]]:
        """
        Detect the financial task type from a query.
        
        Args:
            query: The query text to analyze
            
        Returns:
            Tuple[str, Dict[str, float]]: The detected task type and confidence scores
        """
        # Default to general question if nothing else matches
        default_task = FinancialTaskType.QUESTION_ANSWERING
        
        # Check if query contains any financial terms at all
        has_financial_context = False
        for pattern in self.compiled_financial_entities:
            if pattern.search(query):
                has_financial_context = True
                break
        
        if not has_financial_context:
            return FinancialTaskType.NON_FINANCIAL, {FinancialTaskType.NON_FINANCIAL: 1.0}
        
        # Count matches for each task type
        task_matches: Dict[str, int] = {task_type: 0 for task_type in self.task_patterns.keys()}
        
        for task_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(query)
                task_matches[task_type] += len(matches)
        
        # Calculate confidence scores
        total_matches = sum(task_matches.values())
        if total_matches == 0:
            # If no specific matches, default to general financial
            return FinancialTaskType.GENERAL_FINANCIAL, {FinancialTaskType.GENERAL_FINANCIAL: 1.0}
        
        confidence_scores = {
            task_type: count / total_matches for task_type, count in task_matches.items() if count > 0
        }
        
        # Check for complex task indicators
        complexity_score = 0
        for pattern in self.compiled_complexity_indicators:
            if pattern.search(query):
                complexity_score += 1
        
        # Get the task type with the highest score
        best_task = max(confidence_scores.items(), key=lambda x: x[1])[0]
        
        # For tasks with similar scores, prefer the more complex one if complexity indicators are present
        if complexity_score >= 2:
            # Prioritize more complex tasks when complexity is detected
            complex_tasks = [
                FinancialTaskType.VALUATION, 
                FinancialTaskType.RISK_ASSESSMENT,
                FinancialTaskType.DUE_DILIGENCE,
                FinancialTaskType.RED_FLAG_IDENTIFICATION
            ]
            
            for task in complex_tasks:
                if task in confidence_scores and confidence_scores[task] >= 0.5 * confidence_scores[best_task]:
                    best_task = task
                    break
        
        logger.debug(f"Detected task type: {best_task} with confidence: {confidence_scores.get(best_task, 0)}")
        return best_task, confidence_scores
    
    def get_llm_provider_for_task(self, query: str, override_provider: Optional[str] = None) -> str:
        """
        Determine the best LLM provider for a given query based on task detection.
        
        Args:
            query: The query text to analyze
            override_provider: Optional override to force a specific provider
            
        Returns:
            str: The recommended LLM provider ("openai", "anthropic", or "hybrid")
        """
        if override_provider:
            return override_provider
        
        # Detect the task type
        task_type, confidence_scores = self.detect_task_type(query)
        
        # Get the preferred provider for this task
        provider = self.task_llm_mapping.get(task_type, "openai")
        
        # For lower confidence scores or multiple high-scoring tasks, use hybrid approach
        if confidence_scores.get(task_type, 0) < 0.4:
            # Not very confident, use hybrid
            provider = "hybrid"
        
        # Check for conflicting high-confidence tasks that suggest different providers
        high_confidence_tasks = [
            task for task, score in confidence_scores.items() 
            if score >= 0.25 and self.task_llm_mapping.get(task) != provider
        ]
        
        if high_confidence_tasks:
            # Multiple conflicting task types detected, use hybrid
            provider = "hybrid"
        
        logger.debug(f"Selected LLM provider {provider} for task type {task_type}")
        return provider
    
    def extract_financial_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from a query.
        
        Args:
            query: The query text to analyze
            
        Returns:
            Dict[str, List[str]]: Dictionary of entity types to extracted values
        """
        entities = {
            "monetary_values": [],
            "percentages": [],
            "companies": [],
            "statements": [],
            "ratios": []
        }
        
        # Extract monetary values
        money_pattern = r'\$\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|m|b|k))?'
        entities["monetary_values"] = re.findall(money_pattern, query, re.IGNORECASE)
        
        # Extract percentages
        percent_pattern = r'\d+(?:\.\d+)?\s?%'
        entities["percentages"] = re.findall(percent_pattern, query)
        
        # Extract statement types
        statement_pattern = r'\b(?:income statement|balance sheet|cash flow|p&l|profit and loss)\b'
        entities["statements"] = re.findall(statement_pattern, query, re.IGNORECASE)
        
        # Extract financial ratios
        ratio_pattern = r'\b(?:roi|roe|roa|ebitda|gross margin|net margin|current ratio|debt[- ]to[- ]equity|p/e|ev/ebitda)\b'
        entities["ratios"] = re.findall(ratio_pattern, query, re.IGNORECASE)
        
        # Clean up empty lists
        return {k: v for k, v in entities.items() if v}


# Singleton instance
financial_task_detector = FinancialTaskDetector()
