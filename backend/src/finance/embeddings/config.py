"""
Configuration for the financial embeddings system.

This module defines the configuration settings for the custom financial embeddings model,
including model architecture, training parameters, and domain-specific adaptations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json
import os

from src.config.config import settings, FinancialEntityType


@dataclass
class EmbeddingModelConfig:
    """Configuration for the financial embeddings model."""
    
    # Base model configuration
    base_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    output_dimension: int = 768
    projection_dimension: int = 768  # For financial domain projection - match output for tests
    max_seq_length: int = 512
    
    # Tokenizer settings
    special_tokens: List[str] = field(default_factory=lambda: [
        "[COMPANY]", "[/COMPANY]",
        "[METRIC]", "[/METRIC]",
        "[RATIO]", "[/RATIO]",
        "[ACCOUNT]", "[/ACCOUNT]",
        "[PERIOD]", "[/PERIOD]",
        "[TABLE]", "[/TABLE]", 
        "[ROW]", "[/ROW]",
        "[CELL]", "[/CELL]",
        "[STATEMENT]", "[/STATEMENT]",
        "[BALANCE_SHEET]", "[/BALANCE_SHEET]",
        "[INCOME_STATEMENT]", "[/INCOME_STATEMENT]",
        "[CASH_FLOW]", "[/CASH_FLOW]"
    ])
    
    # Entity weighting configuration
    entity_weights: Dict[str, float] = field(default_factory=lambda: {
        FinancialEntityType.COMPANY.value: 1.5,
        FinancialEntityType.SUBSIDIARY.value: 1.3,
        FinancialEntityType.METRIC.value: 1.8,
        FinancialEntityType.RATIO.value: 1.7,
        FinancialEntityType.STATEMENT.value: 1.6,
        FinancialEntityType.ACCOUNT.value: 1.4,
        FinancialEntityType.PERIOD.value: 1.2,
        FinancialEntityType.CURRENCY.value: 1.1,
        FinancialEntityType.REGULATION.value: 1.3,
        FinancialEntityType.RISK.value: 1.6,
    })
    
    # Projection layer configuration
    use_projection_layer: bool = True
    projection_layer_activation: str = "tanh"
    projection_layer_dropout: float = 0.1
    
    # Cache settings
    embedding_cache_size: int = 10000
    use_disk_cache: bool = True
    cache_dir: Path = settings.FINANCIAL_CACHE_DIR / "embeddings"
    
    # Pooling strategy
    pooling_mode: str = "mean"  # One of: mean, max, cls
    pooling_with_attention: bool = True
    
    # Domain adaptation settings
    use_domain_adaptation: bool = True
    domain_loss_weight: float = 0.3
    contrastive_loss_margin: float = 0.5
    
    # Financial-specific settings
    financial_terms_path: Path = settings.FINANCIAL_MODEL_DIR / "financial_terms.json"
    statement_structure_path: Path = settings.FINANCIAL_MODEL_DIR / "statement_structure.json"
    
    # Training settings (for fine-tuning)
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    epochs: int = 3
    evaluation_steps: int = 500
    
    def __post_init__(self):
        """Create necessary directories and load external resources."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create financial terms file if it doesn't exist
        if not self.financial_terms_path.exists():
            self._create_initial_financial_terms()
            
        # Create statement structure file if it doesn't exist
        if not self.statement_structure_path.exists():
            self._create_initial_statement_structure()
    
    def _create_initial_financial_terms(self):
        """Create initial financial terms dictionary."""
        terms = {
            "metrics": [
                "Revenue", "Net Income", "EBITDA", "Gross Profit", "Operating Income",
                "Total Assets", "Total Liabilities", "Shareholders' Equity", "Cash Flow",
                "Free Cash Flow", "Capital Expenditure", "Dividend", "Interest Expense",
                "Tax Expense", "Depreciation", "Amortization", "Inventory", "Accounts Receivable",
                "Accounts Payable", "Long-term Debt", "Current Assets", "Current Liabilities"
            ],
            "ratios": [
                "P/E Ratio", "EPS", "ROI", "ROE", "ROA", "Current Ratio", "Quick Ratio",
                "Debt-to-Equity", "Gross Margin", "Operating Margin", "Net Profit Margin",
                "Asset Turnover", "Inventory Turnover", "Interest Coverage", "Dividend Yield",
                "Payout Ratio", "Cash Ratio", "ROIC", "Debt Ratio", "Acid-Test Ratio"
            ],
            "statements": [
                "Balance Sheet", "Income Statement", "Cash Flow Statement", "Statement of Changes in Equity",
                "Notes to Financial Statements", "Management Discussion and Analysis", "Audit Report",
                "Annual Report", "Quarterly Report", "10-K", "10-Q", "8-K", "Prospectus", "Proxy Statement"
            ],
            "regulations": [
                "GAAP", "IFRS", "SOX", "FASB", "SEC", "IAS", "ASC", "Dodd-Frank", "Basel III",
                "PCAOB", "AICPA", "IASB", "ESMA", "FCA", "FINRA", "IRS", "CRA", "HMRC"
            ]
        }
        
        self.financial_terms_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.financial_terms_path, 'w') as f:
            json.dump(terms, f, indent=2)
            
    def _create_initial_statement_structure(self):
        """Create initial statement structure dictionary."""
        structure = {
            "balance_sheet": {
                "assets": {
                    "current_assets": [
                        "cash_and_cash_equivalents",
                        "short_term_investments",
                        "accounts_receivable",
                        "inventory",
                        "prepaid_expenses",
                        "other_current_assets"
                    ],
                    "non_current_assets": [
                        "property_plant_equipment",
                        "goodwill",
                        "intangible_assets",
                        "long_term_investments",
                        "deferred_tax_assets",
                        "other_non_current_assets"
                    ]
                },
                "liabilities": {
                    "current_liabilities": [
                        "accounts_payable",
                        "short_term_debt",
                        "current_portion_of_long_term_debt",
                        "accrued_expenses",
                        "deferred_revenue",
                        "income_taxes_payable",
                        "other_current_liabilities"
                    ],
                    "non_current_liabilities": [
                        "long_term_debt",
                        "pension_liabilities",
                        "deferred_tax_liabilities",
                        "other_non_current_liabilities"
                    ]
                },
                "equity": [
                    "common_stock",
                    "preferred_stock",
                    "additional_paid_in_capital",
                    "retained_earnings",
                    "treasury_stock",
                    "accumulated_other_comprehensive_income",
                    "non_controlling_interest"
                ]
            },
            "income_statement": [
                "revenue",
                "cost_of_goods_sold",
                "gross_profit",
                "operating_expenses",
                "research_and_development",
                "selling_general_and_administrative",
                "depreciation_and_amortization",
                "operating_income",
                "non_operating_income",
                "interest_expense",
                "income_before_tax",
                "income_tax_expense",
                "net_income",
                "earnings_per_share_basic",
                "earnings_per_share_diluted",
                "weighted_average_shares_outstanding_basic",
                "weighted_average_shares_outstanding_diluted"
            ],
            "cash_flow_statement": {
                "operating_activities": [
                    "net_income",
                    "depreciation_and_amortization",
                    "stock_based_compensation",
                    "changes_in_working_capital",
                    "accounts_receivable_change",
                    "inventory_change",
                    "accounts_payable_change",
                    "other_operating_activities",
                    "net_cash_from_operating_activities"
                ],
                "investing_activities": [
                    "capital_expenditures",
                    "acquisitions",
                    "purchases_of_investments",
                    "sales_maturities_of_investments",
                    "other_investing_activities",
                    "net_cash_from_investing_activities"
                ],
                "financing_activities": [
                    "debt_issuance",
                    "debt_repayment",
                    "common_stock_issued",
                    "common_stock_repurchased",
                    "dividends_paid",
                    "other_financing_activities",
                    "net_cash_from_financing_activities"
                ],
                "supplemental": [
                    "effect_of_forex_changes_on_cash",
                    "net_change_in_cash",
                    "cash_at_beginning_of_period",
                    "cash_at_end_of_period"
                ]
            }
        }
        
        self.statement_structure_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.statement_structure_path, 'w') as f:
            json.dump(structure, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EmbeddingModelConfig":
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if key in config.__dict__:
                if isinstance(config.__dict__[key], Path) and isinstance(value, str):
                    setattr(config, key, Path(value))
                else:
                    setattr(config, key, value)
        return config
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save config to file."""
        if path is None:
            path = settings.FINANCIAL_MODEL_DIR / "embedding_config.json"
            
        if isinstance(path, str):
            path = Path(path)
            
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "EmbeddingModelConfig":
        """Load config from file."""
        if path is None:
            path = settings.FINANCIAL_MODEL_DIR / "embedding_config.json"
            
        if isinstance(path, str):
            path = Path(path)
            
        if not path.exists():
            return cls()
            
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)


# Create default embedding model configuration
embedding_config = EmbeddingModelConfig()

# Ensure configuration is saved
embedding_config.save()
