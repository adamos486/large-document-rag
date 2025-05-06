"""
Tests for the financial system configuration.

These tests verify that the configuration settings are correctly defined,
loaded, and accessible to all components of the financial system.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from src.config.config import settings, FinancialEntityType, FinancialTaskType, ChunkingStrategy, LLMProvider


class TestFinancialConfiguration(unittest.TestCase):
    """Test suite for financial configuration settings."""
    
    def test_financial_config_paths(self):
        """Test that financial paths are correctly defined."""
        # Verify that all required paths are set
        self.assertIsNotNone(settings.FINANCIAL_DATA_DIR)
        self.assertIsNotNone(settings.FINANCIAL_CACHE_DIR)
        self.assertIsNotNone(settings.FINANCIAL_MODEL_DIR)
        
        # Verify that paths are of the correct type
        self.assertIsInstance(settings.FINANCIAL_DATA_DIR, Path)
        self.assertIsInstance(settings.FINANCIAL_CACHE_DIR, Path)
        self.assertIsInstance(settings.FINANCIAL_MODEL_DIR, Path)
    
    def test_financial_settings_values(self):
        """Test that financial settings have appropriate values."""
        # Check boolean settings
        self.assertIsInstance(settings.FINANCIAL_CUSTOM_EMBEDDING, bool)
        self.assertIsInstance(settings.FINANCIAL_PROJECTION_LAYER, bool)
        self.assertIsInstance(settings.FINANCIAL_ENTITY_WEIGHTING, bool)
        self.assertIsInstance(settings.FINANCIAL_USE_OCR, bool)
        
        # Check numeric settings
        self.assertIsInstance(settings.FINANCIAL_MAX_TABLE_SIZE, int)
        self.assertIsInstance(settings.FINANCIAL_MIN_CONFIDENCE, float)
        self.assertGreaterEqual(settings.FINANCIAL_MIN_CONFIDENCE, 0.0)
        self.assertLessEqual(settings.FINANCIAL_MIN_CONFIDENCE, 1.0)
        
        # Check enum-based settings
        self.assertIn(settings.FINANCIAL_CHUNK_STRATEGY, 
                      [strategy.value for strategy in ChunkingStrategy])
    
    def test_financial_entity_types(self):
        """Test that financial entity types are correctly defined."""
        # Verify that all expected entity types exist
        expected_types = [
            "company", "subsidiary", "metric", "ratio", "statement", 
            "account", "period", "currency", "regulation", "risk"
        ]
        
        for entity_type in expected_types:
            self.assertIn(entity_type, [e.value for e in FinancialEntityType])
    
    def test_financial_task_types(self):
        """Test that financial task types are correctly defined."""
        # Verify that all expected task types exist
        expected_tasks = [
            "ratio_analysis", "trend_analysis", "valuation", "due_diligence",
            "risk_assessment", "statement_analysis", "scenario_analysis",
            "forecasting", "peer_comparison", "custom"
        ]
        
        for task_type in expected_tasks:
            self.assertIn(task_type, [t.value for t in FinancialTaskType])
    
    def test_chunking_strategy_selection(self):
        """Test that the correct chunking strategy is selected for different document types."""
        # Test financial document types
        self.assertEqual(
            settings.get_chunking_strategy("financial"),
            settings.FINANCIAL_CHUNK_STRATEGY
        )
        self.assertEqual(
            settings.get_chunking_strategy("finance"),
            settings.FINANCIAL_CHUNK_STRATEGY
        )
        
        # Test legacy document types (for backward compatibility)
        self.assertEqual(
            settings.get_chunking_strategy("gis"),
            settings.GIS_CHUNK_STRATEGY
        )
        self.assertEqual(
            settings.get_chunking_strategy("cad"),
            settings.CAD_CHUNK_STRATEGY
        )
        
        # Test unknown document type (should default to statement preserving)
        self.assertEqual(
            settings.get_chunking_strategy("unknown"),
            ChunkingStrategy.STATEMENT_PRESERVING.value
        )
    
    def test_llm_provider_selection(self):
        """Test that the correct LLM provider is selected for different tasks."""
        # Test with default provider
        with patch.object(settings, 'DEFAULT_LLM_PROVIDER', LLMProvider.OPENAI.value):
            # When not in hybrid mode, should always return the default provider
            self.assertEqual(
                settings.get_llm_for_task("ratio_analysis"),
                LLMProvider.OPENAI.value
            )
            self.assertEqual(
                settings.get_llm_for_task("unknown_task"),
                LLMProvider.OPENAI.value
            )
        
        # Test with hybrid provider
        with patch.object(settings, 'DEFAULT_LLM_PROVIDER', LLMProvider.HYBRID.value):
            # Should return the provider specified for the task
            self.assertEqual(
                settings.get_llm_for_task(FinancialTaskType.RATIO_ANALYSIS.value),
                settings.HYBRID_LLM_TASKS[FinancialTaskType.RATIO_ANALYSIS.value]
            )
            
            # Should return default provider for unknown tasks
            self.assertEqual(
                settings.get_llm_for_task("unknown_task"),
                settings.HYBRID_LLM_TASKS["default"]
            )


if __name__ == '__main__':
    unittest.main()
