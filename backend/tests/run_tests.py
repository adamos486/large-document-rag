#!/usr/bin/env python
"""
Test runner for the financial document RAG system.

This script discovers and runs all tests in the project, providing
detailed output about test results and integration issues.
"""

import os
import sys
import unittest
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def discover_tests(test_dir: Optional[str] = None, pattern: str = "test_*.py") -> unittest.TestSuite:
    """
    Discover all tests in the specified directory.
    
    Args:
        test_dir: Directory to search for tests. If None, use the 'tests' directory.
        pattern: Pattern to match test files.
    
    Returns:
        TestSuite containing all discovered tests.
    """
    if test_dir is None:
        test_dir = os.path.dirname(__file__)
    
    logger.info(f"Discovering tests in {test_dir} matching pattern {pattern}")
    return unittest.defaultTestLoader.discover(test_dir, pattern=pattern)


def run_tests(test_suite: unittest.TestSuite, verbosity: int = 2) -> unittest.TestResult:
    """
    Run the provided test suite.
    
    Args:
        test_suite: TestSuite to run.
        verbosity: Verbosity level for test output.
    
    Returns:
        TestResult containing test results.
    """
    runner = unittest.TextTestRunner(verbosity=verbosity)
    logger.info("Running tests...")
    return runner.run(test_suite)


def analyze_test_failures(result: unittest.TestResult) -> Tuple[List[str], List[str]]:
    """
    Analyze test failures to identify patterns and potential issues.
    
    Args:
        result: TestResult from running tests.
    
    Returns:
        Tuple containing lists of detected dependency issues and integration issues.
    """
    dependency_issues = []
    integration_issues = []
    
    # Analyze errors and failures
    all_problems = [(test, error) for test, error in result.errors]
    all_problems.extend([(test, error) for test, error in result.failures])
    
    for test, error_text in all_problems:
        # Extract test class and method name
        test_id = str(test)
        
        # Check for dependency issues
        if "ImportError" in error_text or "ModuleNotFoundError" in error_text:
            dependency_issues.append(f"Dependency issue in {test_id}: {error_text.splitlines()[0]}")
        
        # Check for integration issues
        elif any(term in error_text for term in [
            "AttributeError", "IntegrationError", "ConfigurationError", 
            "settings", "configuration", "LLM_PROVIDER", "vector_store"
        ]):
            integration_issues.append(f"Integration issue in {test_id}: {error_text.splitlines()[0]}")
    
    return dependency_issues, integration_issues


def create_test_report(result: unittest.TestResult, dependency_issues: List[str], integration_issues: List[str]) -> str:
    """
    Create a detailed test report.
    
    Args:
        result: TestResult from running tests.
        dependency_issues: List of identified dependency issues.
        integration_issues: List of identified integration issues.
    
    Returns:
        Formatted test report.
    """
    report = []
    report.append("\n" + "=" * 80)
    report.append("TEST REPORT")
    report.append("=" * 80)
    
    # Basic statistics
    total_tests = result.testsRun
    passed = total_tests - len(result.errors) - len(result.failures)
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    
    report.append(f"\nRan {total_tests} tests with {passed} passes ({success_rate:.1f}% success rate)")
    report.append(f"Failures: {len(result.failures)}")
    report.append(f"Errors: {len(result.errors)}")
    report.append(f"Skipped: {len(result.skipped)}")
    
    # Dependency issues
    if dependency_issues:
        report.append("\nDEPENDENCY ISSUES:")
        for issue in dependency_issues:
            report.append(f"  - {issue}")
    
    # Integration issues
    if integration_issues:
        report.append("\nINTEGRATION ISSUES:")
        for issue in integration_issues:
            report.append(f"  - {issue}")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS:")
    if not dependency_issues and not integration_issues and success_rate == 100:
        report.append("  - All tests passed! System appears to be well-integrated.")
    else:
        if dependency_issues:
            report.append("  - Check requirements.txt and install missing dependencies")
            report.append("  - Verify compatibility between package versions")
        
        if integration_issues:
            report.append("  - Review configuration settings across all modules")
            report.append("  - Ensure consistent attribute names between old and new code")
            report.append("  - Verify integration points between financial system and existing RAG system")
    
    report.append("\n" + "=" * 80)
    return "\n".join(report)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for the financial document RAG system")
    parser.add_argument("--test-dir", help="Directory to search for tests")
    parser.add_argument("--pattern", default="test_*.py", help="Pattern to match test files")
    parser.add_argument("--verbosity", type=int, default=2, help="Verbosity level for test output")
    parser.add_argument("--financial-only", action="store_true", help="Run only financial-related tests")
    return parser.parse_args()


def main():
    """Run tests and analyze results."""
    args = parse_arguments()
    
    # Determine test directory
    test_dir = args.test_dir
    if test_dir is None:
        test_dir = os.path.dirname(__file__)
    
    # Set pattern for financial-only tests
    pattern = "test_*.py"
    if args.financial_only:
        pattern = "test_financial*.py"
    elif args.pattern:
        pattern = args.pattern
    
    # Discover and run tests
    test_suite = discover_tests(test_dir, pattern)
    result = run_tests(test_suite, args.verbosity)
    
    # Analyze results
    dependency_issues, integration_issues = analyze_test_failures(result)
    
    # Print report
    print(create_test_report(result, dependency_issues, integration_issues))
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
