"""
Dependency checker for tests

This utility helps verify that all required dependencies are installed
and provides guidance on how to fix missing dependencies.
"""

import importlib
import logging
import sys
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define dependency groups
DEPENDENCY_GROUPS = {
    "core": [
        "numpy", 
        "pandas", 
        "tqdm"
    ],
    "vector_store": [
        "chromadb",
        "hnswlib"
    ],
    "embedding": [
        "torch",
        "transformers",
        "sentence_transformers"
    ],
    "nlp": [
        "spacy",
        "nltk"
    ],
    "document_processing": [
        "unstructured",
        "pdfminer",
        "pdfminer.six",
        "python-docx",
        "openpyxl"
    ],
    "visualization": [
        "matplotlib",
        "seaborn"
    ],
    "financial": [
        "pandas_datareader",
        "scipy"
    ]
}

# Additional package mappings (what to pip install when a package is missing)
PIP_PACKAGE_MAPPING = {
    "pdfminer": "pdfminer.six",
    "sentence_transformers": "sentence-transformers"
}


def check_dependency(package_name: str) -> bool:
    """
    Check if a dependency is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if package is installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def check_dependency_group(group_name: str) -> Tuple[List[str], List[str]]:
    """
    Check a group of dependencies.
    
    Args:
        group_name: Name of the dependency group
        
    Returns:
        Tuple containing lists of installed and missing packages
    """
    if group_name not in DEPENDENCY_GROUPS:
        logger.warning(f"Unknown dependency group: {group_name}")
        return [], []
    
    installed = []
    missing = []
    
    for package in DEPENDENCY_GROUPS[group_name]:
        if check_dependency(package):
            installed.append(package)
        else:
            missing.append(package)
    
    return installed, missing


def check_all_dependencies() -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Check all dependency groups.
    
    Returns:
        Dictionary mapping group names to tuples of (installed, missing) package lists
    """
    results = {}
    
    for group_name in DEPENDENCY_GROUPS:
        installed, missing = check_dependency_group(group_name)
        results[group_name] = (installed, missing)
    
    return results


def generate_pip_commands(missing_packages: List[str]) -> List[str]:
    """
    Generate pip install commands for missing packages.
    
    Args:
        missing_packages: List of missing package names
        
    Returns:
        List of pip install commands
    """
    commands = []
    
    for package in missing_packages:
        pip_package = PIP_PACKAGE_MAPPING.get(package, package)
        commands.append(f"pip install {pip_package}")
    
    return commands


def print_dependency_report(results: Dict[str, Tuple[List[str], List[str]]]):
    """
    Print a dependency report.
    
    Args:
        results: Dictionary mapping group names to tuples of (installed, missing) package lists
    """
    all_missing = []
    
    print("\n" + "=" * 80)
    print("DEPENDENCY REPORT")
    print("=" * 80)
    
    for group_name, (installed, missing) in results.items():
        print(f"\n{group_name.upper()} DEPENDENCIES:")
        print(f"  Installed: {', '.join(installed) if installed else 'None'}")
        print(f"  Missing: {', '.join(missing) if missing else 'None'}")
        
        all_missing.extend(missing)
    
    if all_missing:
        print("\n" + "=" * 80)
        print("MISSING DEPENDENCIES")
        print("=" * 80)
        print("\nRun the following commands to install missing dependencies:")
        for cmd in generate_pip_commands(all_missing):
            print(f"  {cmd}")
    else:
        print("\n" + "=" * 80)
        print("All dependencies are installed!")
        print("=" * 80)


def main():
    """Run the dependency check."""
    results = check_all_dependencies()
    print_dependency_report(results)
    
    # Return exit code 1 if any dependencies are missing
    for _, (_, missing) in results.items():
        if missing:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
