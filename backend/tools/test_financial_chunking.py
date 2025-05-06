#!/usr/bin/env python3
"""
Financial Chunking Test Script

This script tests the advanced financial chunking system by processing sample
financial documents and evaluating the chunking quality, boundary preservation,
and financial entity extraction.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tabulate import tabulate

# Add the parent directory to sys.path to import from the backend package
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processing.base_processor import Document, DocumentChunk
from src.document_processing.chunking import (
    ChunkerFactory,
    FinancialStatementChunker,
    MDAndARiskChunker,
    FinancialNotesChunker,
    FinancialStatementType
)
from src.config.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_sample_documents() -> List[Document]:
    """
    Get or create sample financial documents for testing.
    
    Returns:
        A list of Document objects
    """
    samples = []
    
    # Sample 1: Income Statement
    income_statement = Document(
        doc_id="sample_income_statement",
        content="""
CONSOLIDATED STATEMENTS OF INCOME
For the Years Ended December 31, 2023, 2022, and 2021
(in millions, except per share amounts)

                                    2023        2022        2021
Revenue                          $ 150,000    $ 120,000   $ 100,000
Cost of revenue                     90,000       75,000      65,000
Gross profit                        60,000       45,000      35,000

Operating expenses:
  Research and development          25,000       20,000      15,000
  Sales and marketing               15,000       12,000      10,000
  General and administrative         5,000        4,000       3,500
Total operating expenses            45,000       36,000      28,500

Operating income                    15,000        9,000       6,500
Other income (expense), net            500          250         100
Income before income taxes          15,500        9,250       6,600
Provision for income taxes           3,100        1,850       1,320
Net income                       $  12,400    $   7,400   $   5,280

Earnings per share:
  Basic                          $    3.10    $    1.85   $    1.32
  Diluted                        $    3.05    $    1.82   $    1.30

Weighted-average shares used in computing earnings per share:
  Basic                              4,000        4,000       4,000
  Diluted                            4,065        4,065       4,065
        """,
        metadata={
            "filename": "income_statement.txt",
            "file_type": ".txt",
            "statement_type": "income_statement",
            "document_category": "financial"
        }
    )
    samples.append(income_statement)
    
    # Sample 2: Management Discussion and Analysis
    mda = Document(
        doc_id="sample_mda",
        content="""
ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

OVERVIEW

Our revenue increased 25% to $150 billion in fiscal 2023 compared to $120 billion in fiscal 2022. This increase was driven primarily by growth in our cloud services segment, which saw a 40% year-over-year increase. The global shift to cloud computing continues to accelerate, with enterprises increasingly migrating their workloads to our platform.

RESULTS OF OPERATIONS

Revenue
Revenue increased by $30 billion or 25% in fiscal 2023 compared to fiscal 2022. This increase was primarily due to:
• Cloud services growth of 40% to $75 billion
• Software subscription revenue growth of 15% to $50 billion
• Hardware revenue growth of 10% to $25 billion

The growth in cloud services reflects increased demand for our infrastructure offerings as well as our platform-as-a-service solutions. Software subscription revenue growth was driven by continued customer adoption of our productivity and security solutions. Hardware revenue growth was primarily attributable to increased sales of our new device lineup introduced in the third quarter.

Cost of Revenue and Gross Margin
Cost of revenue increased by $15 billion or 20% in fiscal 2023, resulting in a gross margin of 40%, up from 37.5% in fiscal 2022. The improvement in gross margin was primarily due to a favorable shift in product mix toward higher-margin cloud services and software subscriptions, as well as efficiency improvements in our data center operations.

Operating Expenses
Research and development expenses increased by $5 billion or 25% in fiscal 2023, primarily reflecting investments in artificial intelligence capabilities, quantum computing research, and new cloud service offerings.

Sales and marketing expenses increased by $3 billion or 25% in fiscal 2023, primarily due to increased headcount to support revenue growth and expanded marketing programs for our cloud services.

General and administrative expenses increased by $1 billion or 25% in fiscal 2023, mainly due to increased personnel costs and investments in compliance infrastructure.

LIQUIDITY AND CAPITAL RESOURCES

As of December 31, 2023, we had $50 billion in cash, cash equivalents, and short-term investments, compared with $40 billion as of December 31, 2022. The increase was primarily due to cash generated from operations, partially offset by capital expenditures, dividends, and share repurchases.

Cash provided by operating activities was $35 billion in fiscal 2023, an increase of $10 billion compared to fiscal 2022, primarily due to higher net income and improved working capital management.

Capital expenditures were $15 billion in fiscal 2023, primarily related to investments in data center capacity, server equipment, and facilities to support our growing cloud services business.
        """,
        metadata={
            "filename": "mda.txt",
            "file_type": ".txt",
            "statement_type": "management_discussion",
            "document_category": "financial"
        }
    )
    samples.append(mda)
    
    # Sample 3: Financial Notes
    notes = Document(
        doc_id="sample_notes",
        content="""
NOTES TO CONSOLIDATED FINANCIAL STATEMENTS

NOTE 1: SUMMARY OF SIGNIFICANT ACCOUNTING POLICIES

Basis of Presentation
The consolidated financial statements include the accounts of the Company and its subsidiaries. All intercompany transactions and balances have been eliminated in consolidation.

Use of Estimates
The preparation of financial statements in conformity with U.S. generally accepted accounting principles (GAAP) requires management to make estimates and assumptions that affect the reported amounts of assets and liabilities and disclosure of contingent assets and liabilities at the date of the financial statements and the reported amounts of revenues and expenses during the reporting period. Actual results could differ from those estimates.

Revenue Recognition
The Company recognizes revenue when control of the promised goods or services is transferred to customers, in an amount that reflects the consideration the Company expects to be entitled to in exchange for those goods or services.

Cloud Services Revenue: Cloud services revenue is recognized ratably over the contract period as the customer simultaneously receives and consumes the benefits provided by the Company's performance as the Company performs.

Software Subscription Revenue: Software subscription revenue is recognized ratably over the subscription period, beginning on the date the service is made available to the customer.

Hardware Revenue: Hardware revenue is recognized when control of the products has transferred to the customer, which is generally upon shipment.

NOTE 2: SEGMENT INFORMATION

The Company operates in three reportable segments: Cloud Services, Software Subscriptions, and Hardware. The Company's Chief Operating Decision Maker (CODM), its Chief Executive Officer, evaluates performance based on revenue and operating income of each segment.

Segment information for the years ended December 31, 2023, 2022, and 2021 is as follows (in millions):

Cloud Services:
                       2023        2022        2021
Revenue             $ 75,000    $ 53,571    $ 42,857
Operating income      30,000      21,429      17,143

Software Subscriptions:
                       2023        2022        2021
Revenue             $ 50,000    $ 43,478    $ 39,130
Operating income      20,000      17,391      15,652

Hardware:
                       2023        2022        2021
Revenue             $ 25,000    $ 22,727    $ 18,182
Operating income      10,000       9,091       7,273

NOTE 3: INCOME TAXES

The components of income before income taxes are as follows (in millions):
                       2023        2022        2021
Domestic            $ 10,000    $  6,000    $  4,000
Foreign                5,500       3,250       2,600
Total               $ 15,500    $  9,250    $  6,600
        """,
        metadata={
            "filename": "notes.txt",
            "file_type": ".txt",
            "statement_type": "notes",
            "document_category": "financial"
        }
    )
    samples.append(notes)
    
    return samples


def test_chunking_strategies():
    """
    Test different financial chunking strategies on sample documents.
    """
    samples = get_sample_documents()
    
    # Results table
    results = []
    
    # Test the factory approach first
    logger.info("Testing ChunkerFactory...")
    for doc in samples:
        doc_type = doc.metadata.get("document_category", "financial")
        stmt_type = doc.metadata.get("statement_type", None)
        
        chunker = ChunkerFactory.create_chunker(
            document_type=doc_type,
            statement_type=stmt_type,
            chunk_size=1000,
            chunk_overlap=100
        )
        
        chunks = chunker.chunk(doc)
        
        # Log result
        results.append([
            doc.doc_id,
            type(chunker).__name__,
            len(chunks),
            _has_preserved_financial_boundaries(chunks),
            _average_financial_entities_per_chunk(chunks)
        ])
    
    # Test specific chunkers directly
    logger.info("Testing specific financial chunkers...")
    
    # Test FinancialStatementChunker on all documents
    for doc in samples:
        chunker = FinancialStatementChunker(
            chunk_size=1000,
            chunk_overlap=100,
            respect_financial_boundaries=True
        )
        
        chunks = chunker.chunk(doc)
        
        results.append([
            doc.doc_id,
            "FinancialStatementChunker",
            len(chunks),
            _has_preserved_financial_boundaries(chunks),
            _average_financial_entities_per_chunk(chunks)
        ])
    
    # Test MDAndARiskChunker on all documents
    for doc in samples:
        chunker = MDAndARiskChunker(
            chunk_size=1000,
            chunk_overlap=100,
            respect_financial_boundaries=True
        )
        
        chunks = chunker.chunk(doc)
        
        results.append([
            doc.doc_id,
            "MDAndARiskChunker",
            len(chunks),
            _has_preserved_financial_boundaries(chunks),
            _average_financial_entities_per_chunk(chunks)
        ])
    
    # Test FinancialNotesChunker on all documents
    for doc in samples:
        chunker = FinancialNotesChunker(
            chunk_size=1000,
            chunk_overlap=100,
            respect_financial_boundaries=True
        )
        
        chunks = chunker.chunk(doc)
        
        results.append([
            doc.doc_id,
            "FinancialNotesChunker",
            len(chunks),
            _has_preserved_financial_boundaries(chunks),
            _average_financial_entities_per_chunk(chunks)
        ])
    
    # Display results
    print("\nFinancial Chunking Test Results:")
    print(tabulate(
        results,
        headers=[
            "Document ID", 
            "Chunker Type", 
            "Chunk Count", 
            "Preserved Boundaries",
            "Avg Financial Entities"
        ],
        tablefmt="grid"
    ))


def _has_preserved_financial_boundaries(chunks: List[DocumentChunk]) -> bool:
    """
    Check if chunks have preserved financial statement boundaries.
    A simple heuristic: each chunk should have a statement_type in its metadata.
    """
    for chunk in chunks:
        if "statement_type" not in chunk.metadata:
            return False
    return True


def _average_financial_entities_per_chunk(chunks: List[DocumentChunk]) -> float:
    """
    Calculate the average number of financial entities per chunk.
    """
    total_entities = 0
    for chunk in chunks:
        entities = chunk.metadata.get("financial_entities", [])
        total_entities += len(entities) if isinstance(entities, list) else 0
    
    return total_entities / len(chunks) if chunks else 0


def analyze_chunk_consistency(chunks: List[DocumentChunk]):
    """
    Analyze the consistency of chunks for metadata and content.
    """
    print("\nChunk Consistency Analysis:")
    
    stmt_types = {}
    chunk_sizes = []
    entities_present = 0
    
    for chunk in chunks:
        # Record statement type
        stmt_type = chunk.metadata.get("statement_type", "unknown")
        stmt_types[stmt_type] = stmt_types.get(stmt_type, 0) + 1
        
        # Record chunk size
        chunk_sizes.append(len(chunk.content))
        
        # Check for entities
        if "financial_entities" in chunk.metadata and chunk.metadata["financial_entities"]:
            entities_present += 1
    
    # Print statistics
    print(f"Total chunks: {len(chunks)}")
    print(f"Statement types distribution: {stmt_types}")
    print(f"Average chunk size: {sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0:.2f} characters")
    print(f"Chunks with financial entities: {entities_present} ({(entities_present / len(chunks) * 100) if chunks else 0:.2f}%)")


def detailed_chunk_inspection(doc_id: str, chunks: List[DocumentChunk]):
    """
    Perform a detailed inspection of chunks from a specific document.
    """
    print(f"\nDetailed Inspection for {doc_id}:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}/{len(chunks)} (ID: {chunk.chunk_id}):")
        print(f"  Statement Type: {chunk.metadata.get('statement_type', 'unknown')}")
        
        # Check for financial entities
        entities = chunk.metadata.get("financial_entities", [])
        if entities:
            print(f"  Financial Entities ({len(entities)}): {', '.join(entities[:5])}{', ...' if len(entities) > 5 else ''}")
        
        # Check for tables
        contains_table = chunk.metadata.get("contains_table", False)
        print(f"  Contains Table: {contains_table}")
        
        # Show beginning and end of content
        content = chunk.content.strip()
        print(f"  Content Preview ({len(content)} chars):")
        print(f"    Beginning: {content[:100]}...")
        print(f"    End: ...{content[-100:]}")
        print("  " + "-" * 50)


def main():
    """
    Main function to run the financial chunking tests.
    """
    logger.info("Starting financial chunking tests...")
    
    # Test different chunking strategies
    test_chunking_strategies()
    
    # Get sample documents for more detailed analysis
    samples = get_sample_documents()
    
    for doc in samples:
        # Use the factory to get the appropriate chunker
        chunker = ChunkerFactory.create_chunker(
            document_type=doc.metadata.get("document_category", "financial"),
            statement_type=doc.metadata.get("statement_type", None),
            chunk_size=1000,
            chunk_overlap=100
        )
        
        chunks = chunker.chunk(doc)
        
        # Analyze chunk consistency
        analyze_chunk_consistency(chunks)
        
        # Detailed inspection of chunks
        detailed_chunk_inspection(doc.doc_id, chunks)
    
    logger.info("Financial chunking tests completed.")


if __name__ == "__main__":
    main()
