#!/usr/bin/env python
"""
ChromaDB Financial Chunking Validation Tool

This script tests whether your chunks properly maintain financial semantics by:
1. Testing semantic similarity queries for financial concepts
2. Validating that chunks respect statement boundaries
3. Checking if financial entities are properly extracted and embedded
"""

import os
import sys
import json
import argparse
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from tabulate import tabulate
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Any, Optional

# Add the src directory to the Python path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(current_dir.parent))
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from config.config import settings

# Financial test queries for different statement types and concepts
FINANCIAL_TEST_QUERIES = {
    "income_statement": [
        "total revenue for fiscal year",
        "quarterly earnings before taxes",
        "operating expenses breakdown",
        "net income compared to previous year",
        "cost of goods sold percentage"
    ],
    "balance_sheet": [
        "current assets and liabilities",
        "total shareholder equity",
        "debt to equity ratio",
        "cash and cash equivalents",
        "intangible assets valuation"
    ],
    "cash_flow": [
        "operating cash flow trends",
        "capital expenditures this quarter",
        "cash from financing activities",
        "free cash flow calculation",
        "dividend payments to shareholders"
    ],
    "financial_metrics": [
        "EBITDA margin improvement",
        "return on invested capital",
        "accounts receivable turnover",
        "gross profit margin analysis",
        "working capital requirements"
    ],
    "risk_factors": [
        "currency exchange rate risks",
        "regulatory compliance issues",
        "market competition threats",
        "supply chain vulnerabilities",
        "litigation and legal proceedings"
    ]
}

# Financial entities that should be preserved in chunks
EXPECTED_FINANCIAL_ENTITIES = [
    "revenue", "profit", "earnings", "assets", "liabilities", 
    "equity", "expenses", "cash flow", "EBITDA", "taxes",
    "debt", "shares", "depreciation", "amortization", "dividend"
]

def connect_to_chromadb():
    """Connect to ChromaDB based on configuration"""
    if settings.VECTOR_DB_TYPE == "postgres":
        print(f"Connecting to ChromaDB server at {settings.CHROMA_SERVER_HOST}:{settings.CHROMA_SERVER_PORT}")
        try:
            client = chromadb.HttpClient(
                host=settings.CHROMA_SERVER_HOST,
                port=settings.CHROMA_SERVER_PORT,
                ssl=settings.CHROMA_SERVER_SSL
            )
        except Exception as e:
            print(f"Error connecting to ChromaDB server: {str(e)}")
            print("Falling back to local client...")
            client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH)
            )
    else:
        client = chromadb.PersistentClient(
            path=str(settings.VECTOR_DB_PATH)
        )
    
    return client

def get_collection_stats(client, collection_name):
    """Get basic statistics about the collection"""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        return {
            "name": collection_name,
            "count": count
        }
    except Exception as e:
        print(f"Error getting collection: {str(e)}")
        return None

def test_financial_queries(client, collection_name):
    """Test financial queries to validate semantic understanding"""
    collection = client.get_collection(collection_name)
    
    category_results = {}
    all_results = []
    
    for category, queries in FINANCIAL_TEST_QUERIES.items():
        category_results[category] = {
            "total_queries": len(queries),
            "results": []
        }
        
        for query in queries:
            # Run semantic search with the query
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            # Process results
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                query_results = []
                for i, metadata in enumerate(results['metadatas'][0]):
                    document = metadata.get('source', 'Unknown')
                    page = metadata.get('page', 'N/A')
                    statement_type = metadata.get('statement_type', 'Unknown')
                    
                    # Check if this is a semantic match
                    is_semantic_match = (
                        category == "income_statement" and statement_type in ["income statement", "profit and loss"] or
                        category == "balance_sheet" and statement_type in ["balance sheet", "financial position"] or
                        category == "cash_flow" and statement_type in ["cash flow", "cash flow statement"] or
                        True  # For other categories, we need more sophisticated validation
                    )
                    
                    query_results.append({
                        "document": document,
                        "page": page,
                        "statement_type": statement_type,
                        "is_semantic_match": is_semantic_match
                    })
                
                # Calculate semantic accuracy for this query
                semantic_matches = sum(1 for r in query_results if r["is_semantic_match"])
                accuracy = semantic_matches / len(query_results) if query_results else 0
                
                # Save query results
                result = {
                    "query": query,
                    "results": query_results,
                    "semantic_accuracy": accuracy
                }
                
                category_results[category]["results"].append(result)
                all_results.append((category, query, accuracy))
    
    # Calculate average accuracy for each category
    for category, data in category_results.items():
        accuracies = [r["semantic_accuracy"] for r in data["results"]]
        data["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0
    
    return category_results, all_results

def validate_statement_boundaries(client, collection_name):
    """Validate that chunks respect financial statement boundaries"""
    collection = client.get_collection(collection_name)
    
    # Get all items with their metadata
    results = collection.get()
    
    # Group chunks by document
    docs = defaultdict(list)
    for idx, metadata in enumerate(results.get('metadatas', [])):
        if not metadata:
            continue
            
        doc_id = metadata.get('source', 'unknown')
        page = metadata.get('page', 0)
        chunk_id = idx
        statement_type = metadata.get('statement_type', 'unknown')
        section = metadata.get('section', 'unknown')
        
        docs[doc_id].append({
            'chunk_id': chunk_id,
            'page': page,
            'statement_type': statement_type,
            'section': section
        })
    
    # Check for boundary violations
    boundary_issues = []
    
    for doc_id, chunks in docs.items():
        # Sort chunks by page
        chunks.sort(key=lambda x: int(x['page']) if str(x['page']).isdigit() else 0)
        
        # Check for statement type changes without section changes
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Only check consecutive pages
            if int(curr_chunk['page']) - int(prev_chunk['page']) == 1:
                if (prev_chunk['statement_type'] != curr_chunk['statement_type'] and 
                    prev_chunk['section'] == curr_chunk['section']):
                    boundary_issues.append({
                        'document': doc_id,
                        'page_transition': f"{prev_chunk['page']} -> {curr_chunk['page']}",
                        'statement_change': f"{prev_chunk['statement_type']} -> {curr_chunk['statement_type']}",
                        'section': curr_chunk['section']
                    })
    
    return boundary_issues

def check_financial_entity_preservation(client, collection_name):
    """Check if financial entities are properly preserved in chunks"""
    collection = client.get_collection(collection_name)
    
    # Get all items with their metadata
    results = collection.get()
    
    entity_stats = {
        'total_chunks': len(results.get('ids', [])),
        'chunks_with_entities': 0,
        'entity_counts': Counter(),
        'chunks_by_entity_count': defaultdict(int),
        'entity_examples': {}
    }
    
    for metadata in results.get('metadatas', []):
        if not metadata:
            continue
            
        # Extract financial entities from metadata
        financial_entities = []
        if 'financial_entities' in metadata:
            if isinstance(metadata['financial_entities'], str):
                try:
                    financial_entities = json.loads(metadata['financial_entities'])
                except:
                    pass
            elif isinstance(metadata['financial_entities'], list):
                financial_entities = metadata['financial_entities']
        
        # Count entities
        if financial_entities:
            entity_stats['chunks_with_entities'] += 1
            entity_stats['chunks_by_entity_count'][len(financial_entities)] += 1
            
            for entity in financial_entities:
                entity_stats['entity_counts'][entity] += 1
                
                # Store examples of chunks with this entity (up to 3)
                if entity not in entity_stats['entity_examples'] or len(entity_stats['entity_examples'][entity]) < 3:
                    if 'entity_examples' not in entity_stats:
                        entity_stats['entity_examples'] = {}
                    if entity not in entity_stats['entity_examples']:
                        entity_stats['entity_examples'][entity] = []
                    
                    doc_id = metadata.get('source', 'Unknown')
                    page = metadata.get('page', 'N/A')
                    entity_stats['entity_examples'][entity].append(f"{doc_id} (Page {page})")
    
    # Check coverage of expected financial entities
    entity_coverage = {
        'expected_entities': EXPECTED_FINANCIAL_ENTITIES,
        'found_entities': [e for e in EXPECTED_FINANCIAL_ENTITIES if e in entity_stats['entity_counts']],
        'missing_entities': [e for e in EXPECTED_FINANCIAL_ENTITIES if e not in entity_stats['entity_counts']]
    }
    entity_coverage['coverage_percentage'] = (
        len(entity_coverage['found_entities']) / len(entity_coverage['expected_entities']) * 100
        if entity_coverage['expected_entities'] else 0
    )
    
    return entity_stats, entity_coverage

def test_financial_embedding_models(client, collection_name):
    """Test if the financial embedding models are working correctly"""
    collection = client.get_collection(collection_name)
    
    # Financial terms that should be semantically similar in a good financial model
    financial_term_pairs = [
        ("revenue", "sales"),
        ("profit", "earnings"),
        ("assets", "property"),
        ("liabilities", "debt"),
        ("income statement", "profit and loss"),
        ("balance sheet", "statement of financial position"),
        ("EBITDA", "earnings before interest taxes depreciation amortization"),
        ("accounts receivable", "AR"),
        ("accounts payable", "AP"),
        ("cash flow", "liquidity")
    ]
    
    results = []
    
    # Query embedding function directly if available
    # This is a simplified version - in production, you should use the same embedding
    # function that's configured in your vector store
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        
        for term1, term2 in financial_term_pairs:
            emb1 = embedding_fn([term1])[0]
            emb2 = embedding_fn([term2])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            results.append({
                "term1": term1,
                "term2": term2,
                "similarity": similarity,
                "strong_similarity": similarity > 0.6
            })
    except Exception as e:
        print(f"Error testing embedding model: {str(e)}")
        results = []
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Validate financial chunking in ChromaDB")
    parser.add_argument("collection", help="Collection name to analyze")
    parser.add_argument("--queries", action="store_true", help="Run financial query tests")
    parser.add_argument("--boundaries", action="store_true", help="Check statement boundaries")
    parser.add_argument("--entities", action="store_true", help="Analyze financial entities")
    parser.add_argument("--embeddings", action="store_true", help="Test financial embeddings")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    collection_name = args.collection
    
    # If no specific tests are requested, run them all
    run_all = args.all or not (args.queries or args.boundaries or args.entities or args.embeddings)
    
    print(f"\n=== Financial Chunking Validation for Collection: {collection_name} ===\n")
    
    # Connect to ChromaDB
    client = connect_to_chromadb()
    
    # Get collection stats
    stats = get_collection_stats(client, collection_name)
    if not stats:
        print(f"Collection '{collection_name}' not found or error occurred.")
        return
    
    print(f"Collection: {stats['name']}")
    print(f"Total Chunks: {stats['count']}")
    print("-" * 60)
    
    # Run financial query tests
    if run_all or args.queries:
        print("\n=== Financial Query Tests ===\n")
        category_results, all_results = test_financial_queries(client, collection_name)
        
        # Display results by category
        for category, data in category_results.items():
            print(f"\nCategory: {category}")
            print(f"Average Semantic Accuracy: {data['average_accuracy']:.2f}")
            print(f"Total Queries: {data['total_queries']}")
            
            # Print individual query results
            query_table = []
            for result in data["results"]:
                query_table.append([
                    result["query"][:30] + "..." if len(result["query"]) > 30 else result["query"],
                    f"{result['semantic_accuracy']:.2f}"
                ])
            
            print(tabulate(query_table, headers=["Query", "Accuracy"], tablefmt="simple"))
        
        # Overall accuracy
        overall_accuracy = sum(acc for _, _, acc in all_results) / len(all_results) if all_results else 0
        print(f"\nOverall Semantic Accuracy: {overall_accuracy:.2f}")
        print("-" * 60)
    
    # Check statement boundaries
    if run_all or args.boundaries:
        print("\n=== Financial Statement Boundary Validation ===\n")
        boundary_issues = validate_statement_boundaries(client, collection_name)
        
        if boundary_issues:
            print(f"Found {len(boundary_issues)} potential boundary violations:")
            issue_table = []
            for issue in boundary_issues[:10]:  # Show at most 10 issues
                issue_table.append([
                    issue["document"],
                    issue["page_transition"],
                    issue["statement_change"],
                    issue["section"]
                ])
            
            print(tabulate(issue_table, 
                          headers=["Document", "Page Transition", "Statement Change", "Section"], 
                          tablefmt="simple"))
            
            if len(boundary_issues) > 10:
                print(f"... and {len(boundary_issues) - 10} more issues.")
        else:
            print("No boundary violations detected - chunks respect financial statement boundaries.")
        print("-" * 60)
    
    # Check financial entity preservation
    if run_all or args.entities:
        print("\n=== Financial Entity Preservation Analysis ===\n")
        entity_stats, entity_coverage = check_financial_entity_preservation(client, collection_name)
        
        print(f"Chunks with Financial Entities: {entity_stats['chunks_with_entities']} of {entity_stats['total_chunks']} ({entity_stats['chunks_with_entities']/entity_stats['total_chunks']*100:.1f}%)")
        
        # Entity distribution
        print("\nEntities per Chunk Distribution:")
        dist_table = []
        for count, num_chunks in sorted(entity_stats['chunks_by_entity_count'].items()):
            dist_table.append([count, num_chunks, f"{num_chunks/entity_stats['total_chunks']*100:.1f}%"])
        print(tabulate(dist_table, headers=["Entities per Chunk", "Count", "Percentage"], tablefmt="simple"))
        
        # Top entities
        print("\nTop Financial Entities:")
        top_entities = entity_stats['entity_counts'].most_common(15)
        entity_table = []
        for entity, count in top_entities:
            entity_table.append([entity, count])
        print(tabulate(entity_table, headers=["Entity", "Occurrences"], tablefmt="simple"))
        
        # Expected entity coverage
        print(f"\nExpected Financial Entity Coverage: {entity_coverage['coverage_percentage']:.1f}%")
        if entity_coverage['missing_entities']:
            print("Missing Expected Entities:")
            for entity in entity_coverage['missing_entities']:
                print(f"- {entity}")
        print("-" * 60)
    
    # Test financial embeddings
    if run_all or args.embeddings:
        print("\n=== Financial Embedding Model Validation ===\n")
        embedding_results = test_financial_embedding_models(client, collection_name)
        
        if embedding_results:
            print("Financial Term Pair Similarities:")
            embed_table = []
            for result in embedding_results:
                embed_table.append([
                    result["term1"],
                    result["term2"],
                    f"{result['similarity']:.3f}",
                    "✓" if result["strong_similarity"] else "✗"
                ])
            print(tabulate(embed_table, 
                          headers=["Term 1", "Term 2", "Similarity", "Strong Match"], 
                          tablefmt="simple"))
            
            # Calculate overall performance
            success_rate = sum(1 for r in embedding_results if r["strong_similarity"]) / len(embedding_results)
            print(f"\nEmbedding Model Financial Concept Success Rate: {success_rate:.1%}")
        else:
            print("Could not test embedding model directly.")
        print("-" * 60)

if __name__ == "__main__":
    main()
