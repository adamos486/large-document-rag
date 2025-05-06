import os
import sys
import json
from pathlib import Path

# Set up path for imports - make it robust to different execution contexts
root_dir = Path(__file__).parent.parent  # backend dir when run from within tools/
sys.path.insert(0, str(root_dir))

# Import necessary modules
try:
    # Try direct import first (when running from backend/ directory)
    from src.vector_store.vector_db import VectorStore
    from src.config.config import settings
except ImportError:
    print("Trying alternative import paths...")
    try:
        # Try full module path (when running from project root)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root
        from backend.src.vector_store.vector_db import VectorStore
        from backend.src.config.config import settings
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        print(f"Current sys.path: {sys.path}")
        
        # Absolute last resort - try to modify path directly based on expected structure
        project_root = Path.cwd()
        while project_root.name and project_root.name != 'large-document-rag':
            project_root = project_root.parent
            
        if not project_root.name:
            print("Could not find project root directory")
            sys.exit(1)
            
        print(f"Detected project root: {project_root}")
        sys.path.insert(0, str(project_root / 'backend'))
        
        try:
            from src.vector_store.vector_db import VectorStore
            from src.config.config import settings
            print("Successfully imported modules with modified path")
        except ImportError as e:
            print(f"All import attempts failed: {e}")
            sys.exit(1)

# Import rich for better terminal output
USE_RICH = False
console = None
Table = None
Panel = None

try:
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    USE_RICH = True
    console = Console()
except ImportError:
    USE_RICH = False
    console = None
    # Define stub classes for type checking
    class Table:
        def __init__(self, *args, **kwargs):
            pass
        def add_column(self, *args, **kwargs):
            pass
        def add_row(self, *args, **kwargs):
            pass

def explore_collection(collection_name="default"):
    """Explore the contents of a ChromaDB collection."""
    if USE_RICH:
        console.print(f"[bold cyan]EXPLORING COLLECTION:[/bold cyan] [green]{collection_name}[/green]\n")
    else:
        print(f"\n{'='*50}")
        print(f"EXPLORING COLLECTION: {collection_name}")
        print(f"{'='*50}\n")
    
    # Initialize vector store
    try:
        vector_store = VectorStore(collection_name=collection_name)
    except Exception as e:
        if USE_RICH:
            console.print(f"[bold red]Error initializing vector store:[/bold red] {e}")
        else:
            print(f"Error initializing vector store: {e}")
        return
    
    # Get all collections
    if USE_RICH:
        console.print("[bold]Available collections:[/bold]")
        all_collections = vector_store.client.list_collections()
        for coll in all_collections:
            console.print(f"  • [cyan]{coll.name}[/cyan] ({coll.count()} items)")
        console.print("")
    
    # Try to get chunks and stats
    try:
        # Updated ChromaDB query approach to be more compatible with newer ChromaDB versions
        # Try different query methods since ChromaDB API might differ between versions
        chunks = []
        
        try:
            # First approach: Using where parameter with a simple condition that's always true
            # This works for newer ChromaDB versions that require a specific 'where' operator
            result = vector_store.collection.get(where={"doc_id": {"$ne": "non_existent_placeholder"}})
        except Exception as query_err:
            print(f"First query approach failed: {str(query_err)}")
            try:
                # Second approach: Try with include parameter
                result = vector_store.collection.get(include=["metadatas", "documents", "embeddings"])
            except Exception as query_err2:
                print(f"Second query approach failed: {str(query_err2)}")
                try:
                    # Third approach: List all IDs first, then get by IDs
                    all_ids = vector_store.collection.get(include=["ids"])["ids"]
                    if all_ids:
                        result = vector_store.collection.get(ids=all_ids)  
                    else:
                        result = {"ids": [], "documents": [], "metadatas": []}
                except Exception as query_err3:
                    print(f"All query approaches failed: {str(query_err3)}")
                    result = {"ids": [], "documents": [], "metadatas": []}
        
        if result and "documents" in result and result["documents"]:
            for i, doc in enumerate(result["documents"]):
                chunk_id = result["ids"][i] if "ids" in result else "Unknown"
                metadata = result["metadatas"][i] if "metadatas" in result else {}
                
                # Handling financial document metadata specifically
                doc_name = metadata.get("doc_name", metadata.get("filename", "Unknown"))
                doc_type = metadata.get("doc_type", metadata.get("file_type", "Unknown"))
                
                # Financial-specific metadata extraction
                financial_entities = metadata.get("financial_entities", [])
                financial_metrics = metadata.get("financial_metrics", {})
                period_info = metadata.get("period", "N/A")
                
                # Enhanced metadata for financial documents
                enhanced_metadata = {
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "financial_entities": financial_entities,
                    "financial_metrics": financial_metrics,
                    "period": period_info,
                    **metadata  # Include all original metadata
                }
                
                chunks.append({"id": chunk_id, "metadata": enhanced_metadata, "content": doc})
        
        # If we have chunks, process them
        if chunks:
            total_chunks = len(chunks)
            doc_set = set()
            doc_types = {}
            financial_entity_counts = {}
            
            for chunk in chunks:
                metadata = chunk["metadata"]
                doc_name = metadata.get("doc_name", "Unknown")
                doc_set.add(doc_name)
                
                # Count document types
                doc_type = metadata.get("doc_type", "Unknown")
                if doc_type in doc_types:
                    doc_types[doc_type] += 1
                else:
                    doc_types[doc_type] = 1
                    
                # Count financial entities
                for entity in metadata.get("financial_entities", []):
                    if entity in financial_entity_counts:
                        financial_entity_counts[entity] += 1
                    else:
                        financial_entity_counts[entity] = 1
            
            total_docs = len(doc_set)
            
            print(f"Total chunks: {total_chunks}")
            print(f"Total unique documents: {total_docs}\n")
            
            print("Document Types:")
            if doc_types:
                for doc_type, count in doc_types.items():
                    print(f"  • {doc_type}: {count} chunks")
            else:
                print("  No document type information available")
            
            # Print financial entity information if available
            if financial_entity_counts:
                print("\nTop Financial Entities:")
                sorted_entities = sorted(financial_entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for entity, count in sorted_entities:
                    print(f"  • {entity}: {count} mentions")
            
            # Print sample of chunks
            print("\nSample Chunks:")
            sample_size = min(5, len(chunks))
            
            # Check if rich is available and properly imported
            if USE_RICH and console:
                table = Table(show_header=True, header_style="bold")
                table.add_column("ID", style="dim")
                table.add_column("Document")
                table.add_column("Content Preview", width=60)
                
                for i in range(sample_size):
                    content_preview = chunks[i]["content"][:57] + "..." if len(chunks[i]["content"]) > 60 else chunks[i]["content"]
                    table.add_row(
                        chunks[i]["id"][:8], 
                        chunks[i]["metadata"]["doc_name"], 
                        content_preview
                    )
                
                console.print(table)
            else:
                for i in range(sample_size):
                    print(f"Chunk {i+1}/{sample_size}:")
                    print(f"  ID: {chunks[i]['id'][:8]}")
                    print(f"  Document: {chunks[i]['metadata']['doc_name']}")
                    print(f"  Type: {chunks[i]['metadata'].get('doc_type', 'Unknown')}")
                    if 'financial_metrics' in chunks[i]['metadata'] and chunks[i]['metadata']['financial_metrics']:
                        print(f"  Financial Metrics: {json.dumps(chunks[i]['metadata']['financial_metrics'], indent=2)}")
                    content_preview = chunks[i]["content"][:100] + "..." if len(chunks[i]["content"]) > 100 else chunks[i]["content"]
                    print(f"  Content: {content_preview}\n")
        else:
            print("Total chunks: 0")
            print("Total unique documents: 0\n")
            
            print("Document Types:")
            print("  No document type information available")
            
            print("\nSample Chunks:")
            if USE_RICH and console:
                table = Table(show_header=True, header_style="bold")
                table.add_column("ID", style="dim")
                table.add_column("Document")
                table.add_column("Content Preview", width=60)
                console.print(table)
            else:
                print("  No chunks to display")
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        print("Total chunks: 0")
        print("Total unique documents: 0\n")
        
        print("Document Types:")
        print("  No document type information available")
        
        print("\nSample Chunks:")
        if USE_RICH and console:
            table = Table(show_header=True, header_style="bold")
            table.add_column("ID", style="dim")
            table.add_column("Document")
            table.add_column("Content Preview", width=60)
            console.print(table)
        else:
            print("  No chunks to display")

    # Return chunks for further exploration if needed
    return chunks

def main():
    """Main function to handle command line arguments."""
    # Get collection name from command line
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    
    if USE_RICH:
        console.print("[bold]ChromaDB Exploration Utility[/bold]")
        console.print(f"[dim]Using configuration from:[/dim] {settings.PROJECT_ROOT}")
        console.print(f"[dim]Vector DB path:[/dim] {settings.VECTOR_DB_PATH}\n")
    else:
        print("ChromaDB Exploration Utility")
        print(f"Using configuration from: {settings.PROJECT_ROOT}")
        print(f"Vector DB path: {settings.VECTOR_DB_PATH}\n")
    
    # Explore the collection
    explore_collection(collection_name)

if __name__ == "__main__":
    main()