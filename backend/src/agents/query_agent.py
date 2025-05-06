import os
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
import json
import datetime

from ..config.config import settings
from .base_agent import Agent
from ..utils.embeddings import EmbeddingModel
from ..vector_store.vector_db import VectorStore
from ..document_processing.financial_indexer import FinancialIndexer
from ..llm.llm_provider import llm_provider_factory, LLMTask
from ..llm.financial_task_detector import financial_task_detector, FinancialTaskType
from ..exceptions import QueryProcessingError, LLMProviderError
from langchain.prompts.prompt import PromptTemplate

# Set up logging
logger = logging.getLogger(__name__)

class QueryAgent(Agent):
    """Agent for querying the vector database and generating responses."""
    
    def __init__(
        self, 
        collection_name: str = "default",
        n_results: int = 5,
        agent_id: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.0,
        use_financial_indexer: bool = True
    ):
        """Initialize the query agent.
        
        Args:
            collection_name: Name of the vector database collection to use.
            n_results: Number of results to retrieve from the vector store.
            agent_id: Unique identifier for this agent.
            llm_model: Language model to use for response generation.
            temperature: Temperature for response generation.
            use_financial_indexer: Whether to use enhanced financial indexing for better retrieval.
        """
        super().__init__(agent_id=agent_id, name="QueryAgent")
        self.collection_name = collection_name
        self.n_results = n_results
        self.llm_model = llm_model or settings.LLM_MODEL
        self.temperature = temperature or settings.TEMPERATURE
        self.use_financial_indexer = use_financial_indexer
        
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(collection_name=self.collection_name)
        
        # Initialize financial indexer if enabled
        if self.use_financial_indexer:
            index_path = settings.DATA_DIR / 'financial_indices' / self.collection_name
            self.financial_indexer = FinancialIndexer(index_path=index_path)
        
        # Don't initialize LLM here - we'll get the right one for each query
    
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Query the vector database and generate a response.
        
        Args:
            query: The query text.
            filters: Optional metadata filters for the vector store query.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary with query results and generated response.
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Get query embedding
            start_time = time.time()
            query_embedding = self.embedding_model.get_embedding(query)
            embedding_time = time.time() - start_time
            
            # Query vector store and financial indexer if enabled
            start_time = time.time()
            
            # If financial indexer is enabled, use it to enhance the search
            enhanced_results = []
            if self.use_financial_indexer:
                try:
                    # Get relevant chunk IDs from financial indexer
                    enhanced_chunk_ids = self.financial_indexer.search(
                        query=query,
                        filters=filters,
                        max_results=self.n_results * 2  # Get more results for re-ranking
                    )
                    
                    if enhanced_chunk_ids:
                        logger.info(f"Financial indexer found {len(enhanced_chunk_ids)} relevant chunks")
                        # Get chunks from vector store by IDs
                        for chunk_id in enhanced_chunk_ids:
                            chunk = self.vector_store.get_chunk_by_id(chunk_id)
                            if chunk:
                                enhanced_results.append(chunk)
                except Exception as e:
                    logger.warning(f"Error using financial indexer: {e}. Falling back to vector search.")
            
            # Standard vector search (used as fallback or to supplement financial indexer results)
            results = self.vector_store.query(
                query_text=query,
                n_results=self.n_results - len(enhanced_results) if enhanced_results else self.n_results,
                where=filters
            )
            retrieval_time = time.time() - start_time
            
            # Process results from vector search
            retrieved_docs = []
            
            # Add enhanced results from financial indexer if available
            for i, chunk in enumerate(enhanced_results):
                retrieved_docs.append({
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "document_id": chunk.chunk_id,
                    "distance": 0.0,  # Indicates it came from financial indexer
                    "source": "financial_indexer"
                })
            
            # Add results from vector search
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if "distances" in results and results["distances"] else None
                    doc_id = results["ids"][0][i] if results["ids"] else f"result_{i}"
                    
                    # Check if this document is already in retrieved_docs (from financial indexer)
                    if not any(d["document_id"] == doc_id for d in retrieved_docs):
                        retrieved_docs.append({
                            "content": doc,
                            "metadata": metadata,
                            "document_id": doc_id,
                            "distance": distance,
                            "source": "vector_search"
                        })
            
            # Generate response using the appropriate LLM based on query analysis
            llm_response = None
            generation_time = 0
            query_type = None
            financial_entities = {}
            provider_used = None
            confidence_score = None
            
            # Analyze the query using financial task detector
            task_type, confidence_scores, financial_entities = llm_provider_factory.analyze_financial_query(query)
            query_type = task_type
            
            # Get the most confident score
            if confidence_scores:
                confidence_score = max(confidence_scores.values())
            
            # Get override provider if specified in kwargs
            override_provider = kwargs.get('llm_provider')
            if override_provider:
                logger.info(f"Using override LLM provider: {override_provider}")
            
            if retrieved_docs:
                start_time = time.time()
                llm_response, provider_used = self._generate_response(query, retrieved_docs, task_type, override_provider)
                generation_time = time.time() - start_time
            
            # Return query results
            response = {
                "financial_task_type": query_type,
                "financial_entities": financial_entities,
                "llm_provider": provider_used,
                "confidence_score": confidence_score,
                "query": query,
                "retrieved_documents": retrieved_docs,
                "document_count": len(retrieved_docs),
                "llm_response": llm_response,
                "embedding_time": embedding_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": embedding_time + retrieval_time + generation_time,
                "financial_index_enhanced": len(enhanced_results) > 0 if self.use_financial_indexer else False,
                "financial_entities": self._extract_financial_entities_from_query(query) if self.use_financial_indexer else {}
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            raise
    
    def _extract_financial_entities_from_query(self, query: str) -> Dict[str, List[str]]:
        """Extract financial entities from the query for debugging and explanation purposes."""
        if not self.use_financial_indexer:
            return {}
        
        try:
            # The financial indexer has methods to extract entities, leverage that
            entity_patterns = self.financial_indexer.entity_patterns
            entities = {}
            
            import re
            for entity_type, pattern in entity_patterns.items():
                matches = re.findall(pattern, query)
                if matches:
                    entities[entity_type] = list(set(matches))
            
            return entities
        except Exception as e:
            logger.warning(f"Error extracting financial entities: {e}")
            return {}
    
    def _generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                        task_type: str = None, override_provider: str = None) -> Tuple[str, str]:
        """Generate a response using the LLM based on retrieved documents.
        
        Args:
            query: The original query.
            retrieved_docs: List of retrieved documents.
            task_type: The detected financial task type.
            override_provider: Optional override for LLM provider.
            
        Returns:
            Tuple containing generated response text and the provider used.
        """
        try:
            # Use the financial task type if provided, otherwise determine from query
            task = task_type or self._determine_query_type(query, retrieved_docs)
            logger.info(f"Using task type for response generation: {task}")
            
            # Get task-specific prompt
            prompt_template = self._get_prompt_for_task(task, query)
            
            # Prepare context from retrieved documents
            context_docs = []
            for i, doc in enumerate(retrieved_docs):
                # Add document with its metadata
                doc_text = f"Document {i+1}:\n{doc['content']}\n"
                
                # Add relevant metadata if available
                metadata = doc.get('metadata', {})
                if metadata:
                    # Include only relevant metadata for context
                    relevant_meta = {}
                    for key in ['source', 'title', 'page', 'chunk_id', 'doc_type', 'document_type', 
                               'reporting_entity', 'financial_period', 'risk_level', 'sentiment']:
                        if key in metadata:
                            relevant_meta[key] = metadata[key]
                    
                    if relevant_meta:
                        doc_text += f"Metadata: {json.dumps(relevant_meta)}\n"
                
                context_docs.append(doc_text)
            
            # Join all document contexts
            context = "\n\n".join(context_docs)
            
            # If context is too long, truncate it (token limit considerations)
            if len(context) > 15000:  # Approximate limit to avoid token issues
                context = context[:15000] + "\n[Context truncated due to length]\n"
            
            # Get task-specific LLM - use the query text for intelligent task routing
            task_llm = llm_provider_factory.get_llm_for_task(task, query=query, override_provider=override_provider)
            if not task_llm:
                raise LLMProviderError("No LLM provider available for this task")
            
            # Track which provider was used
            if override_provider:
                provider_used = override_provider
            elif task_type:
                # Get the mapped provider for this financial task type
                provider_used = financial_task_detector.task_llm_mapping.get(task_type, "unknown")
            else:
                provider_used = "default"  # Fallback
            
            # Format prompt with context and query
            prompt = PromptTemplate.from_template(prompt_template).format(
                context=context,
                query=query,
                task_type=task_type or "general query",
                current_date=datetime.datetime.now().strftime("%Y-%m-%d")
            )
            
            # Generate response
            if hasattr(task_llm, 'invoke'):
                # For newer LLM interfaces
                response = task_llm.invoke(prompt)
            else:
                # For older LLM interfaces
                response = task_llm(prompt)
                
            # Extract text from response (different LLMs return different formats)
            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict) and 'content' in response:
                response_text = response['content']
            elif isinstance(response, str):
                response_text = response
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                response_text = str(response)
            
            return response_text, provider_used
        
        except LLMProviderError as e:
            logger.error(f"LLM provider error: {e}")
            return f"Error: Unable to generate response due to LLM provider issue: {e}", "none"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise QueryProcessingError(f"Failed to generate response: {str(e)}", query=query)
            
    def _determine_query_type(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Determine what type of financial query/task is being requested.
        
        Args:
            query: The user query
            retrieved_docs: Documents retrieved from vector store
            
        Returns:
            The LLMTask type for this query
        """
        # Basic keyword-based task classification
        query_lower = query.lower()
        doc_content = " ".join([doc.get("content", "").lower() for doc in retrieved_docs])
        
        # Financial analysis indicators
        financial_keywords = [
            "financial performance", "balance sheet", "income statement", "cash flow",
            "revenue", "profit margin", "debt", "liability", "asset", "valuation",
            "ebitda", "earnings", "fiscal", "quarter", "financial risk", "ratios", "statements"
        ]
        
        # Document summarization indicators
        summarization_keywords = [
            "summarize", "summary", "overview", "brief", "outline", "recap",
            "summarization", "digest", "key points", "synopsis", "highlights"
        ]
        
        # Check document content size - large documents might benefit from Claude's larger context window
        combined_content = " ".join([doc.get("content", "") for doc in retrieved_docs])
        is_large_context = len(combined_content) > 10000  # 10k characters threshold
        
        # Count keyword matches
        financial_score = sum(1 for kw in financial_keywords if kw in query_lower or kw in doc_content)
        summarization_score = sum(1 for kw in summarization_keywords if kw in query_lower)
        
        # Determine the task type
        if financial_score > 2 or "financial analysis" in query_lower:
            return LLMTask.FINANCIAL_ANALYSIS
        elif summarization_score > 0 or is_large_context:
            return LLMTask.DOCUMENT_SUMMARIZATION
        else:
            return LLMTask.QUERY_RESPONSE
    
    def _get_prompt_for_task(self, task: str, query: str) -> str:
        """Get a task-specific prompt template.
        
        Args:
            task: The task type (either from LLMTask or FinancialTaskType)
            query: The original query
            
        Returns:
            A prompt template string
        """
        # Financial-specific task prompts
        if task in [FinancialTaskType.RATIO_ANALYSIS, FinancialTaskType.TREND_ANALYSIS, 
                   FinancialTaskType.FINANCIAL_ANALYSIS]:
            return """
            You are an expert financial analyst specializing in ratio and trend analysis. Your task is to provide accurate 
            financial analysis based on the provided documents.
            
            CONTEXT INFORMATION:
            {context}
            
            ANALYSIS TYPE: {task_type}
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Provide a clear, concise, and accurate response based solely on the information in the context. 
            Focus on financial insights, metrics, and trends. Include specific numbers and data points when available.
            If the context doesn't contain enough information to fully answer the query, acknowledge the limitations in your response.
            """
            
        elif task in [FinancialTaskType.INCOME_STATEMENT_ANALYSIS, FinancialTaskType.BALANCE_SHEET_ANALYSIS, 
                     FinancialTaskType.CASH_FLOW_ANALYSIS]:
            return """
            You are an expert financial statement analyst. Your task is to analyze financial statements based on the provided documents.
            
            CONTEXT INFORMATION:
            {context}
            
            ANALYSIS TYPE: {task_type}
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Provide a detailed analysis of the financial statements in the context. Focus on key line items and their significance.
            For income statements: Focus on revenue, expenses, profitability metrics, and operational efficiency.
            For balance sheets: Analyze assets, liabilities, liquidity, solvency, and capital structure.
            For cash flow statements: Examine operational, investing, and financing cash flows, and cash management efficiency.
            Include specific figures, calculate relevant metrics, and explain their implications.
            Compare periods if multiple time periods are available in the context.
            """
        
        elif task in [FinancialTaskType.VALUATION, FinancialTaskType.FORECASTING]:
            return """
            You are an expert in financial valuation and forecasting. Your task is to provide valuation insights 
            or financial projections based on the provided documents.
            
            CONTEXT INFORMATION:
            {context}
            
            ANALYSIS TYPE: {task_type}
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Provide a detailed valuation analysis or forecast based solely on the information in the context.
            For valuations: Discuss valuation methods applicable to the data, key value drivers, and justification for multiples or assumptions.
            For forecasts: Provide projected figures, growth assumptions, and confidence intervals when possible.
            Be explicit about any assumptions you make and note where additional information would improve the analysis.
            Present your analysis in a structured format with clear reasoning for conclusions.
            """
        
        elif task in [FinancialTaskType.RISK_ASSESSMENT, FinancialTaskType.RED_FLAG_IDENTIFICATION]:
            return """
            You are an expert in financial risk analysis. Your task is to identify and assess financial risks 
            based on the provided documents.
            
            CONTEXT INFORMATION:
            {context}
            
            ANALYSIS TYPE: {task_type}
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Provide a comprehensive risk analysis based on the information in the context.
            Identify specific risk factors, red flags, or areas of concern in the financial data.
            Categorize risks by type (market, credit, operational, liquidity, etc.) when possible.
            Assess the potential impact and likelihood of each identified risk.
            Suggest potential mitigation strategies if appropriate.
            Be factual and objective, avoiding speculation beyond what the data supports.
            """
        
        elif task in [FinancialTaskType.DUE_DILIGENCE, FinancialTaskType.COMPLIANCE_CHECK]:
            return """
            You are an expert in financial due diligence and compliance. Your task is to provide insights 
            for due diligence or compliance purposes based on the provided documents.
            
            CONTEXT INFORMATION:
            {context}
            
            ANALYSIS TYPE: {task_type}
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Provide a structured due diligence analysis based on the information in the context.
            Focus on key areas of concern for M&A or investment decisions, including financial performance, risks, and compliance issues.
            Highlight any material findings that could impact valuation or transaction decisions.
            For compliance checks, assess adherence to relevant accounting standards, regulations, or reporting requirements.
            Identify any gaps in information that would be critical for completing a thorough due diligence review.
            """
        
        elif task in [FinancialTaskType.DOCUMENT_EXTRACTION, FinancialTaskType.TABLE_INTERPRETATION]:
            return """
            You are an expert in financial document and data extraction. Your task is to extract and interpret 
            specific financial information from the provided documents.
            
            CONTEXT INFORMATION:
            {context}
            
            EXTRACTION TYPE: {task_type}
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Extract the specific financial information requested in the query from the context.
            For table interpretation, present the data in a clear, structured format that preserves the relationships between data points.
            For document extraction, focus on extracting the exact figures, dates, and facts requested.
            When extracting numerical data, maintain the original units and notation (currency symbols, percentages, etc.).
            If the information cannot be found in the context, clearly state this rather than making assumptions.
            """
        
        elif task == FinancialTaskType.DOCUMENT_SUMMARIZATION:
            return """
            You are a financial document summarization specialist. Your task is to provide accurate and concise summaries 
            of financial documents.
            
            CONTEXT INFORMATION:
            {context}
            
            SUMMARY TYPE: {task_type}
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Provide a clear and structured summary of the financial information in the context. 
            Organize your summary by main themes, key financial data points, and important findings.
            Highlight the most critical financial information. Use bullet points where appropriate.
            If the context spans multiple documents, synthesize the information into a cohesive summary.
            
            Include these sections in your summary:
            1. Overview of document type and purpose
            2. Key financial highlights and metrics
            3. Notable trends or patterns
            4. Material risks or concerns (if any)
            5. Limitations of the information provided
            """
            
        # Standard task prompts from LLMTask if not a specialized financial task
        elif task == LLMTask.FINANCIAL_ANALYSIS:
            return """
            You are an expert financial analyst assistant. Your task is to provide accurate financial analysis based on the provided documents.
            
            CONTEXT INFORMATION:
            {context}
            
            USER QUERY: {query}
            CURRENT DATE: {current_date}
            
            Provide a clear, concise, and accurate response based solely on the information in the context. 
            Focus on financial insights, metrics, and trends. Include specific numbers and data points when available.
            If the context doesn't contain enough information to fully answer the query, acknowledge the limitations in your response.
            """
            
        elif task == LLMTask.DOCUMENT_SUMMARIZATION:
            return """
            You are a financial document summarization assistant. Your task is to provide accurate and concise summaries of financial documents.
            
            CONTEXT INFORMATION:
            {context}
Summary:"""
        
        else:  # Default query response prompt
            return """You are a financial due diligence expert assisting with M&A transactions. 
            
User Query: {query}

Here is relevant information from financial documents:

{context}

Based on the financial information provided, please answer the query. If the information is not in the provided context, say you don't have enough information to answer accurately rather than making up facts. 

Your response should be well-structured and insightful, highlighting key financial metrics and considerations that would be important for investment banking professionals evaluating this transaction.

Answer:"""
