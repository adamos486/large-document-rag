# LLM Integration Guide

This document explains how the Financial Due Diligence RAG system integrates with Large Language Models (LLMs), with a focus on the hybrid multi-vendor approach that combines OpenAI and Anthropic models. The system includes a specialized financial task detection system to intelligently route specific financial tasks to the most suitable LLM.

## Supported LLM Providers

The system supports multiple LLM providers through a flexible, modular architecture:

1. **OpenAI** - Integrated through both direct API calls and LangChain
2. **Anthropic Claude** - Support for Claude models with specialized financial capabilities
3. **Hybrid Mode** - Task-optimized approach using different models for different financial tasks

## Configuration

### Environment Variables

Set the following environment variables in your `.env` file:

```
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Provider Selection (one of: "openai", "anthropic", "hybrid")
LLM_PROVIDER=hybrid
```

### Configuration Settings

Configure LLM behavior in `src/config/config.py`:

```python
# LLM settings
LLM_PROVIDER: str = "openai"  # 'openai', 'anthropic', or 'hybrid'
LLM_MODEL: str = "gpt-4"  # Default model for OpenAI
ANTHROPIC_MODEL: str = "claude-3-opus-20240229"  # Default model for Anthropic
TEMPERATURE: float = 0.0

# Task-specific LLM mapping for hybrid mode
HYBRID_LLM_TASKS: Dict[str, str] = {
    "financial_analysis": "anthropic",
    "document_summarization": "anthropic",
    "query_response": "openai",
    "default": "openai"
}
```

## Financial Task Detection System

The system implements a sophisticated financial task detection mechanism that analyzes queries to determine the specific type of financial task being requested and route it to the most appropriate LLM.

### Task Types

The system recognizes the following specialized financial task types:

| Task Type | Description | Preferred LLM |
|-----------|-------------|---------------|
| `RATIO_ANALYSIS` | Analysis of financial ratios and metrics | Anthropic |
| `TREND_ANALYSIS` | Analysis of financial trends over time | Hybrid |
| `VALUATION` | Company valuation and multiples analysis | Anthropic |
| `RISK_ASSESSMENT` | Financial risk analysis | Anthropic |
| `FORECASTING` | Financial projections and forecasts | Hybrid |
| `DOCUMENT_EXTRACTION` | Extracting specific data from documents | OpenAI |
| `DOCUMENT_SUMMARIZATION` | Summarizing financial documents | OpenAI |
| `TABLE_INTERPRETATION` | Interpreting financial tables and data | OpenAI |
| `INCOME_STATEMENT_ANALYSIS` | Analysis of income statements | Hybrid |
| `BALANCE_SHEET_ANALYSIS` | Analysis of balance sheets | Hybrid |
| `CASH_FLOW_ANALYSIS` | Analysis of cash flow statements | Hybrid |
| `DUE_DILIGENCE` | M&A due diligence analysis | Anthropic |
| `RED_FLAG_IDENTIFICATION` | Identifying financial warning signs | Anthropic |
| `COMPLIANCE_CHECK` | Regulatory and compliance analysis | Anthropic |

### How Task Detection Works

1. **Pattern Analysis**: The system analyzes the query text using specialized financial patterns for each task type
2. **Confidence Scoring**: Multiple task types may be detected with different confidence scores
3. **Complexity Assessment**: The system evaluates task complexity based on linguistic markers
4. **Financial Entity Extraction**: Monetary values, percentages, company names, and other financial entities are extracted
5. **LLM Selection**: The most appropriate LLM is selected based on task type, complexity, and confidence scores

### Hybrid Mode Configuration

Beyond the basic hybrid mode, the financial task detector provides more granular routing:

```python
# Example task-LLM mapping in financial_task_detector.py
task_llm_mapping = {
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
    
    # Hybrid tasks - could use either depending on complexity
    FinancialTaskType.TREND_ANALYSIS: "hybrid",
    FinancialTaskType.FORECASTING: "hybrid",
    FinancialTaskType.INCOME_STATEMENT_ANALYSIS: "hybrid",
    FinancialTaskType.BALANCE_SHEET_ANALYSIS: "hybrid",
    FinancialTaskType.CASH_FLOW_ANALYSIS: "hybrid"
}
```

## Provider Details

### OpenAI Models

The system supports all OpenAI models through LangChain integration:

| Model | Strengths for Financial Due Diligence |
|-------|---------------------------------------|
| gpt-4 | Most capable model for complex financial reasoning |
| gpt-3.5-turbo | Fast responses for simpler financial queries |

### Anthropic Claude Models

Support for Anthropic Claude models with specific advantages for financial analysis:

| Model | Strengths for Financial Due Diligence |
|-------|---------------------------------------|
| claude-3-opus | Superior handling of financial documents and longer context |
| claude-3-sonnet | Good balance of performance and cost for financial tasks |
| claude-3-haiku | Faster responses for simple financial queries |

## Hybrid Task-Specific Approach

The Financial Due Diligence RAG system implements a sophisticated task detection system that automatically routes different financial tasks to the LLM best suited for that specific task:

### 1. Financial Analysis Tasks

Claude models are optimized for:
- Analyzing financial statements
- Risk assessment
- Ratio analysis
- Spotting financial irregularities
- Evaluating capital structure

Example task: "Analyze the target company's debt-to-equity ratio trend over the past 5 years"

### 2. Document Summarization Tasks

Claude models excel at:
- Processing longer financial documents
- Maintaining context across large reports
- Creating structured summaries
- Extracting key financial metrics from large documents

Example task: "Summarize the key findings from this 50-page due diligence report"

### 3. Query Response Tasks

OpenAI models are well-suited for:
- Direct question answering
- Specific financial lookups
- Concise explanations of financial concepts
- Integration with existing OpenAI-based systems

Example task: "What was the EBITDA margin reported in Q3 2023?"

## Task Detection Logic

The system implements automatic task detection logic to determine the most appropriate LLM for each query:

```python
def _determine_query_type(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Determines the task type based on query content and retrieved documents
    """
    query_lower = query.lower()
    
    # Check for financial analysis indicators
    financial_keywords = [
        "financial performance", "balance sheet", "income statement", "cash flow",
        "revenue", "profit margin", "debt", "liability", "asset", "valuation",
        "ebitda", "earnings", "fiscal", "quarter", "financial risk"
    ]
    
    # Check for document summarization indicators
    summarization_keywords = [
        "summarize", "summary", "overview", "brief", "outline", "recap",
        "key points", "highlights"
    ]
    
    # Make task determination based on keyword matches
    if any(kw in query_lower for kw in financial_keywords):
        return "financial_analysis"
    elif any(kw in query_lower for kw in summarization_keywords):
        return "document_summarization"
    else:
        return "query_response"
```

## Task-Optimized Prompts

Each task type uses specialized prompts designed to get the best results from each model:

### Financial Analysis Prompt

Optimized for Claude's strengths in detailed financial analysis:

```
You are a senior financial analyst at a top investment bank specializing in M&A due diligence.

User Query: {query}

Here is relevant information from financial documents:

{context}

Provide a comprehensive financial analysis based on the above information. Include:
1. Key financial metrics and their implications
2. Trend analysis if temporal data is present
3. Risk assessment of any identified financial concerns
4. Areas where additional financial information would be beneficial

Format your analysis in a clear, professional structure suitable for investment banking professionals.
```

### Document Summarization Prompt

Leverages Claude's larger context window and summarization capabilities:

```
You are a financial document specialist focusing on extracting key information for M&A due diligence.

User Query: {query}

Here are the documents to analyze:

{context}

Provide a comprehensive yet concise summary that captures the most important financial information. Structure your summary with clear sections covering:

1. Executive overview
2. Key financial figures and metrics
3. Notable risks or concerns
4. Areas that merit deeper investigation

Use bulletpoints and formatting to enhance readability.
```

### Query Response Prompt

Optimized for OpenAI's direct question answering:

```
You are a financial due diligence expert assisting with M&A transactions.

User Query: {query}

Here is relevant information from financial documents:

{context}

Based on the financial information provided, please answer the query.
Your response should be well-structured and insightful, highlighting key financial metrics and considerations that would be important for investment banking professionals evaluating this transaction.
```

## Extending LLM Support

The modular LLM provider architecture makes it easy to add support for additional LLM providers or models:

1. Update the `LLMProvider` enum in `src/llm/llm_provider.py`
2. Add a new provider creation method in the `LLMProviderFactory` class
3. Update the configuration options in `src/config/config.py`

The system is designed to gracefully handle provider unavailability by falling back to alternative providers when necessary.

## Best Practices for Financial LLM Integration

1. **Use Specialized Task Detection**: The financial task detection system provides more intelligent routing than basic hybrid mode for financial applications.

2. **Task-Optimized Provider Selection**: 
   - Use Anthropic Claude for complex financial analysis, risk assessment, and due diligence tasks
   - Use OpenAI for document extraction, summarization, and table interpretation
   - For mixed or uncertain tasks, let the system detect the best provider based on the query

3. **Provider-Optimized Prompts**: The system includes specialized prompts for each task type and provider combination to maximize performance.

4. **Set Appropriate Temperature**: 
   - Use lower temperature (0.0-0.3) for detailed financial analysis and compliance checks
   - Use moderate temperature (0.4-0.6) for document summarization and trend analysis
   - Use higher temperature (0.6-0.8) only for brainstorming potential interpretations

5. **Pass Query Context to Task Detector**: When using LLMs programmatically, always pass the full query text to the task detector:
   ```python
   # Correct usage - passes query for intelligent routing
   llm = llm_provider_factory.get_llm_for_task(task=LLMTask.DEFAULT, query=query_text)
   ```

6. **Leverage Financial Entity Extraction**: Use the extracted financial entities for enhanced understanding of query intent:
   ```python
   task_type, confidence_scores, financial_entities = llm_provider_factory.analyze_financial_query(query)
   # Use the extracted entities for additional context
   ```

7. **Handle Task-Specific Errors**: Use the specialized exception types for better error handling:
   ```python
   from src.exceptions import LLMProviderError, QueryProcessingError
   
   try:
       response = query_agent.run(query)
   except LLMProviderError as e:
       # Handle LLM-specific errors (API failures, token limits)
   except QueryProcessingError as e:
       # Handle query processing errors
   ```

8. **Consider Confidence Scores**: For critical financial analysis, check the confidence score of task detection and consider using a more deterministic approach for low-confidence cases.

## API Integration

The system provides API endpoints that allow clients to specify which LLM provider to use for specific queries, or to leverage the automated task detection system.

### LLM Provider Selection in API Requests

When making a query request, you can specify the LLM provider to use:

```json
{
  "query": "What is the trend in gross profit margin over the last 3 quarters?",
  "collection_name": "financial_reports",
  "n_results": 5,
  "llm_provider": "anthropic"  // Override provider selection
}
```

### Task Type Information in API Responses

API responses include information about the detected financial task type and the LLM that was used:

```json
{
  "query": "What is the trend in gross profit margin over the last 3 quarters?",
  "llm_response": "The gross profit margin shows an upward trend...",
  "financial_task_type": "trend_analysis",
  "llm_provider": "anthropic",
  "confidence_score": 0.85,
  "financial_entities": {
    "percentages": ["gross profit margin"],
    "time_periods": ["last 3 quarters"]  
  },
  "processing_time": 2.34
}
```

## Using LLMs in Your Application Code

### Per-Query LLM Selection

You can select the LLM provider on a per-query basis using the `llm_provider_factory`:

```python
# Select Anthropic Claude for a specific query
llm = llm_provider_factory.get_llm_for_task(task=LLMTask.DEFAULT, query="What is the company's current market capitalization?", llm_provider="anthropic")
response = llm.run(query)
```

### Task Detection and Routing

Alternatively, you can use the automated task detection system to route the query to the best LLM provider:

```python
# Use task detection to select the best LLM provider
task_type, confidence_scores, financial_entities = llm_provider_factory.analyze_financial_query(query)
llm = llm_provider_factory.get_llm_for_task(task=task_type, query=query)
response = llm.run(query)
```

## Troubleshooting

### Common Issues

1. **Invalid API Key**
   - Error: "Authentication error" or "Invalid API key"
   - Solution: Verify your API keys are correct and properly set in the .env file

2. **Rate Limiting**
   - Error: "Rate limit exceeded" or "Too many requests"
   - Solution: Implement request batching or throttling in your application

3. **Model Unavailability**
   - Error: "Model not available" or "Model capacity exceeded"
   - Solution: The system will automatically fall back to alternative models; you can also configure backup models

4. **Context Length Exceeded**
   - Error: "Maximum context length exceeded"
   - Solution: Use Claude for larger documents or adjust the chunking strategy in the configuration
