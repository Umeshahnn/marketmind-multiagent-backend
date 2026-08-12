import os
import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger("marketmind.tavily")

def execute_tavily_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Executes web search via native TavilyClient SDK with fallback to LangChain integration.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key or tavily_key.strip() == "" or "your_tavily_api_key" in tavily_key:
        logger.error("TAVILY_API_KEY is missing or invalid in environment variables!")
        raise ValueError("TAVILY_API_KEY is not configured in environment variables.")

    logger.info(f"Starting Tavily Web Search for query: '{query}' (max_results={max_results})")
    start_time = time.time()

    try:
        raw_results = []
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query=query, max_results=max_results)
            raw_results = response.get("results", [])
            logger.info("Executed Tavily via native TavilyClient Python SDK.")
        except Exception as sdk_err:
            logger.warning(f"Native TavilyClient failed ({str(sdk_err)}), falling back to TavilySearchResults.")
            from langchain_community.tools.tavily_search import TavilySearchResults
            search_tool = TavilySearchResults(max_results=max_results)
            raw_results = search_tool.invoke({"query": query})
        
        sources: List[Dict[str, str]] = []
        formatted_snippets = []

        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    title = item.get("title", item.get("url", "Web Resource"))
                    url = item.get("url", "#")
                    content = item.get("content", "")[:250]
                    sources.append({"title": title, "url": url})
                    formatted_snippets.append(f"Source: {title} ({url})\nSnippet: {content}")
                else:
                    formatted_snippets.append(str(item)[:250])

        duration = (time.time() - start_time) * 1000
        logger.info(f"Tavily Search completed in {duration:.1f}ms | Found {len(sources)} sources.")

        results_text = "\n---\n".join(formatted_snippets) if formatted_snippets else "No search results returned."
        
        return {
            "text": results_text,
            "sources": sources
        }
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Tavily Search failed after {duration:.1f}ms: {str(e)}", exc_info=True)
        raise RuntimeError(f"Tavily search failed for query '{query}': {str(e)}")
