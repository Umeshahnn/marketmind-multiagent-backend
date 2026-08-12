import logging
import time
from typing import Dict
from langchain_core.messages import AIMessage, HumanMessage
from app.graph.state import SupervisorState
from app.utils.llm import get_groq_llm
from app.tools.tavily_search import execute_tavily_search

logger = logging.getLogger("marketmind.agents.researcher")

def researcher_agent(state: SupervisorState) -> Dict:
    """Researcher agent using Groq and Tavily"""
    start_time = time.time()
    task = state.get("current_task", "research topic")
    
    logger.info(f"🔍 [RESEARCHER] Starting research execution for task: '{task}'")

    # Gather search data via Tavily
    try:
        search_res = execute_tavily_search(task, max_results=3)
        web_snippets = search_res.get("text", "")
        sources = search_res.get("sources", [])
        logger.info(f"🔍 [RESEARCHER] Tavily search retrieved {len(sources)} sources.")
    except Exception as search_err:
        logger.warning(f"🔍 [RESEARCHER] Tavily search failed: {str(search_err)}. Continuing with empty web context.", exc_info=True)
        web_snippets = "No web search data available."
        sources = []

    try:
        llm = get_groq_llm(temperature=0.2, max_tokens=500)
        research_prompt = f"""As a research specialist, provide comprehensive information about: {task}

Web Search Data:
{web_snippets}

Include:
1. Key facts and background
2. Current trends or developments
3. Important statistics or data points
4. Notable examples or case studies

Be concise but thorough."""
        
        research_response = llm.invoke([HumanMessage(content=research_prompt)])
        research_data = str(research_response.content)
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"🔍 [RESEARCHER] Completed research synthesis in {duration:.1f}ms | Research Data Length: {len(research_data)} chars")

        agent_message = f"🔍 Researcher: I've completed the research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."
        
        return {
            "messages": [AIMessage(content=agent_message)],
            "research_data": research_data,
            "sources": sources,
            "next_agent": "supervisor"
        }
    except Exception as e:
        logger.error(f"🔍 [RESEARCHER] Failed to execute research synthesis: {str(e)}", exc_info=True)
        raise
