import logging
import time
from typing import Dict
from langchain_core.messages import AIMessage, HumanMessage
from app.graph.state import SupervisorState
from app.utils.llm import get_groq_llm

logger = logging.getLogger("marketmind.agents.analyst")

def analyst_agent(state: SupervisorState) -> Dict:
    """Analyst agent using Groq"""
    start_time = time.time()
    research_data = state.get("research_data", "")
    task = state.get("current_task", "")
    
    logger.info(f"📊 [ANALYST] Starting analysis execution for task: '{task}' (Research length: {len(research_data)} chars)")

    try:
        llm = get_groq_llm(temperature=0.2, max_tokens=500)
        analysis_prompt = f"""As a data analyst, analyze this research data and provide insights:

Research Data:
{research_data}

Provide:
1. Key insights and patterns
2. Strategic implications
3. Risks and opportunities
4. Recommendations

Focus on actionable insights related to: {task}"""
        
        analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis = str(analysis_response.content)
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"📊 [ANALYST] Completed analysis in {duration:.1f}ms | Analysis Length: {len(analysis)} chars")

        agent_message = f"📊 Analyst: I've completed the analysis.\n\nTop insights:\n{analysis[:400]}..."
        
        return {
            "messages": [AIMessage(content=agent_message)],
            "analysis": analysis,
            "next_agent": "supervisor"
        }
    except Exception as e:
        logger.error(f"📊 [ANALYST] Exception during analysis execution: {str(e)}", exc_info=True)
        raise
