import logging
import time
from typing import Dict
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from app.graph.state import SupervisorState
from app.utils.llm import get_groq_llm

logger = logging.getLogger("marketmind.agents.writer")

def writer_agent(state: SupervisorState) -> Dict:
    """Writer agent using Groq"""
    start_time = time.time()
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    task = state.get("current_task", "")
    sources = state.get("sources", [])
    
    logger.info(f"✍️ [WRITER] Starting report composition for task: '{task}'")

    try:
        llm = get_groq_llm(temperature=0.3, max_tokens=700)
        writing_prompt = f"""As a professional writer, create an executive report based on:

Task: {task}

Research Findings:
{research_data[:1000]}

Analysis:
{analysis[:1000]}

Create a well-structured report with:
1. Executive Summary
2. Key Findings  
3. Analysis & Insights
4. Recommendations
5. Conclusion

Keep it professional and concise."""
        
        report_response = llm.invoke([HumanMessage(content=writing_prompt)])
        report = str(report_response.content)
        
        sources_section = "\n\n## Sources & Web References\n"
        if sources:
            for idx, src in enumerate(sources, 1):
                title = src.get("title", "Web Link")
                url = src.get("url", "#")
                sources_section += f"{idx}. [{title}]({url})\n"
        else:
            sources_section += "*Search data synthesized via Tavily Search API.*\n"

        final_report = f"""
📄 FINAL REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Topic: {task}
{'='*50}

{report}
{sources_section}
{'='*50}
Report compiled by Multi-Agent AI System powered by Groq
"""
        duration = (time.time() - start_time) * 1000
        logger.info(f"✍️ [WRITER] Report compilation completed in {duration:.1f}ms | Final Report Length: {len(final_report)} chars")

        return {
            "messages": [AIMessage(content="✍️ Writer: Report complete! See below for the full document.")],
            "final_report": final_report,
            "next_agent": "supervisor",
            "task_complete": True
        }
    except Exception as e:
        logger.error(f"✍️ [WRITER] Exception during report composition: {str(e)}", exc_info=True)
        raise
