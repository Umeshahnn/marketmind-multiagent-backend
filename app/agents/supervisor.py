import logging
import time
from typing import Dict
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import SupervisorState
from app.utils.llm import get_groq_llm

logger = logging.getLogger("marketmind.agents.supervisor")

def create_supervisor_chain():
    """Creates the supervisor decision chain"""
    llm = get_groq_llm(temperature=0.0, max_tokens=20)
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of agents:
        
1. Researcher - Gathers information and data
2. Analyst - Analyzes data and provides insights  
3. Writer - Creates reports and summaries

Based on the current state and conversation, decide which agent should work next.
If the task is complete, respond with 'DONE'.

Current state:
- Has research data: {has_research}
- Has analysis: {has_analysis}
- Has report: {has_report}

Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.
"""),
        ("human", "{task}")
    ])
    return supervisor_prompt | llm

def supervisor_agent(state: SupervisorState) -> Dict:
    """Supervisor decides next agent using Groq LLM"""
    start_time = time.time()
    messages = state.get("messages", [])
    task = messages[-1].content if messages else (state.get("current_task") or "No task")
    
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))
    
    logger.info(f"🤖 [SUPERVISOR] Evaluating state -> Task: '{task[:60]}...' | Research: {has_research} | Analysis: {has_analysis} | Report: {has_report}")

    try:
        chain = create_supervisor_chain()
        decision = chain.invoke({
            "task": task,
            "has_research": has_research,
            "has_analysis": has_analysis,
            "has_report": has_report
        })
        
        decision_text = str(decision.content).strip().lower()
        duration = (time.time() - start_time) * 1000
        logger.info(f"🤖 [SUPERVISOR] Groq Decision: '{decision_text}' (took {duration:.1f}ms)")
        
        if "done" in decision_text or has_report:
            next_agent = "end"
            supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
        elif "researcher" in decision_text or not has_research:
            next_agent = "researcher"
            supervisor_msg = "📋 Supervisor: Let's start with research. Assigning to Researcher..."
        elif "analyst" in decision_text or (has_research and not has_analysis):
            next_agent = "analyst"
            supervisor_msg = "📋 Supervisor: Research done. Time for analysis. Assigning to Analyst..."
        elif "writer" in decision_text or (has_analysis and not has_report):
            next_agent = "writer"
            supervisor_msg = "📋 Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
        else:
            next_agent = "end"
            supervisor_msg = "✅ Supervisor: Task seems complete."
        
        logger.info(f"🤖 [SUPERVISOR] Next Agent Selected: '{next_agent}'")

        return {
            "messages": [AIMessage(content=supervisor_msg)],
            "next_agent": next_agent,
            "current_task": task
        }
    except Exception as e:
        logger.error(f"🤖 [SUPERVISOR] Exception during decision: {str(e)}", exc_info=True)
        raise
