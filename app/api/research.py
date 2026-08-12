import time
import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from app.graph.workflow import marketmind_graph

logger = logging.getLogger("marketmind.api.research")

router = APIRouter()

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research question or market topic")

class SourceItem(BaseModel):
    title: str
    url: str

class ResearchResponse(BaseModel):
    status: str
    query: str
    report: Optional[str] = None
    sources: Optional[List[SourceItem]] = []
    steps_completed: Optional[List[str]] = []
    message: Optional[str] = None

@router.post("/research", response_model=ResearchResponse)
def execute_research(request: ResearchRequest):
    """
    Triggers the LangGraph multi-agent market intelligence workflow.
    Supervisor -> Researcher -> Analyst -> Writer -> Final Report.
    """
    start_time = time.time()
    query_text = request.query.strip()
    logger.info(f"🚀 [API] Received research request: '{query_text}'")

    if not query_text:
        logger.warning("🚀 [API] Empty query text received in research request.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter cannot be empty."
        )

    initial_state = {
        "messages": [HumanMessage(content=query_text)],
        "current_task": query_text,
        "next_agent": "supervisor",
        "research_data": "",
        "sources": [],
        "analysis": "",
        "final_report": "",
        "task_complete": False,
        "steps_completed": []
    }

    try:
        logger.info("🚀 [API] Invoking LangGraph marketmind_graph...")
        final_state = marketmind_graph.invoke(initial_state)
        duration = (time.time() - start_time) * 1000

        report = final_state.get("final_report", "")
        sources = final_state.get("sources", [])
        steps = final_state.get("steps_completed", ["supervisor", "researcher", "analyst", "writer"])

        logger.info(f"🚀 [API] Workflow completed in {duration:.1f}ms | Report generated ({len(report)} chars) | Sources: {len(sources)}")

        if not report:
            logger.error("🚀 [API] Workflow finished but final_report is empty!")
            raise RuntimeError("Workflow finished but failed to generate a report.")

        return ResearchResponse(
            status="success",
            query=query_text,
            report=report,
            sources=[SourceItem(title=s.get("title", "Link"), url=s.get("url", "#")) for s in sources],
            steps_completed=steps
        )

    except ValueError as ve:
        duration = (time.time() - start_time) * 1000
        logger.error(f"❌ [API] Configuration ValueError after {duration:.1f}ms: {str(ve)}", exc_info=True)
        return ResearchResponse(
            status="error",
            query=query_text,
            message=f"Configuration error: {str(ve)}"
        )
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"❌ [API] Agent Workflow Exception after {duration:.1f}ms: {str(e)}", exc_info=True)
        return ResearchResponse(
            status="error",
            query=query_text,
            message=f"Agent execution error: {str(e)}"
        )
