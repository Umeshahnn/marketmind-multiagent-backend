from typing import Literal
from langgraph.graph import StateGraph, END
from app.graph.state import SupervisorState
from app.agents.supervisor import supervisor_agent
from app.agents.researcher import researcher_agent
from app.agents.analyst import analyst_agent
from app.agents.writer import writer_agent

def router(state: SupervisorState) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    """Routes to next agent based on state (Cell 22 of original notebook)"""
    next_agent = state.get("next_agent", "supervisor")
    
    if next_agent == "end" or state.get("task_complete", False):
        return END
        
    if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
        return next_agent
        
    return "supervisor"

def create_marketmind_graph():
    """Builds workflow graph (Cell 23 of original notebook)"""
    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("writer", writer_agent)

    workflow.set_entry_point("supervisor")

    for node in ["supervisor", "researcher", "analyst", "writer"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "supervisor": "supervisor",
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END
            }
        )

    return workflow.compile()

marketmind_graph = create_marketmind_graph()
