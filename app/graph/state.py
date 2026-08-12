from typing import List, Dict, Any
from langgraph.graph import MessagesState

class SupervisorState(MessagesState):
    """Exact state definition from Cell 16 of original notebook"""
    next_agent: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""
    sources: List[Dict[str, str]] = []
