import time
import sys
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from app.graph.workflow import marketmind_graph

def run():
    query = "Analyze the Indian EV market."
    print(f"Starting test for query: '{query}'", flush=True)
    t0 = time.time()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_task": query,
        "next_agent": "supervisor",
        "research_data": "",
        "sources": [],
        "analysis": "",
        "final_report": "",
        "task_complete": False,
        "steps_completed": []
    }
    
    res = marketmind_graph.invoke(initial_state)
    elapsed = time.time() - t0
    
    print(f"SUCCESS! Total Execution Time: {elapsed:.2f} seconds", flush=True)
    print(f"Steps sequence: {res.get('steps_completed')}", flush=True)
    print(f"Sources found: {len(res.get('sources', []))}", flush=True)

if __name__ == "__main__":
    run()
