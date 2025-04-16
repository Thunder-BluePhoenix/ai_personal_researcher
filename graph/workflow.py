# graph/workflow.py
from typing import Dict, Any, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import langgraph as lg
from langgraph.graph import StateGraph, END
from agents.researcher import ResearcherAgent
from agents.summarizer import SummarizerAgent
from agents.fact_checker import FactCheckerAgent
from agents.advisor import AdvisorAgent
from agents.reporter import ReporterAgent

# Define state
class AgentState(TypedDict):
    query: str
    research_findings: str
    summary: str
    verified_information: str
    recommendations: str
    final_report: str
    confidence: float
    
# Define agent nodes
researcher = ResearcherAgent()
summarizer = SummarizerAgent()
fact_checker = FactCheckerAgent()
advisor = AdvisorAgent()
reporter = ReporterAgent()

# Create graph
def create_agent_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("researcher", researcher.run)
    workflow.add_node("summarizer", summarizer.run)
    workflow.add_node("fact_checker", fact_checker.run)
    workflow.add_node("advisor", advisor.run)
    workflow.add_node("reporter", reporter.run)
    
    # Define edges
    workflow.add_edge("researcher", "summarizer")
    
    # Conditional edge: if confidence is low, go back to researcher
    def route_after_fact_check(state):
        if not state.get("confidence") or state.get("confidence", 0) < 0.7:
            return "researcher"
        return "advisor"
    
    workflow.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {
            "researcher": "researcher",
            "advisor": "advisor"
        }
    )
    
    workflow.add_edge("summarizer", "fact_checker")
    workflow.add_edge("advisor", "reporter")
    workflow.add_edge("reporter", END)
    
    # Set entry point
    workflow.set_entry_point("researcher")
    
    return workflow.compile()

# Create runnable graph
agent_graph = create_agent_workflow()