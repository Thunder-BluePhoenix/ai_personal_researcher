# main.py
import os
from dotenv import load_dotenv
from graph.workflow import agent_graph
import asyncio
import json

# Load environment variables
load_dotenv()

# Initialize LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

async def run_research_workflow(query):
    """Run the complete research workflow on a query."""
    try:
        result = await agent_graph.ainvoke({"query": query})
        return result["final_report"]
    except Exception as e:
        print(f"Error in research workflow: {e}")
        return f"Research workflow failed with error: {str(e)}"

# Example usage
if __name__ == "__main__":
    research_query = "What are the most sustainable solar panels in 2025?"
    report = asyncio.run(run_research_workflow(research_query))
    print("\n=== FINAL REPORT ===\n")
    print(report)