# main.py (with interactive input)
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from graph.workflow import agent_graph

# Apply nest_asyncio to prevent event loop errors
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

print(f"OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")
if os.getenv('OPENAI_API_KEY'):
    print(f"OPENAI_API_KEY starts with: {os.getenv('OPENAI_API_KEY')[:5]}...")

async def run_research_workflow(query):
    """Run the complete research workflow on a query."""
    try:
        result = await agent_graph.ainvoke({"query": query})
        return result["final_report"]
    except Exception as e:
        print(f"Error in research workflow: {str(e)}")
        return f"Research workflow failed with error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Get user input interactively
    research_query = input("Enter your research question: ")
    
    try:
        print(f"\nResearching: {research_query}\n")
        print("This may take a few minutes...\n")
        report = loop.run_until_complete(run_research_workflow(research_query))
        print("\n=== FINAL REPORT ===\n")
        print(report)
    finally:
        # Close the loop properly
        loop.close()