# agents/reporter.py
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from utils.prompts import reporter_prompt
from langchain_core.tools import Tool  # Add this import

class ReporterAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create a dummy tool
        dummy_tool = Tool(
            name="no_op_reporter",
            description="This tool does nothing but satisfies the OpenAI functions requirement.",
            func=lambda x: "Tool not used"
        )
        
        self.tools = [dummy_tool]  # Include the dummy tool in the tools list
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=reporter_prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def run(self, state):
        result = await self.agent_executor.ainvoke({
            "query": state["query"],
            "research_summary": state["summary"],
            "fact_check_results": state["verified_information"],
            "recommendations": state["recommendations"]
        })
        
        return {
            **state,
            "final_report": result["output"]
        }