# agents/fact_checker.py
import os
import re
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from utils.prompts import fact_checker_prompt
from langchain_core.tools import Tool  # Add this import

class FactCheckerAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create a dummy tool
        dummy_tool = Tool(
            name="no_op_fact_checker",
            description="This tool does nothing but satisfies the OpenAI functions requirement.",
            func=lambda x: "Tool not used"
        )
        
        self.tools = [dummy_tool]  # Include the dummy tool in the tools list
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=fact_checker_prompt
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
            "summary": state["summary"]
        })
        
        # Calculate confidence score based on verification results
        confidence = self._calculate_confidence(result["output"])
        
        return {
            **state,
            "verified_information": result["output"],
            "confidence": confidence
        }
    
    def _calculate_confidence(self, verification_text):
        # Simple confidence calculation - count high/medium/low ratings
        high_count = len(re.findall(r'High', verification_text, re.IGNORECASE))
        medium_count = len(re.findall(r'Medium', verification_text, re.IGNORECASE))
        low_count = len(re.findall(r'Low', verification_text, re.IGNORECASE))
        
        total = high_count + medium_count + low_count
        if total == 0:
            return 0.7  # Default confidence
            
        # Weighted confidence calculation
        confidence = (high_count * 1.0 + medium_count * 0.7 + low_count * 0.3) / total
        return confidence