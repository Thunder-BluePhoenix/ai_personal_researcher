# agents/advisor.py
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from utils.prompts import advisor_prompt

class AdvisorAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.tools = []
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=advisor_prompt
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
            "verified_information": state["verified_information"]
        })
        
        return {
            **state,
            "recommendations": result["output"]
        }