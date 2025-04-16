# agents/researcher.py
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import BaseTool
from utils.prompts import researcher_prompt
from utils.tools import create_search_tool, create_web_loader_tool
import json

class ResearcherAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.tools = [create_search_tool(), create_web_loader_tool()]
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=researcher_prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def run(self, query):
        result = await self.agent_executor.ainvoke({"query": query})
        # Structure the output for passing to next agent
        return {
            "query": query,
            "research_findings": result["output"],
            "sources": self._extract_sources(result["output"])
        }
    
    def _extract_sources(self, text):
        # Simple extraction - in production use regex or better parsing
        sources = []
        for line in text.split("\n"):
            if "http" in line:
                sources.append(line.strip())
        return sources