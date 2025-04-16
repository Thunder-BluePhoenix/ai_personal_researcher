# agents/summarizer.py
import os
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from utils.prompts import summarizer_prompt

class SummarizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create a dummy tool
        dummy_tool = Tool(
            name="no_op",
            description="This tool does nothing but is required by OpenAI functions agent.",
            func=lambda x: "Tool not used"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", summarizer_prompt.template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.agent = create_openai_functions_agent(self.llm, [dummy_tool], self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[dummy_tool],  # Include the dummy tool here too
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def run(self, state):
        # Prepare the inputs as expected by your prompt
        inputs = {
            "query": state["query"],
            "research_findings": state["research_findings"],
            "tools": "",
            "tool_names": "",
            "agent_scratchpad": []
        }
        
        result = await self.agent_executor.ainvoke(inputs)
        
        return {
            **state,
            "summary": result["output"]
        }