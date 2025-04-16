# agents/summarizer.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from utils.prompts import summarizer_prompt
from langchain_core.tools import Tool  # Add this import

class SummarizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create a dummy tool that doesn't do anything important
        dummy_tool = Tool(
            name="no_op_summarizer",
            description="This tool does nothing but satisfies the OpenAI functions requirement.",
            func=lambda x: "Tool not used"
        )
        
        # Instead of using react agent with no tools, use a simple chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", summarizer_prompt.template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Include the dummy tool in the agent creation
        self.agent = create_openai_functions_agent(self.llm, [dummy_tool], self.prompt)
        
        # Include the dummy tool in the executor as well
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[dummy_tool],
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