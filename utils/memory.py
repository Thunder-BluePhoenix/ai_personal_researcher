# utils/memory.py
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

def create_memory():
    """Create a memory component for contextual conversation."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        output_key="final_report"
    )