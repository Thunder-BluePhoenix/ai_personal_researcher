# utils/tools.py
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Search tool
def create_search_tool():
    try:
        search = DuckDuckGoSearchResults()
        return Tool(
            name="web_search",
            description="Search the web for information on a topic.",
            func=search.run
        )
    except ImportError:
        # Fallback if DuckDuckGo isn't available
        def simple_search(query):
            return f"Search results for: {query}\n- Please install duckduckgo-search for real search results."
        
        return Tool(
            name="web_search",
            description="Search the web for information on a topic.",
            func=simple_search
        )

# Web content loader tool
# utils/tools.py - updated web_loader_tool

def create_web_loader_tool():
    def load_and_process_url(url):
        try:
            # Try to import required packages
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return "Error: Beautiful Soup (bs4) is not installed. Please install it with 'pip install bs4'."

            loader = WebBaseLoader(url)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            return f"Error loading URL: {str(e)}"
    
    return Tool(
        name="web_loader",
        description="Load and process content from a web URL.",
        func=load_and_process_url
    )

# Vector store tool
def create_vector_store(documents, query):
    try:
        # Try OpenAI embeddings first
        embeddings = OpenAIEmbeddings()
    except:
        # Fall back to HuggingFace if OpenAI isn't available
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    relevant_docs = vectorstore.similarity_search(query, k=5)
    return relevant_docs

# Optional tools that can be added
def create_arxiv_tool():
    """Create a tool for searching academic papers on Arxiv."""
    from langchain_community.tools import ArxivQueryRun
    arxiv_tool = ArxivQueryRun()
    return Tool(
        name="arxiv_search",
        description="Search for academic papers on a topic.",
        func=arxiv_tool.run
    )