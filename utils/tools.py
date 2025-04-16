# utils/tools.py
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# Search tool
def create_search_tool():
    search = DuckDuckGoSearchResults()
    return Tool(
        name="web_search",
        description="Search the web for information on a topic.",
        func=search.run
    )

# Web content loader tool
def create_web_loader_tool():
    def load_and_process_url(url):
        loader = WebBaseLoader(url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    return Tool(
        name="web_loader",
        description="Load and process content from a web URL.",
        func=load_and_process_url
    )

# Vector store tool
def create_vector_store(documents, query):
    # Use another embedding model, e.g. HuggingFace embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    relevant_docs = vectorstore.similarity_search(query, k=5)
    return relevant_docs



# utils/tools.py (add these functions)
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import NewsApiIO

def create_arxiv_tool():
    """Create a tool for searching academic papers on Arxiv."""
    arxiv_tool = ArxivQueryRun()
    return Tool(
        name="arxiv_search",
        description="Search for academic papers on a topic.",
        func=arxiv_tool.run
    )

def create_news_tool():
    """Create a tool for searching recent news articles."""
    news_api = NewsApiIO()
    
    def search_news(query):
        return news_api.run(query)
    
    return Tool(
        name="news_search",
        description="Search for recent news articles on a topic.",
        func=search_news
    )