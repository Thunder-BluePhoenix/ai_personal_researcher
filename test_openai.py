# test_openai.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Check if OpenAI API key is available
api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API key exists: {bool(api_key)}")
if api_key:
    print(f"OpenAI API key starts with: {api_key[:5]}...")

try:
    # Create the OpenAI model
    openai_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # Simple test
    response = openai_model.invoke("What is the capital of France?")
    print("Test successful!")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"Test failed with error: {e}")