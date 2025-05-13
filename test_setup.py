import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(dotenv_path='/Users/willemvandemierop/Desktop/ai_agent_project/.env')

# Let's check if we loaded the right key
print('key; ',os.getenv("OPENAI_API_KEY"))

# Initialize the ChatOpenAI instance
llm = ChatOpenAI(model = "gpt-4o-mini")

# Test the setup
response = llm.invoke("Hello! Are you working?")
print(response.content)

