from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv(dotenv_path='/Users/willemvandemierop/Desktop/ai_agent_project/.env')

# Initialize the LLM

llm = ChatOpenAI(temperature=0.9, model="gpt-4o-mini")
template = """

You are an AI assistant with expertise in data analysis and automation. Answer the following question: 
Question: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)


query = "what is the impact of AI in healthcare?"

response = chain.run(question = query)

print(f"Agent response: {response}")

