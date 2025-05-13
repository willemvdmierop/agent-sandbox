from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(dotenv_path='/Users/willemvandemierop/Desktop/ai_agent_project/.env')

def load_and_process_pdf(pdf_path):
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

def create_pdf_agent(pdf_path):
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    # Create vector store from PDF
    vectorstore = load_and_process_pdf(pdf_path)
    
    # Create the PDF search tool
    pdf_search = Tool(
        name="search_pdf",
        description="Use this tool when you need to search through the PDF document for specific information. Input should be a search query.",
        func=vectorstore.similarity_search
    )
    
    # Create the direct answer tool
    direct_answer = Tool(
        name="direct_answer",
        description="Use this tool when you can answer the question directly without consulting the PDF. This is for general knowledge questions or questions that don't require specific information from the document.",
        func=lambda x: "I can answer this question directly without consulting the PDF."
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that can either answer questions directly or consult a PDF document when needed.
        You have access to two tools:
        1. search_pdf: Use this when you need specific information from the PDF
        2. direct_answer: Use this when you can answer without consulting the PDF
        
        Always think carefully about whether you need to consult the PDF or not.
        If the question is about general knowledge or doesn't require specific information from the PDF, use direct_answer.
        If the question requires specific information from the PDF, use search_pdf.
        
        When using search_pdf, make sure to provide a clear and specific search query."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the agent
    agent = create_openai_functions_agent(llm, [pdf_search, direct_answer], prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[pdf_search, direct_answer],
        memory=memory,
        verbose=True
    )
    
    return agent_executor

def main():
    # Path to your PDF
    pdf_path = "/Users/willemvandemierop/Desktop/sandbox_owen/Syndicusovereenkomst_michiel.pdf"
    
    print("Initializing PDF Agent...")
    agent = create_pdf_agent(pdf_path)
    
    print("\nPDF Agent System")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        try:
            # Get the answer
            response = agent.invoke({"input": question})
            print("\nAnswer:", response["output"])
                    
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 