import os
from typing import List, Dict, Any, Annotated, TypedDict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
from document_processor import process_document, get_vectorstore_path
import json
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

# Define the state
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    context: List[Document]
    current_language: str

# Define the tools
@tool
def search_document(query: str, vectorstore: FAISS) -> List[Document]:
    """Search the document for relevant information"""
    return vectorstore.similarity_search(query, k=3)

# Define the nodes
def retrieve_context(state: AgentState, vectorstore: FAISS) -> AgentState:
    """Retrieve relevant context from the document"""
    last_message = state["messages"][-1]["content"]
    print(f"\nSearching document for: {last_message}")
    
    # Use similarity_search directly instead of the tool
    context = vectorstore.similarity_search(last_message, k=3)
    print(f"\nFound {len(context)} relevant sections:")
    for i, doc in enumerate(context, 1):
        print(f"\nSection {i} (Page {doc.metadata.get('page', 'N/A')}):")
        print(f"{doc.page_content[:200]}...")
    
    # Update the state with new context while preserving other state
    return {
        "messages": state["messages"],
        "context": context,
        "current_language": state.get("current_language", "en")
    }

def detect_language(state: AgentState) -> AgentState:
    """Detect the language of the question"""
    last_message = state["messages"][-1]["content"]
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(
        f"Detect the language of this text and respond with only the language name: {last_message}"
    )
    language = response.content.strip()
    
    # Update the state with detected language while preserving context
    return {
        "messages": state["messages"],
        "context": state["context"],
        "current_language": language
    }

def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on the context and language"""
    context = state["context"]
    language = state["current_language"]
    messages = state["messages"]
    
    if not context:
        response = "I couldn't find any relevant information in the document to answer your question."
        messages.append({"role": "assistant", "content": response})
        return {
            "messages": messages,
            "context": context,
            "current_language": language
        }
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about documents.
        Always provide detailed answers with specific references to the source material.
        Include page numbers and relevant quotes when possible.
        Answer in the same language as the question.
        If you cannot find the answer in the provided context, say so clearly.
        Format your response as follows:
        
        Answer: [Your detailed answer]
        
        Sources:
        [List of sources with page numbers and relevant quotes]"""),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Context from the document:\n{context}")
    ])
    
    # Create the chain
    chain = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    
    # Format context for better readability
    formatted_context = "\n\n".join([
        f"Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}"
        for doc in context
    ])
    
    # Generate the response
    response = chain.invoke({
        "messages": messages,
        "context": formatted_context
    })
    
    # Add the response to the messages
    messages.append({"role": "assistant", "content": response})
    
    # Return complete state
    return {
        "messages": messages,
        "context": context,
        "current_language": language
    }

def create_agent_graph(vectorstore: FAISS) -> Graph:
    """Create the agent graph"""
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("retrieve_context", lambda x: retrieve_context(x, vectorstore))
    workflow.add_node("detect_language", detect_language)
    workflow.add_node("generate_response", generate_response)
    
    # Add the edges
    workflow.add_edge("retrieve_context", "detect_language")
    workflow.add_edge("detect_language", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set the entry point
    workflow.set_entry_point("retrieve_context")
    
    # Compile the graph
    graph = workflow.compile()
    
    # Create visualizations directory if it doesn't exist
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save the graph visualization
    graph_image = graph.get_graph().draw_mermaid_png()
    graph_path = os.path.join(viz_dir, "agent_graph.png")
    with open(graph_path, "wb") as f:
        f.write(graph_image)
    print(f"Graph visualization saved to: {graph_path}")
    
    return graph

def main():
    """Main function to run the agent"""
    # Process the document
    folder_path = '/Users/willemvandemierop/Downloads/'
    pdf_path = os.path.join(folder_path, 'TVN_265afvalwater.pdf')
    
    print("Loading and processing the document...")
    vectorstore = process_document(pdf_path)
    
    if not vectorstore:
        print("Failed to process document")
        return
    
    # Create the agent graph
    print("Creating agent graph...")
    agent = create_agent_graph(vectorstore)
    
    print("\nDocument Question Answering System")
    print("Type 'exit' to end")
    print("-" * 50)
    
    # Initialize the state
    state = {
        "messages": [],
        "context": [],
        "current_language": "en"
    }
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
        
        # Add the question to the messages
        state["messages"].append({"role": "user", "content": question})
        
        try:
            # Run the agent
            result = agent.invoke(state)
            
            # Print the response
            print("\n" + result["messages"][-1]["content"])
            
            # Update the state
            state = result
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 