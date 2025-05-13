import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='/Users/willemvandemierop/Desktop/ai_agent_project/.env')

# Let's check if we loaded the right key
# print('key; ',os.getenv("OPENAI_API_KEY"))

# Initialize the ChatOpenAI instance
# llm = ChatOpenAI(model = "gpt-4o-mini")

# # Test the setup
# response = llm.invoke("Hello! Are you working?")
# print(response.content)



# Creating our Agent's Memory
# Define a TypedDict named 'State' to represent a structured dictionary
class State(TypedDict):

    text: str # Stores the original input text
    classification: str # Represents the classification result (e.g., category)
    entities: List[str] # Holds a list of extracted entities (e.g., named entities)
    summary: str # Stores a summarized version of the text


llm = ChatOpenAI(model='gpt-4o-mini', temperature = 0) # Temperature=0: Focused, deterministic responses

# Adding Agent's Capabilities
def classification_node(state: State):
    """
   Classify the text into one of predefined categories.
   
   Parameters:
       state (State): The current state dictionary containing the text to classify
       
   Returns:
       dict: A dictionary with the "classification" key containing the category result
       
   Categories:
       - News: Factual reporting of current events
       - Blog: Personal or informal web writing
       - Research: Academic or scientific content
       - Other: Content that doesn't fit the above categories
   """
    # Define a prompt template that asks the model to classify the given text
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText: {text}\n\nCategory:"
   )
    
    # Format the prompt with the input text from the state
    message = HumanMessage(content=prompt.format(text=state["text"]))

    # Invoke the language model to classify the text based on the prompt
    classification = llm.invoke([message]).content.strip()
    
    # Return the classification result in a dictionary
    return {"classification": classification}

def entity_extraction_node(state: State):
    # Function to identify and extract named entities from text
  # Organized by category (Person, Organization, Location)
  
  # Create template for entity extraction prompt
  # Specifies what entities to look for and format (comma-separated)
  prompt = PromptTemplate(
      input_variables=["text"],
      template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText: {text}\n\nEntities:"
  )
  
  # Format the prompt with text from state and wrap in HumanMessage
  message = HumanMessage(content=prompt.format(text=state["text"]))
  
  # Send to language model, get response, clean whitespace, split into list
  entities = llm.invoke([message]).content.strip().split(", ")
  
  # Return dictionary with entities list to be merged into agent state
  return {"entities": entities}

def summarize_node(state: State):
    # Create a template for the summarization prompt
    # This tells the model to summarize the input text in one sentence
    summarization_prompt = PromptTemplate.from_template(
        """Summarize the following text in one short sentence.
        
        Text: {text}
        
        Summary:"""
    )
    
    # Create a chain by connecting the prompt template to the language model
    # The "|" operator pipes the output of the prompt into the model
    chain = summarization_prompt | llm
    
    # Execute the chain with the input text from the state dictionary
    # This passes the text to be summarized to the model
    response = chain.invoke({"text": state["text"]})
    
    # Return a dictionary with the summary extracted from the model's response
    # This will be merged into the agent's state
    return {"summary": response.content}

workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Define a sample text about Anthropic's MCP to test our agent
    sample_text = """
    Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
    """

    # Create the initial state with our sample text
    state_input = {"text": sample_text}

    # Run the agent's full workflow on our sample text
    result = app.invoke(state_input)

    # Print each component of the result:
    # - The classification category (News, Blog, Research, or Other)
    print("Classification:", result["classification"])

    # - The extracted entities (People, Organizations, Locations)
    print("\nEntities:", result["entities"])

    # - The generated summary of the text
    print("\nSummary:", result["summary"])