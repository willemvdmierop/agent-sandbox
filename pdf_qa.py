from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from src.document_processing.processor import DocumentProcessor
import os

# Load environment variables
load_dotenv(dotenv_path='/Users/willemvandemierop/Desktop/ai_agent_project/.env')

def load_and_process_document(file_path: str, use_ocr: bool = False, xpath_query: str = None):
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Process the document
    print(f"Loading document from: {file_path}")
    documents = processor.process_document(file_path, use_ocr=use_ocr, xpath_query=xpath_query)
    
    print(f"\nCreated {len(documents)} chunks")
    print("First chunk preview:")
    if documents:
        print(documents[0].page_content[:1000])
        print("\nChunk metadata:")
        print(documents[0].metadata)
    
    # Create embeddings and vector store
    print("\nCreating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

def create_qa_chain(vectorstore):
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")  # Changed to gpt-4 for better performance
    
    # Create memory for conversation history with explicit output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create the QA chain with more specific prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        ),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        verbose=True  # Add verbose output to see what's happening
    )
    
    return qa_chain

def main():
    # Path to your document
    file_path = "/Users/willemvandemierop/Desktop/sandbox_owen/basisakte-montebello.pdf"
    
    # For XML files, you can specify an XPath query to filter specific elements
    # xpath_query = ".//your-element-name"  # Uncomment and modify for XML files
    
    print("Loading and processing the document...")
    vectorstore = load_and_process_document(file_path, use_ocr=True)  # Set use_ocr=True for scanned PDFs
    
    print("Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore)
    
    print("\nDocument Question Answering System")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        try:
            # Get the answer
            result = qa_chain.invoke({"question": question})
            
            # Print all available keys and their content
            print("\n=== Available Output Keys ===")
            for key in result.keys():
                print(f"\nKey: {key}")
                if key == "answer":
                    print(f"Content: {result[key]}")
                elif key == "source_documents":
                    print("Content: [Source Documents]")
                    for i, doc in enumerate(result[key][:2], 1):
                        print(f"\nSource {i}:")
                        print(f"File: {doc.metadata.get('source', 'N/A')}")
                        print(f"Type: {doc.metadata.get('file_type', 'N/A')}")
                        print(f"Chunk: {doc.metadata.get('chunk_index', 'N/A')}")
                        print(f"Content: {doc.page_content[:200]}...")
                else:
                    print(f"Content: {result[key]}")
            print("\n" + "="*30)
                    
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Full error details:", e.__dict__)

if __name__ == "__main__":
    main() 