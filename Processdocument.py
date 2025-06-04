import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from large_pdf_processor import perform_mistral_ocr

def get_vectorstore_path(pdf_path: str) -> str:
    """Get the path for the vector store file"""
    # Create a vectorstore directory if it doesn't exist
    vectorstore_dir = os.path.join(os.path.dirname(pdf_path), "vectorstore")
    os.makedirs(vectorstore_dir, exist_ok=True)
    
    # Create a unique filename based on the PDF name
    pdf_name = os.path.basename(pdf_path)
    vectorstore_name = f"{os.path.splitext(pdf_name)[0]}_vectorstore"
    return os.path.join(vectorstore_dir, vectorstore_name)

def create_document_chunks(page_contents: List[str], file_path: str) -> Optional[List[Document]]:
    """Create and split document chunks from OCR results"""
    # Create documents from OCR results
    documents = []
    for i, content in enumerate(page_contents):
        if content and content.strip():
            documents.append(Document(
                page_content=content,
                metadata={
                    'source': file_path,
                    'page': i + 1,
                    'file_type': 'pdf',
                    'chunk_index': i
                }
            ))
    
    if not documents:
        print("No valid documents were created")
        return None
        
    print(f"\nCreated {len(documents)} chunks")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        print("No chunks were created")
        return None
        
    return chunks

def process_document(file_path: str) -> Optional[FAISS]:
    """Process the document and create a vector store"""
    print(f"Processing document: {file_path}")
    
    # Check if vector store already exists
    vectorstore_path = get_vectorstore_path(file_path)
    if os.path.exists(vectorstore_path):
        print("Found existing vector store, loading...")
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(vectorstore_path, embeddings)
            print("Successfully loaded existing vector store")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            print("Will create a new vector store...")
    
    # Use Mistral OCR for PDF
    page_contents = perform_mistral_ocr(file_path)
    if not page_contents:
        print("No content extracted from the document")
        return None
    
    # Create and split document chunks
    chunks = create_document_chunks(page_contents, file_path)
    if not chunks:
        return None
    
    # Create embeddings and vector store
    print("\nCreating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store
    print(f"Saving vector store to {vectorstore_path}")
    vectorstore.save_local(vectorstore_path)
    print("Vector store saved successfully")
    
    return vectorstore

def create_qa_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """Create a QA chain for the document"""
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create the QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 3 most relevant chunks
        ),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        verbose=True
    )
    
    return qa_chain

def main():
    """Main function to run the QA system"""
    # Process the BAV PDF
    folder_path = '/Users/willemvandemierop/Downloads/'
    bav_pdf = os.path.join(folder_path, 'INVOICE-VFZ_C29_25_00220.pdf')
    
    print("Loading and processing the document...")
    vectorstore = process_document(bav_pdf)
    
    if not vectorstore:
        print("Failed to process document")
        return
        
    print("Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore)
    
    print("\nDocument Question Answering System")
    print("Type 'exit' to end")
    print("-" * 50)
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
            
        try:
            # Get the answer
            result = qa_chain.invoke({"question": question})
            
            # Print the answer
            print("\nAnswer:", result["answer"])
            
            # Print source information
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"][:2], 1):
                print(f"\nSource {i}:")
                print(f"Page: {doc.metadata.get('page', 'N/A')}")
                print(f"Content: {doc.page_content[:200]}...")
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

