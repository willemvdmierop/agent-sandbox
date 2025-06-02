import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from large_pdf_processor import perform_mistral_ocr

def get_vectorstore_path(pdf_path: str) -> str:
    """Get the path for the vector store file"""
    vectorstore_dir = os.path.join(os.path.dirname(pdf_path), "vectorstore")
    os.makedirs(vectorstore_dir, exist_ok=True)
    pdf_name = os.path.basename(pdf_path)
    vectorstore_name = f"{os.path.splitext(pdf_name)[0]}_vectorstore"
    return os.path.join(vectorstore_dir, vectorstore_name)

def create_document_chunks(page_contents: List[str], file_path: str) -> Optional[List[Document]]:
    """Create and split document chunks from OCR results"""
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
    
    vectorstore_path = get_vectorstore_path(file_path)
    if os.path.exists(vectorstore_path):
        print("Found existing vector store, loading...")
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                vectorstore_path, 
                embeddings,
                allow_dangerous_deserialization=True  # Safe since we created the vector store
            )
            print("Successfully loaded existing vector store")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            print("Will create a new vector store...")
    
    page_contents = perform_mistral_ocr(file_path)
    if not page_contents:
        print("No content extracted from the document")
        return None
    
    chunks = create_document_chunks(page_contents, file_path)
    if not chunks:
        return None
    
    print("\nCreating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving vector store to {vectorstore_path}")
    vectorstore.save_local(vectorstore_path)
    print("Vector store saved successfully")
    
    return vectorstore 