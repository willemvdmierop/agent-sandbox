from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
import json
from datetime import datetime
import hashlib
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentMetadata(dict):
    """Metadata for processed documents."""
    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.update({
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'last_modified': os.path.getmtime(file_path),
            'processing_date': datetime.now().isoformat(),
            'hash': self._calculate_file_hash(file_path)
        })
    
    @staticmethod
    def _calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class DocumentProcessor:
    """Process documents into vector stores."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.docx': UnstructuredWordDocumentLoader
    }
    
    def __init__(self, embeddings: Optional[OpenAIEmbeddings] = None):
        """Initialize the document processor.
        
        Args:
            embeddings: OpenAI embeddings instance. If None, creates a new one.
        """
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_folder(self, folder_path: str) -> FAISS:
        """Process all supported documents in a folder into a vector store.
        
        Args:
            folder_path: Path to the folder containing documents
            
        Returns:
            FAISS vector store containing all processed documents
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")
            
        # Find all supported documents
        documents = []
        for ext in self.SUPPORTED_EXTENSIONS:
            documents.extend(folder.glob(f"**/*{ext}"))
            
        if not documents:
            raise ValueError(f"No supported documents found in {folder}")
            
        # Process each document
        all_texts = []
        all_metadatas = []
        
        for doc_path in documents:
            try:
                texts, metadatas = self._process_document(doc_path)
                all_texts.extend(texts)
                all_metadatas.extend(metadatas)
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                continue
                
        if not all_texts:
            raise ValueError("No documents were successfully processed")
            
        # Create vector store
        return FAISS.from_texts(
            all_texts,
            self.embeddings,
            metadatas=all_metadatas
        )
        
    def _process_document(self, doc_path: Path) -> Tuple[List[str], List[Dict]]:
        """Process a single document.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Tuple of (list of text chunks, list of metadata dicts)
        """
        # Get the appropriate loader
        extension = doc_path.suffix.lower()
        loader_cls = self.SUPPORTED_EXTENSIONS.get(extension)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {extension}")
            
        # Load and split the document
        loader = loader_cls(str(doc_path))
        documents = loader.load()
        
        # Create base metadata
        base_metadata = DocumentMetadata(str(doc_path))
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Extract text and enhance metadata
        texts = []
        metadatas = []
        
        for i, split in enumerate(splits):
            texts.append(split.page_content)
            
            # Combine base metadata with split-specific metadata
            metadata = base_metadata.copy()
            metadata.update(split.metadata)
            metadata['chunk_index'] = i
            metadatas.append(metadata)
            
        return texts, metadatas
        
    def save_vectorstore(self, vectorstore: FAISS, save_path: str):
        """Save a vector store to disk.
        
        Args:
            vectorstore: The FAISS vector store to save
            save_path: Path where to save the vector store
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir)) 