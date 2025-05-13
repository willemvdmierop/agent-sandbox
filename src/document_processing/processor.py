import os
from typing import List, Optional, Union, Protocol
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
import xml.etree.ElementTree as ET
from langchain.schema import Document
from abc import ABC, abstractmethod

class DocumentLoader(ABC):
    """Base class for document loaders."""
    
    def __init__(self, output_dir: str = "processed_documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    @abstractmethod
    def load(self, file_path: str, **kwargs) -> List[str]:
        """Load and process a document.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional arguments specific to the loader
            
        Returns:
            List of text chunks
        """
        pass
    
    def save_processed_text(self, text: str, file_path: Path) -> None:
        """Save processed text to a file.
        
        Args:
            text: The processed text to save
            file_path: Path to save the text to
        """
        output_file = self.output_dir / f"{file_path.stem}_processed.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

class PDFLoader(DocumentLoader):
    """Loader for PDF documents."""
    
    def __init__(self, output_dir: str = "processed_documents"):
        super().__init__(output_dir)
        # Configure pytesseract path for macOS
        if os.path.exists("/opt/homebrew/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    
    def load(self, file_path: str, use_ocr: bool = False) -> List[str]:
        """Load and process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            use_ocr: Whether to use OCR for text extraction
            
        Returns:
            List of text chunks
        """
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if use_ocr:
            # Convert PDF to images and use OCR
            images = convert_from_path(pdf_path)
            texts = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang='nld')  # Using Dutch language
                texts.append(text)
            full_text = "\n".join(texts)
        else:
            # Use PyMuPDF to extract text
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(full_text)
        
        # Save processed text
        self.save_processed_text(full_text, pdf_path)
            
        return chunks

class XMLLoader(DocumentLoader):
    """Loader for XML documents."""
    
    def load(self, file_path: str, xpath_query: Optional[str] = None) -> List[str]:
        """Load and process an XML file.
        
        Args:
            file_path: Path to the XML file
            xpath_query: Optional XPath query to filter specific elements
            
        Returns:
            List of text chunks
        """
        xml_path = Path(file_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract text based on XPath query if provided
        if xpath_query:
            elements = root.findall(xpath_query)
            texts = [elem.text.strip() for elem in elements if elem.text and elem.text.strip()]
        else:
            # Extract all text from the XML
            texts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())

        full_text = "\n".join(texts)

        # Split text into chunks
        chunks = self.text_splitter.split_text(full_text)

        # Save processed text
        self.save_processed_text(full_text, xml_path)

        return chunks

class DocumentProcessor:
    """Main document processor that handles different file types."""
    
    def __init__(self, output_dir: str = "processed_documents"):
        self.output_dir = output_dir
        self.loaders = {
            '.pdf': PDFLoader(output_dir),
            '.xml': XMLLoader(output_dir)
        }
    
    def process_document(self, file_path: str, use_ocr: bool = False, xpath_query: Optional[str] = None) -> List[Document]:
        """Process a document and return LangChain documents.
        
        Args:
            file_path: Path to the document
            use_ocr: Whether to use OCR for PDF files
            xpath_query: Optional XPath query for XML files
            
        Returns:
            List of LangChain Document objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get the appropriate loader based on file extension
        loader = self.loaders.get(file_path.suffix.lower())
        if not loader:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Process the document with appropriate arguments based on file type
        if file_path.suffix.lower() == '.pdf':
            chunks = loader.load(str(file_path), use_ocr=use_ocr)
        elif file_path.suffix.lower() == '.xml':
            chunks = loader.load(str(file_path), xpath_query=xpath_query)
        else:
            chunks = loader.load(str(file_path))

        # Convert chunks to LangChain documents
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path),
                    "file_type": file_path.suffix[1:],
                    "chunk_index": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        return documents 