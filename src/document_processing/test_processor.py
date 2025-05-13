from processor import DocumentProcessor
import os

def main():
    # Create document processor
    processor = DocumentProcessor()
    
    # Test with a sample PDF
    pdf_path = "/Users/willemvandemierop/Desktop/sandbox_owen/Syndicusovereenkomst_michiel.pdf"  # Replace with your PDF path
    
    if not os.path.exists(pdf_path):
        print(f"Please place a PDF file named 'sample.pdf' in the current directory")
        return
        
    try:
        # Process PDF without OCR
        print("Processing PDF without OCR...")
        chunks = processor.process_pdf(pdf_path, use_ocr=False)
        print(f"Extracted {len(chunks)} text chunks")
        print("\nFirst chunk preview:")
        print(chunks[0][:200] + "...")
        
        # Process PDF with OCR
        print("\nProcessing PDF with OCR...")
        chunks_ocr = processor.process_pdf(pdf_path, use_ocr=True)
        print(f"Extracted {len(chunks_ocr)} text chunks with OCR")
        print("\nFirst chunk preview:")
        print(chunks_ocr[0][:200] + "...")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    main() 