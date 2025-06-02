import time
import base64
import requests
from typing import List
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import os

def encode_pdf(pdf_path: str) -> str:
    """Encode the PDF to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def perform_mistral_ocr(pdf_path: str) -> List[str]:
    """Use Mistral AI's OCR API to process the PDF"""
    headers = {
        "Authorization": "Bearer 5hT66s9SYBFG96uTNzKTdecpBdN4Bj9D",
        "Content-Type": "application/json"
    }
    
    # Encode the PDF
    base64_pdf = encode_pdf(pdf_path)
    if not base64_pdf:
        return []
    
    data = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}"
        },
        "include_image_base64": True
    }
    
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            # Return list of page contents
            return [page.get("markdown", "") for page in result.get("pages", [])]
        else:
            print(f"Error with Mistral OCR API: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error processing OCR: {str(e)}")
        return []

def process_large_pdf(pdf_path: str, headers: dict) -> List[str]:
    """Process a large PDF by splitting it into smaller chunks"""
    try:
        reader = PdfReader(pdf_path)
        chunks = []
        current_chunk = PdfWriter()
        current_size = 0
        max_chunk_size = 4 * 1024 * 1024  # 4MB
        
        print(f"Splitting PDF into chunks...")
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_size = len(page.extract_text().encode('utf-8'))
            
            if current_size + page_size > max_chunk_size and current_chunk.get_num_pages() > 0:
                output = BytesIO()
                current_chunk.write(output)
                chunks.append(output.getvalue())
                current_chunk = PdfWriter()
                current_size = 0
            
            current_chunk.add_page(page)
            current_size += page_size
        
        if current_chunk.get_num_pages() > 0:
            output = BytesIO()
            current_chunk.write(output)
            chunks.append(output.getvalue())
        
        print(f"Created {len(chunks)} chunks")
        all_pages = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            base64_pdf = base64.b64encode(chunk).decode('utf-8')
            
            data = {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                "include_image_base64": True
            }
            
            try:
                if i > 0:
                    print("Waiting 60 seconds before processing next chunk...")
                    time.sleep(60)
                
                response = requests.post(
                    "https://api.mistral.ai/v1/ocr",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "pages" in result:
                        chunk_pages = [page.get("markdown", "") for page in result["pages"]]
                        all_pages.extend(chunk_pages)
                        print(f"Successfully processed chunk {i+1}")
                    else:
                        print(f"Warning: Unexpected response format for chunk {i+1}")
                        return []
                else:
                    print(f"Warning: API error for chunk {i+1}: {response.status_code}")
                    print(f"Response content: {response.text}")
                    return []
                    
            except requests.Timeout:
                print(f"Timeout error processing chunk {i+1}. Retrying...")
                time.sleep(30)
                try:
                    response = requests.post(
                        "https://api.mistral.ai/v1/ocr",
                        headers=headers,
                        json=data,
                        timeout=180
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if "pages" in result:
                            chunk_pages = [page.get("markdown", "") for page in result["pages"]]
                            all_pages.extend(chunk_pages)
                            print(f"Successfully processed chunk {i+1} on retry")
                        else:
                            print(f"Warning: Unexpected response format for chunk {i+1} on retry")
                            return []
                    else:
                        print(f"Warning: API error for chunk {i+1} on retry: {response.status_code}")
                        return []
                except Exception as e:
                    print(f"Error on retry for chunk {i+1}: {str(e)}")
                    return []
            except Exception as e:
                print(f"Warning: Could not process chunk {i+1}: {str(e)}")
                return []
        
        return all_pages if all_pages else []
        
    except Exception as e:
        print(f"Error splitting PDF: {str(e)}")
        return []

def test_pdf_processing(pdf_path: str):
    """Test function to demonstrate PDF processing functionality"""
    print("\n=== Testing PDF Processing ===")
    print(f"Testing with PDF: {pdf_path}")
    
    # Test 1: Test encode_pdf function
    print("\n1. Testing encode_pdf function...")
    encoded_pdf = encode_pdf(pdf_path)
    if encoded_pdf:
        print("✓ PDF encoding successful")
        print(f"Encoded length: {len(encoded_pdf)} characters")
    else:
        print("✗ PDF encoding failed")
        return
    
    # Test 2: Test perform_mistral_ocr function
    print("\n2. Testing perform_mistral_ocr function...")
    ocr_results = perform_mistral_ocr(pdf_path)
    if ocr_results:
        print("✓ OCR processing successful")
        print(f"Number of pages processed: {len(ocr_results)}")
        print("\nFirst page preview:")
        print("-" * 50)
        print(ocr_results[0][:500] + "..." if len(ocr_results[0]) > 500 else ocr_results[0])
        print("-" * 50)
    else:
        print("✗ OCR processing failed")
        return
    
    print("\n=== All tests completed successfully ===")

def main():
    """Main function to test the PDF processor"""
    # Test with a small PDF (less than 5MB)
    small_pdf = "/Users/willemvandemierop/Desktop/sandbox_owen/Test-AV/2025BAV-verslag.pdf"
    print("\n=== Testing with small PDF ===")
    test_pdf_processing(small_pdf)
    
    # Test with a large PDF (if available)
    # large_pdf = "/Users/willemvandemierop/Desktop/sandbox_owen/Test-AV/2025RvM-verslag.pdf"
    # if os.path.exists(large_pdf):
    #     print("\n=== Testing with large PDF ===")
    #     test_pdf_processing(large_pdf)
    # else:
    #     print("\nLarge PDF not found, skipping large PDF test")

if __name__ == "__main__":
    main() 