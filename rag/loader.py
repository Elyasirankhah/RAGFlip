"""
Document loader for PDF and TXT files.
"""
from typing import List
import PyPDF2
from io import BytesIO


def load_pdf(file_content: bytes) -> str:
    """
    Load text content from a PDF file.
    
    Args:
        file_content: PDF file content as bytes
        
    Returns:
        Extracted text from PDF
    """
    pdf_file = BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    text_parts = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    
    return "\n\n".join(text_parts)


def load_txt(file_content: bytes) -> str:
    """
    Load text content from a TXT file.
    
    Args:
        file_content: TXT file content as bytes
        
    Returns:
        Text content decoded as UTF-8
    """
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    try:
        return file_content.decode('windows-1252')
    except UnicodeDecodeError:
        pass
    
    try:
        return file_content.decode('latin-1', errors='replace')
    except Exception:
        return file_content.decode('utf-8', errors='replace')


def load_document(file_content: bytes, filename: str) -> str:
    """
    Load document based on file extension.
    
    Args:
        file_content: File content as bytes
        filename: Name of the file (used to determine type)
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is not supported
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return load_pdf(file_content)
    elif filename_lower.endswith('.txt'):
        return load_txt(file_content)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Supported: .pdf, .txt")
