"""
Text chunking utilities for splitting documents into smaller pieces.
"""
from typing import List
from utils.text_utils import clean_text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[dict]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with 'text' and 'start_idx' keys
    """
    text = clean_text(text)
    
    if len(text) <= chunk_size:
        return [{"text": text, "start_idx": 0}]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        if end < len(text):
            last_space = chunk_text.rfind(' ')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_space, last_newline)
            
            if break_point > chunk_size * 0.5:
                chunk_text = chunk_text[:break_point]
                end = start + break_point
        
        chunks.append({
            "text": chunk_text.strip(),
            "start_idx": start
        })
        
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks
