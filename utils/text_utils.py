"""Text utility functions for sentence splitting and text processing."""
import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while handling missing spaces after punctuation."""
    text = text.strip()
    if not text:
        return []

    # Some dataset passages glue sentences together as "well.Many". Insert a
    # boundary before the next likely sentence start so one paragraph does not
    # become a single claim.
    text = re.sub(r'([.!?])([A-Z0-9"\'])', r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [part.strip() for part in parts if part.strip()]


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
