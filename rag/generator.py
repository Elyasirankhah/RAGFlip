"""
Answer generation using OpenAI Chat Completion API.
"""
from typing import List, Dict
import openai
import os


class AnswerGenerator:
    """Generate answers using OpenAI Chat Completion API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the answer generator.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-3.5-turbo"
    
    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate an answer based on question and retrieved chunks.
        
        Args:
            question: User's question
            retrieved_chunks: List of retrieved chunk dictionaries with 'text' key
            
        Returns:
            Generated answer text
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Chunk {i}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context documents. 
If the context does not contain enough information to answer the question, say so clearly.
Do not add information that is not present in the context."""
        
        user_prompt = f"""Context documents:
{context}

Question: {question}

Answer the question using ONLY the information from the context documents above. If the context doesn't contain enough information, state that clearly."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
