"""
Vector retrieval using FAISS for similarity search.
"""
from typing import List, Dict
import numpy as np
import faiss


class VectorRetriever:
    """FAISS-based vector retriever for semantic search."""
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize the vector retriever.
        
        Args:
            dimension: Dimension of embedding vectors (1536 for OpenAI ada-002)
        """
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self._reset()
    
    def _reset(self):
        """Reset the index and chunk storage."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Add chunks and their embeddings to the index.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'doc_id', 'chunk_id', 'start_idx'
            embeddings: List of embedding vectors for each chunk
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store chunk metadata
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries with 'doc_id', 'chunk_id', 'text', 'start_idx', 'score'
        """
        if self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(-distances[0][i])
            results.append(chunk)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Dict:
        """
        Retrieve a chunk by its ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk dictionary or None if not found
        """
        for chunk in self.chunks:
            if chunk.get('chunk_id') == chunk_id:
                return chunk
        return None
