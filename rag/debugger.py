"""
RAG Debugger: Analyzes which chunks support each sentence in the answer.
"""
from typing import List, Dict
from utils.text_utils import split_into_sentences
from rag.embeddings import EmbeddingGenerator


class RAGDebugger:
    """Debugger that aligns answer sentences with supporting chunks."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, similarity_threshold: float = 0.7):
        """
        Initialize the RAG debugger.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            similarity_threshold: Minimum cosine similarity to consider a chunk as supporting a sentence
        """
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
    
    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def analyze_evidence(
        self,
        answer: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Analyze which chunks support each sentence in the answer.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: List of retrieved chunks with 'text', 'chunk_id', 'doc_id'
            
        Returns:
            Dictionary with:
            - evidence_alignment: List of {answer_sentence, supporting_chunks}
            - unsupported_sentences: List of sentences with no supporting chunks
        """
        answer_sentences = split_into_sentences(answer)
        sentence_embeddings = self.embedding_generator.embed_batch(answer_sentences)
        
        chunk_texts = [chunk['text'] for chunk in retrieved_chunks]
        chunk_embeddings = self.embedding_generator.embed_batch(chunk_texts)
        
        evidence_alignment = []
        unsupported_sentences = []
        
        for sentence, sentence_emb in zip(answer_sentences, sentence_embeddings):
            supporting_chunk_ids = []
            
            for chunk, chunk_emb in zip(retrieved_chunks, chunk_embeddings):
                similarity = self.compute_cosine_similarity(sentence_emb, chunk_emb)
                
                if similarity >= self.similarity_threshold:
                    supporting_chunk_ids.append(chunk['chunk_id'])
            
            if supporting_chunk_ids:
                evidence_alignment.append({
                    "answer_sentence": sentence,
                    "supporting_chunks": supporting_chunk_ids
                })
            else:
                unsupported_sentences.append(sentence)
                evidence_alignment.append({
                    "answer_sentence": sentence,
                    "supporting_chunks": []
                })
        
        return {
            "evidence_alignment": evidence_alignment,
            "unsupported_sentences": unsupported_sentences
        }
