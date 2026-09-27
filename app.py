"""
FastAPI application for RAG Debugger.
"""
import sys
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import os

from pathlib import Path

from rag.loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import EmbeddingGenerator
from rag.retriever import VectorRetriever
from rag.generator import AnswerGenerator
from rag.debugger import RAGDebugger
from rag.analyzer import TraceAnalyzer
from rag.judge import SupportJudge
from rag.diagnose import diagnose
from rag.trace import Trace

_UI_DIR = Path(__file__).resolve().parent / "ui"

app = FastAPI(title="RAG Debugger", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

documents = {}
retriever = None
embedding_generator = None
answer_generator = None
debugger = None
analyzer = None


class QuestionRequest(BaseModel):
    question: str


class TraceChunk(BaseModel):
    text: str
    id: Optional[str] = None
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None

    class Config:
        extra = "ignore"


class AnalyzeRequest(BaseModel):
    question: str = ""
    answer: str = Field(..., min_length=1)
    retrieved_chunks: List[TraceChunk] = Field(default_factory=list)
    corpus_chunks: Optional[List[TraceChunk]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global embedding_generator, retriever, answer_generator, debugger, analyzer
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    embedding_generator = EmbeddingGenerator(api_key=api_key)
    retriever = VectorRetriever(dimension=1536)
    answer_generator = AnswerGenerator(api_key=api_key)
    debugger = RAGDebugger(embedding_generator, similarity_threshold=0.7)
    analyzer = TraceAnalyzer(embedding_generator, SupportJudge(api_key=api_key))


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents (PDF or TXT).
    
    Returns:
        List of document IDs
    """
    global retriever, embedding_generator
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    document_ids = []
    
    for file in files:
        # Read file content
        content = await file.read()
        
        # Load document
        try:
            text = load_document(content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading {file.filename}: {str(e)}")
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Chunk the document
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        
        # Generate chunk IDs and prepare chunk metadata
        chunk_metadata = []
        chunk_texts = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "start_idx": chunk["start_idx"]
            })
            chunk_texts.append(chunk["text"])
        
        # Generate embeddings for chunks
        chunk_embeddings = embedding_generator.embed_batch(chunk_texts)
        
        # Add to retriever
        retriever.add_chunks(chunk_metadata, chunk_embeddings)
        
        # Store document metadata
        documents[doc_id] = {
            "filename": file.filename,
            "text": text,
            "chunks": chunk_metadata
        }
        
        document_ids.append(doc_id)
    
    return {"document_ids": document_ids}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get answer with evidence alignment.
    
    Args:
        request: QuestionRequest with question field
        
    Returns:
        JSON with answer, retrieved chunks, and evidence alignment
    """
    global retriever, embedding_generator, answer_generator, debugger
    
    question = request.question
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if retriever.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload documents first.")
    
    query_embedding = embedding_generator.embed_text(question)
    top_k = 5
    retrieved_chunks = retriever.search(query_embedding, top_k=top_k)
    
    if not retrieved_chunks:
        return JSONResponse(content={
            "answer": "No relevant information found in the uploaded documents.",
            "retrieved_chunks": [],
            "evidence_alignment": [],
            "unsupported_sentences": []
        })
    
    answer = answer_generator.generate_answer(question, retrieved_chunks)
    evidence_analysis = debugger.analyze_evidence(answer, retrieved_chunks)
    
    response = {
        "answer": answer,
        "retrieved_chunks": [
            {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": chunk["score"]
            }
            for chunk in retrieved_chunks
        ],
        "evidence_alignment": evidence_analysis["evidence_alignment"],
        "unsupported_sentences": evidence_analysis["unsupported_sentences"]
    }
    
    return JSONResponse(content=response)


@app.post("/analyze")
async def analyze_trace(request: AnalyzeRequest):
    """
    Debug an existing RAG run. Pass the question, model answer, retrieved
    chunks, and optionally extra corpus chunks. Returns sentence-level faults:
    supported, retrieval_miss, hallucination, chunking_miss, or unsupported.
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer is not initialized")

    retrieved = [
        {"text": chunk.text, "id": chunk.id, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id}
        for chunk in request.retrieved_chunks
    ]
    corpus = None
    if request.corpus_chunks is not None:
        corpus = [
            {"text": chunk.text, "id": chunk.id, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id}
            for chunk in request.corpus_chunks
        ]
    trace = Trace.from_dict(
        {
            "question": request.question,
            "answer": request.answer,
            "retrieved_chunks": retrieved,
            "corpus_chunks": corpus or [],
        }
    )
    return diagnose(trace, analyzer=analyzer).to_dict()


@app.post("/diagnose")
async def diagnose_trace(request: AnalyzeRequest):
    """Alias for /analyze: root-cause a bad RAG answer."""
    return await analyze_trace(request)


@app.post("/repair")
async def repair_trace(request: AnalyzeRequest):
    """Replay the trace with larger k and hybrid retrieval; report before vs after."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer is not initialized")
    from rag.repair import run_repair_experiments

    retrieved = [
        {"text": chunk.text, "id": chunk.id, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id}
        for chunk in request.retrieved_chunks
    ]
    corpus = None
    if request.corpus_chunks is not None:
        corpus = [
            {"text": chunk.text, "id": chunk.id, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id}
            for chunk in request.corpus_chunks
        ]
    return run_repair_experiments(
        analyzer,
        request.question,
        request.answer,
        retrieved,
        corpus,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "documents_loaded": len(documents)}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Debugger",
        "tagline": "Tells you why your RAG failed and tests how to fix it.",
        "endpoints": {
            "POST /diagnose": "Root-cause a RAG trace (component, rank, suggested fixes)",
            "POST /analyze": "Same as /diagnose",
            "POST /repair": "Replay larger k / hybrid retrieval and report before vs after",
            "POST /upload": "Demo only: upload documents",
            "POST /ask": "Demo only: ask against uploaded documents",
            "GET /health": "Health check",
        },
        "ui_url": "/ui"
    }
