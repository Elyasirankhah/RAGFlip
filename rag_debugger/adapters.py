"""Normalize traces from LangChain, LlamaIndex, OpenAI, and OpenInference."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _text_of(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("text") or doc.get("page_content") or doc.get("content") or "")
    return str(
        getattr(doc, "page_content", None)
        or getattr(getattr(doc, "node", None), "text", None)
        or getattr(doc, "text", None)
        or ""
    )


def _id_of(doc: Any, index: int) -> str:
    if isinstance(doc, dict):
        return str(doc.get("id") or doc.get("chunk_id") or doc.get("doc_id") or f"chunk_{index}")
    meta = getattr(doc, "metadata", None) or {}
    if isinstance(meta, dict) and (meta.get("id") or meta.get("source")):
        return str(meta.get("id") or meta.get("source"))
    node = getattr(doc, "node", None)
    if node is not None and getattr(node, "id_", None):
        return str(node.id_)
    return f"chunk_{index}"


def chunks_from_docs(docs: Optional[List[Any]]) -> List[Dict[str, str]]:
    out = []
    for i, doc in enumerate(docs or []):
        out.append({"id": _id_of(doc, i), "text": _text_of(doc)})
    return out


def from_langchain(query: str, result: Any, corpus: Optional[List[Any]] = None) -> Dict[str, Any]:
    """One-line adapter for RetrievalQA-style LangChain results."""
    if isinstance(result, dict):
        answer = str(result.get("result") or result.get("answer") or result.get("output") or "")
        docs = result.get("source_documents") or result.get("context") or []
    else:
        answer = str(getattr(result, "content", None) or getattr(result, "answer", None) or result)
        docs = getattr(result, "source_documents", None) or []
    return {
        "question": query,
        "answer": answer,
        "retrieved_chunks": chunks_from_docs(docs),
        "corpus_chunks": chunks_from_docs(corpus) if corpus is not None else None,
    }


def from_llama_index(query: str, response: Any, corpus: Optional[List[Any]] = None) -> Dict[str, Any]:
    """One-line adapter for LlamaIndex query engines."""
    answer = str(getattr(response, "response", None) or getattr(response, "text", None) or response)
    nodes = getattr(response, "source_nodes", None) or []
    return {
        "question": query,
        "answer": answer,
        "retrieved_chunks": chunks_from_docs(nodes),
        "corpus_chunks": chunks_from_docs(corpus) if corpus is not None else None,
    }


def from_openai(query: str, completion: Any, retrieved_chunks: List[Any], corpus: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Adapter for OpenAI SDK chat completions plus the docs you retrieved."""
    answer = ""
    if isinstance(completion, dict):
        choices = completion.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            answer = str(message.get("content") or "")
    else:
        try:
            answer = str(completion.choices[0].message.content or "")
        except Exception:
            answer = str(completion)
    return {
        "question": query,
        "answer": answer,
        "retrieved_chunks": chunks_from_docs(retrieved_chunks),
        "corpus_chunks": chunks_from_docs(corpus) if corpus is not None else None,
    }


def from_openinference(span: Dict[str, Any], corpus: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Adapter for OpenInference / OpenTelemetry retrieval+llm spans."""
    attrs = span.get("attributes") or span
    question = str(attrs.get("input.value") or attrs.get("llm.input_messages") or "")
    answer = str(attrs.get("output.value") or attrs.get("llm.output_messages") or "")
    docs = (
        attrs.get("retrieval.documents")
        or attrs.get("reranker.output_documents")
        or span.get("retrieved_chunks")
        or []
    )
    return {
        "question": question if isinstance(question, str) else str(question),
        "answer": answer if isinstance(answer, str) else str(answer),
        "retrieved_chunks": chunks_from_docs(docs),
        "corpus_chunks": chunks_from_docs(corpus) if corpus is not None else None,
    }


def ingest(obj: Any, query: str = "", corpus: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Best-effort ingest from a dict, LangChain result, LlamaIndex response, or span."""
    if isinstance(obj, dict) and obj.get("answer") and "retrieved_chunks" in obj:
        return obj
    if isinstance(obj, dict) and ("attributes" in obj or "retrieval.documents" in (obj.get("attributes") or {})):
        return from_openinference(obj, corpus=corpus)
    if getattr(obj, "source_nodes", None) is not None:
        return from_llama_index(query, obj, corpus=corpus)
    return from_langchain(query, obj, corpus=corpus)
