"""LangChain adapter. Produces the stable Trace schema and a retriever hook."""
from typing import Any, Callable, Optional

from rag.trace import Trace
from rag_debugger.adapters import chunks_from_docs, from_langchain


def trace(
    query: str,
    result: Any,
    corpus: Optional[list] = None,
    retriever: Optional[str] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
) -> Trace:
    data = from_langchain(query, result, corpus=corpus)
    data["metadata"] = {
        "retriever": retriever,
        "top_k": top_k,
        "model": model,
    }
    return Trace.from_dict(data)


def as_retriever(langchain_retriever: Any) -> Callable[[str, int], list]:
    """Wrap a LangChain retriever as retrieve(query, k) -> [{id, text}, ...].

    Sets k on search_kwargs when that dict exists, then calls invoke.
    """

    def retrieve(query: str, k: int) -> list:
        search_kwargs = getattr(langchain_retriever, "search_kwargs", None)
        if isinstance(search_kwargs, dict):
            langchain_retriever.search_kwargs = {**search_kwargs, "k": int(k)}
        if hasattr(langchain_retriever, "k"):
            langchain_retriever.k = int(k)
        if hasattr(langchain_retriever, "invoke"):
            docs = langchain_retriever.invoke(query)
        elif hasattr(langchain_retriever, "get_relevant_documents"):
            docs = langchain_retriever.get_relevant_documents(query)
        else:
            docs = langchain_retriever(query)
        return chunks_from_docs(list(docs or []))

    return retrieve
