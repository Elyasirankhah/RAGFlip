"""Failures observed from five retrieval pipelines on HotpotQA.

The pipelines search the full paragraph or window index. Nothing is hidden,
and a case is kept from its rank or split, not from a repair result.
RECTIFY is not part of this set: its sandbox reruns RAGVue through an LLM
endpoint instead of this retriever hook.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from rag.analyzer import _query_overlap
from rag.pipelines.sparse import SparseIndex
from rag.trace import Trace

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "mixed"
PARAGRAPH_PATH = ROOT / "paragraphs.json"
WINDOW_PATH = ROOT / "windows.json"
TRACE_PATH = ROOT / "traces.jsonl"
SOURCE = "hotpotqa/hotpot_qa validation"
OPERATING_K = 2
WINDOW_WORDS = 16
SPLIT_K = 6
CANDIDATES = 10

# Failure quotas are filled before controls. Order is fixed so scarce modes
# are taken first. A Hotpot example is used at most twice.
FAILURE_QUOTAS = {
    "chunk_split": 8,
    "query_rewrite": 8,
    "rerank_logged": 8,
    "wider_k_reaches": 8,
    "wider_k_misses": 8,
    "answer_not_in_corpus": 4,
}
CONTROL_QUOTAS = {
    "supported": 6,
    "wider_k_drops_support": 6,
}
REPAIR_OBSERVED = set(FAILURE_QUOTAS)


def load_mixed_cases() -> List[dict]:
    paragraphs = json.loads(PARAGRAPH_PATH.read_text(encoding="utf-8"))
    windows = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
    paragraph_index = SparseIndex(paragraphs)
    cases = []
    for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["pipeline"] == "window_bm25":
            chunks = [chunk for chunk in windows if chunk.get("source_id") == record["source_id"]]
            index = SparseIndex(chunks)
        else:
            chunks = paragraphs
            index = paragraph_index
        cases.append(_materialize(record, chunks, index))
    return cases


def select_records(rows: Sequence[dict]) -> dict:
    """Walk Hotpot rows in order and keep the first case that matches each quota."""
    paragraphs = _paragraphs(rows)
    paragraph_index = SparseIndex(paragraphs)
    by_row = _paragraphs_by_row(paragraphs)
    counts = {name: 0 for name in list(FAILURE_QUOTAS) + list(CONTROL_QUOTAS)}
    used: Dict[str, int] = {}
    records = []
    for observed, quota in list(FAILURE_QUOTAS.items()) + list(CONTROL_QUOTAS.items()):
        for row in rows:
            if counts[observed] >= quota:
                break
            source_id = str(row.get("id") or "")
            if used.get(source_id, 0) >= 2:
                continue
            record = _observe(
                observed,
                row,
                by_row.get(source_id, []),
                paragraph_index,
            )
            if record is None:
                continue
            record["name"] = f"{observed}-{counts[observed] + 1}"
            records.append(record)
            counts[observed] += 1
            used[source_id] = used.get(source_id, 0) + 1
    window_sources = {record["source_id"] for record in records if record["pipeline"] == "window_bm25"}
    windows = _windows([chunk for chunk in paragraphs if chunk["source_id"] in window_sources])
    return {
        "paragraphs": paragraphs,
        "windows": windows,
        "records": records,
        "counts": counts,
    }


def write_dataset(payload: dict, root: Path = ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "paragraphs.json").write_text(json.dumps(payload["paragraphs"]), encoding="utf-8")
    (root / "windows.json").write_text(json.dumps(payload["windows"]), encoding="utf-8")
    lines = [json.dumps(record) for record in payload["records"]]
    (root / "traces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize(record: dict, chunks: Sequence[dict], index: SparseIndex) -> dict:
    by_id = {chunk["id"]: chunk for chunk in chunks}
    ranked = index.ranked(record["question"])
    rank_of = {chunk["id"]: (rank, score) for rank, (chunk, score) in enumerate(ranked, start=1)}
    answer_ids = [chunk["id"] for chunk in chunks if _contains(chunk.get("text") or "", record["answer"])]
    keep_ids = [chunk["id"] for chunk, _score in ranked[:25]]
    for chunk_id in answer_ids:
        if chunk_id not in keep_ids:
            keep_ids.append(chunk_id)
    corpus = []
    for chunk_id in keep_ids:
        chunk = by_id[chunk_id]
        rank, score = rank_of[chunk_id]
        corpus.append({"id": chunk_id, "text": chunk.get("text") or "", "rank": rank, "score": score})
    retrieved = []
    for chunk_id in record["retrieved_ids"]:
        chunk = by_id[chunk_id]
        rank, score = rank_of[chunk_id]
        retrieved.append({"id": chunk_id, "text": chunk.get("text") or "", "rank": rank, "score": score})
    candidates = []
    for item in record.get("candidates") or []:
        chunk = by_id[item["id"]]
        candidates.append({"id": item["id"], "text": chunk.get("text") or "", "score": item["score"]})
    kind = "budget_bm25" if record["pipeline"] == "budget_bm25" else "bm25"
    trace = Trace.from_dict(
        {
            "question": record["question"],
            "answer": record["answer"],
            "retrieved_chunks": retrieved,
            "corpus_chunks": corpus,
            "candidates": candidates,
            "metadata": {
                "retriever": record["pipeline"],
                "top_k": record["top_k"],
                "original_query": record.get("original_query"),
                "rewritten_query": record.get("rewritten_query"),
                "pipeline": record["pipeline"],
                "source": SOURCE,
                "source_id": record["source_id"],
                "user_question": record.get("user_question") or record["question"],
            },
        }
    )
    return {
        "name": record["name"],
        "observed": record["observed"],
        "pipeline": record["pipeline"],
        "trace": trace,
        "retriever": _retriever(index, kind),
    }


def _retriever(index: SparseIndex, kind: str):
    def retrieve(query: str, k: int):
        if kind == "budget_bm25":
            return index.budget(query, k)
        return index.top(query, k)

    return retrieve


def _observe(
    observed: str,
    row: dict,
    own_paragraphs: Sequence[dict],
    paragraph_index: SparseIndex,
) -> Optional[dict]:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    source_id = str(row.get("id") or "")
    if not question or not answer or len(answer) > 80:
        return None
    if answer.lower() in {"yes", "no"} and observed != "answer_not_in_corpus":
        return None
    own_ids = {chunk["id"] for chunk in own_paragraphs}
    answer_paragraphs = [chunk for chunk in own_paragraphs if _contains(chunk.get("text") or "", answer)]
    if observed == "answer_not_in_corpus":
        in_index = any(_contains(chunk.get("text") or "", answer) for chunk in paragraph_index.chunks)
        if in_index or len(answer) < 2:
            return None
        hits = paragraph_index.top(question, OPERATING_K)
        return _record("bm25", observed, question, answer, source_id, OPERATING_K, hits, question)
    if not answer_paragraphs:
        return None
    if observed == "supported":
        hits = paragraph_index.top(question, OPERATING_K)
        if not _ids(hits) & {chunk["id"] for chunk in answer_paragraphs}:
            return None
        return _record("bm25", observed, question, answer, source_id, OPERATING_K, hits, question)
    if observed == "wider_k_reaches":
        return _rank_case(question, answer, source_id, answer_paragraphs, paragraph_index, reaches=True)
    if observed == "wider_k_misses":
        return _rank_case(question, answer, source_id, answer_paragraphs, paragraph_index, reaches=False)
    if observed == "query_rewrite":
        return _rewrite_case(question, answer, source_id, answer_paragraphs, paragraph_index)
    if observed == "rerank_logged":
        return _rerank_case(question, answer, source_id, answer_paragraphs, paragraph_index)
    if observed == "chunk_split":
        return _split_case(question, answer, source_id, own_paragraphs, answer_paragraphs)
    if observed == "wider_k_drops_support":
        return _budget_case(question, answer, source_id, answer_paragraphs, paragraph_index, own_ids)
    return None


def _rank_case(question, answer, source_id, answer_paragraphs, index: SparseIndex, reaches: bool):
    ranked = index.ranked(question)
    rank = _best_rank(ranked, {chunk["id"] for chunk in answer_paragraphs})
    increased = min(20, max(OPERATING_K * 3, OPERATING_K + 5))
    if rank is None or rank <= OPERATING_K:
        return None
    can_reach = rank <= increased
    if can_reach != reaches:
        return None
    hits = [_hit_from_ranked(ranked, position) for position in range(OPERATING_K)]
    return _record("bm25", "wider_k_reaches" if reaches else "wider_k_misses", question, answer, source_id, OPERATING_K, hits, question)


def _rewrite_case(question, answer, source_id, answer_paragraphs, index: SparseIndex):
    rewritten = index.keyword_query(question)
    if not rewritten or rewritten == question.lower():
        return None
    answer_ids = {chunk["id"] for chunk in answer_paragraphs}
    original_hits = index.top(question, OPERATING_K)
    rewritten_hits = index.top(rewritten, OPERATING_K)
    if not (_ids(original_hits) & answer_ids) or (_ids(rewritten_hits) & answer_ids):
        return None
    gold = next(chunk for chunk in answer_paragraphs if chunk["id"] in _ids(original_hits))
    if _query_overlap(question, gold.get("text") or "") <= _query_overlap(rewritten, gold.get("text") or "") + 0.2:
        return None
    return _record(
        "keyword_bm25",
        "query_rewrite",
        rewritten,
        answer,
        source_id,
        OPERATING_K,
        rewritten_hits,
        question,
        original_query=question,
        rewritten_query=rewritten,
    )


def _rerank_case(question, answer, source_id, answer_paragraphs, index: SparseIndex):
    pool = index.top(question, CANDIDATES)
    if len(pool) < OPERATING_K:
        return None
    scores = index.tfidf(question, [hit["id"] for hit in pool])
    best_id = max(pool, key=lambda hit: (scores.get(hit["id"], 0.0), -hit["rank"]))["id"]
    served = pool[:OPERATING_K]
    answer_ids = {chunk["id"] for chunk in answer_paragraphs}
    if best_id in _ids(served) or best_id not in answer_ids:
        return None
    candidates = [{"id": hit["id"], "score": scores.get(hit["id"], 0.0)} for hit in pool]
    return _record(
        "bm25_tfidf_candidates",
        "rerank_logged",
        question,
        answer,
        source_id,
        OPERATING_K,
        served,
        question,
        candidates=candidates,
    )


def _split_case(question, answer, source_id, own_paragraphs, answer_paragraphs):
    """16-word windows over this question's own paragraphs, not the global index."""
    windows = _windows(own_paragraphs)
    if not windows:
        return None
    index = SparseIndex(windows)
    hits = index.top(question, SPLIT_K)
    hit_ids = _ids(hits)
    for paragraph in answer_paragraphs:
        group = [window for window in windows if window.get("parent_id") == paragraph["id"]]
        run = _covering_run(group, answer)
        if len(run) < 2:
            continue
        if all(window["id"] in hit_ids for window in run):
            return _record("window_bm25", "chunk_split", question, answer, source_id, SPLIT_K, hits, question)
    return None


def _budget_case(question, answer, source_id, answer_paragraphs, index: SparseIndex, own_ids: set):
    del own_ids
    top = index.budget(question, 1)
    wider = index.budget(question, min(20, max(1 * 3, 1 + 5)))
    answer_ids = {chunk["id"] for chunk in answer_paragraphs}
    if not (_ids(top) & answer_ids) or (_ids(wider) & answer_ids):
        return None
    return _record("budget_bm25", "wider_k_drops_support", question, answer, source_id, 1, top, question)


def _record(
    pipeline: str,
    observed: str,
    question: str,
    answer: str,
    source_id: str,
    top_k: int,
    hits: Sequence[dict],
    user_question: str,
    original_query: Optional[str] = None,
    rewritten_query: Optional[str] = None,
    candidates: Optional[List[dict]] = None,
) -> dict:
    return {
        "pipeline": pipeline,
        "observed": observed,
        "question": question,
        "answer": answer,
        "source_id": source_id,
        "source": SOURCE,
        "top_k": top_k,
        "retrieved_ids": [hit["id"] for hit in hits],
        "user_question": user_question,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "candidates": candidates or [],
    }


def _paragraphs(rows: Sequence[dict]) -> List[dict]:
    chunks = []
    for row_index, row in enumerate(rows):
        context = row.get("context") or {}
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        source_id = str(row.get("id") or row_index)
        for index, title in enumerate(titles):
            parts = sentences[index] if index < len(sentences) else []
            text = " ".join(part.strip() for part in parts if str(part).strip())
            if not text:
                continue
            chunks.append(
                {
                    "id": f"p{row_index}-{index}",
                    "text": text,
                    "title": str(title),
                    "source_id": source_id,
                }
            )
    return chunks


def _paragraphs_by_row(paragraphs: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for chunk in paragraphs:
        grouped.setdefault(chunk["source_id"], []).append(chunk)
    return grouped


def _windows(paragraphs: Sequence[dict]) -> List[dict]:
    windows = []
    for paragraph in paragraphs:
        words = paragraph["text"].split()
        start = 0
        piece = 0
        while start < len(words):
            text = " ".join(words[start : start + WINDOW_WORDS])
            if len(text.split()) >= 8:
                windows.append(
                    {
                        "id": f"{paragraph['id']}-w{piece}",
                        "text": text,
                        "parent_id": paragraph["id"],
                        "source_id": paragraph["source_id"],
                    }
                )
                piece += 1
            if start + WINDOW_WORDS >= len(words):
                break
            start += WINDOW_WORDS
    return windows


def _covering_run(windows: Sequence[dict], answer: str) -> List[dict]:
    for start, _window in enumerate(windows):
        combined = ""
        for end in range(start, len(windows)):
            combined = f"{combined} {windows[end]['text']}".strip()
            if _contains(combined, answer):
                return list(windows[start : end + 1])
    return []


def _best_rank(ranked: Sequence, wanted: Iterable[str]) -> Optional[int]:
    wanted = set(wanted)
    for rank, (chunk, _score) in enumerate(ranked, start=1):
        if chunk["id"] in wanted:
            return rank
    return None


def _hit_from_ranked(ranked: Sequence, position: int) -> dict:
    chunk, score = ranked[position]
    return {"id": chunk["id"], "text": chunk.get("text") or "", "score": score, "rank": position + 1}


def _ids(hits: Sequence[dict]) -> set:
    return {hit["id"] for hit in hits}


def _contains(text: str, answer: str) -> bool:
    pattern = r"\b" + re.escape(answer.strip().lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def fetch_hotpot_rows(limit: int = 160) -> List[dict]:
    import urllib.request

    rows = []
    offset = 0
    while len(rows) < limit:
        page = min(50, limit - len(rows))
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation&offset={offset}&length={page}"
        )
        payload = json.load(urllib.request.urlopen(url, timeout=90))
        batch = [item["row"] for item in payload.get("rows") or []]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:limit]


def main() -> None:
    rows = fetch_hotpot_rows(160)
    payload = select_records(rows)
    write_dataset(payload)
    print(json.dumps({"rows": len(rows), "counts": payload["counts"], "kept": len(payload["records"])}, indent=2))


if __name__ == "__main__":
    main()
