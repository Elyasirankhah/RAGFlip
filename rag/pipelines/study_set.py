"""SQuAD dev traces for the selective-repair study.

Not HotpotQA and not RAGTruth. A case is kept from its rank under one
retriever. Nothing here calls select_repair. The dev and test ids are a
function of the question id and SPLIT_SEED, written once with the traces.
"""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rag.pipelines.sparse import SparseIndex, tokenize
from rag.study import ENVIRONMENTS, SPLIT_SEED, STUDY_ID, dataset_supported
from rag.trace import Trace, TraceMetadata

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "study"
SOURCE_PATH = ROOT / "squad_dev_v1.1.json"
SOURCE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
PASSAGE_PATH = ROOT / "passages.json"
TRACE_PATH = ROOT / "traces.jsonl"
DEV_IDS_PATH = ROOT / "dev_ids.json"
TEST_IDS_PATH = ROOT / "test_ids.json"
MANIFEST_PATH = ROOT / "manifest.json"

DISTRACTORS = 15
WINDOW_WORDS = 40
CANDIDATES = 8
MIN_ANSWER_CHARS = 4
OPERATING_K = {
    "bm25": 1,
    "dense": 1,
    "hybrid": 1,
    "rerank": 1,
    "query_rewrite": 1,
    "chunk_window": 4,
    "context_budget": 6,
}
DENSE_DIM = 128


def build_records(questions: Sequence[dict], passages: Sequence[dict]) -> List[dict]:
    """Assign each question to one environment and record the operating retrieval."""
    by_id = {passage["id"]: passage for passage in passages}
    passage_ids = sorted(by_id)
    records = []
    for question in questions:
        answer = " ".join(str(question.get("answer") or "").split())
        if len(answer) < MIN_ANSWER_CHARS:
            continue
        passage_id = str(question["passage_id"])
        if passage_id not in by_id:
            continue
        name = str(question["id"])
        environment = _bucket("env", name, ENVIRONMENTS)
        pool = _pool(name, passage_id, passage_ids)
        record = _observe(name, environment, question["question"], answer, pool, by_id)
        record["split"] = _bucket("split", f"{SPLIT_SEED}:{name}", ("dev", "test"))
        records.append(record)
    return records


def write_study(records: Sequence[dict], passages: Sequence[dict], root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "passages.json").write_text(json.dumps(list(passages)), encoding="utf-8")
    lines = [json.dumps(record) for record in records]
    (root / "traces.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    dev_ids = sorted(record["name"] for record in records if record["split"] == "dev")
    test_ids = sorted(record["name"] for record in records if record["split"] == "test")
    (root / "dev_ids.json").write_text(json.dumps(dev_ids, indent=2), encoding="utf-8")
    (root / "test_ids.json").write_text(json.dumps(test_ids, indent=2), encoding="utf-8")
    manifest = _manifest(records, dev_ids, test_ids)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_study_cases(root: Path = ROOT) -> List[dict]:
    passages = json.loads((root / "passages.json").read_text(encoding="utf-8"))
    by_id = {passage["id"]: passage for passage in passages}
    cases = []
    for line in (root / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        cases.append(_materialize(record, by_id))
    return cases


def flatten_squad(payload: dict) -> Tuple[List[dict], List[dict]]:
    passages = []
    questions = []
    for article_index, article in enumerate(payload.get("data") or []):
        for paragraph_index, paragraph in enumerate(article.get("paragraphs") or []):
            passage_id = f"p{article_index}-{paragraph_index}"
            passages.append(
                {
                    "id": passage_id,
                    "text": str(paragraph.get("context") or ""),
                    "title": str(article.get("title") or ""),
                }
            )
            for qa in paragraph.get("qas") or []:
                answers = (qa.get("answers") or [])
                answer = ""
                if answers:
                    answer = str(answers[0].get("text") or "")
                questions.append(
                    {
                        "id": str(qa.get("id") or ""),
                        "passage_id": passage_id,
                        "question": str(qa.get("question") or ""),
                        "answer": answer,
                    }
                )
    return questions, passages


def ensure_squad(path: Path = SOURCE_PATH) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "rag-debugger"})
        path.write_bytes(urllib.request.urlopen(request, timeout=180).read())
    return json.loads(path.read_text(encoding="utf-8"))


def _observe(
    name: str,
    environment: str,
    question: str,
    answer: str,
    pool: Sequence[str],
    by_id: Dict[str, dict],
) -> dict:
    k = OPERATING_K[environment]
    query = question
    original = None
    rewritten = None
    chunks = [by_id[passage_id] for passage_id in pool]
    if environment == "chunk_window":
        chunks = _windows(chunks)
    index = SparseIndex(chunks)
    if environment == "query_rewrite":
        rewritten = index.keyword_query(question)
        query = rewritten or question
        original = question
    retrieved, candidates = _retrieve(environment, query, k, index)
    text = "\n".join(chunk.get("text") or "" for chunk in retrieved)
    kind = "control" if dataset_supported(answer, text) else "failure"
    return {
        "name": name,
        "environment": environment,
        "kind": kind,
        "question": query,
        "original_question": question,
        "answer": answer,
        "retrieved_ids": [chunk["id"] for chunk in retrieved],
        "corpus_ids": [chunk["id"] for chunk in chunks],
        "candidates": [
            {"id": chunk["id"], "score": chunk.get("score"), "rank": chunk.get("rank")}
            for chunk in candidates
        ],
        "metadata": {
            "retriever": environment,
            "top_k": k,
            "original_query": original,
            "rewritten_query": rewritten,
        },
    }


def _retrieve(environment: str, query: str, k: int, index: SparseIndex) -> Tuple[List[dict], List[dict]]:
    if environment == "dense":
        return _dense(index, query, k), []
    if environment == "hybrid":
        return _hybrid(index, query, k), []
    if environment == "rerank":
        pooled = index.top(query, CANDIDATES)
        scores = index.tfidf(query, [chunk["id"] for chunk in pooled])
        candidates = []
        for chunk in pooled:
            copied = dict(chunk)
            copied["score"] = float(scores.get(chunk["id"], 0.0))
            candidates.append(copied)
        return index.top(query, k), candidates
    if environment == "context_budget":
        return index.budget(query, k), []
    return index.top(query, k), []


def _materialize(record: dict, by_id: Dict[str, dict]) -> dict:
    chunks = []
    for chunk_id in record["corpus_ids"]:
        if chunk_id in by_id:
            chunks.append(by_id[chunk_id])
            continue
        passage_id, start = chunk_id.rsplit(":", 1)
        passage = by_id[passage_id]
        windows = _windows([passage])
        matched = next(window for window in windows if window["id"] == chunk_id)
        chunks.append(matched)
        del start
    index = SparseIndex(chunks)
    retrieved = []
    by_chunk = {chunk["id"]: chunk for chunk in chunks}
    for rank, chunk_id in enumerate(record["retrieved_ids"], start=1):
        chunk = dict(by_chunk[chunk_id])
        chunk["rank"] = rank
        retrieved.append(chunk)
    candidates = []
    for item in record.get("candidates") or []:
        chunk = dict(by_chunk[item["id"]])
        chunk["score"] = item.get("score")
        chunk["rank"] = item.get("rank")
        candidates.append(chunk)
    meta = record["metadata"]
    trace = Trace(
        question=record["question"],
        answer=record["answer"],
        retrieved_chunks=[_chunk(chunk) for chunk in retrieved],
        corpus_chunks=[_chunk(chunk) for chunk in chunks],
        candidates=[_chunk(chunk) for chunk in candidates],
        metadata=TraceMetadata(
            retriever=meta.get("retriever"),
            top_k=meta.get("top_k"),
            original_query=meta.get("original_query"),
            rewritten_query=meta.get("rewritten_query"),
        ),
    )

    def retrieve(query: str, k: int) -> List[dict]:
        hits, _candidates = _retrieve(record["environment"], query, int(k), index)
        if record["environment"] == "rerank":
            hits, _candidates = _retrieve("bm25", query, int(k), index)
        return [{"id": hit["id"], "text": hit.get("text") or ""} for hit in hits]

    return {
        "name": record["name"],
        "environment": record["environment"],
        "split": record["split"],
        "kind": record["kind"],
        "trace": trace,
        "retriever": retrieve,
    }


def _dense(index: SparseIndex, query: str, k: int) -> List[dict]:
    query_vec = _vector(tokenize(query))
    scored = []
    for chunk, tokens in zip(index.chunks, index.docs):
        scored.append((chunk, _cosine(query_vec, _vector(tokens))))
    scored.sort(key=lambda item: (-item[1], item[0]["id"]))
    return [_pack(chunk, score, rank) for rank, (chunk, score) in enumerate(scored[: max(int(k), 1)], start=1)]


def _hybrid(index: SparseIndex, query: str, k: int) -> List[dict]:
    lexical = index.ranked(query)
    dense_score = {chunk["id"]: score for chunk, score in _dense_scored(index, query)}
    lexical_score = {chunk["id"]: score for chunk, score in lexical}
    lexical_unit = _unit(lexical_score)
    dense_unit = _unit(dense_score)
    combined = []
    for chunk, _score in lexical:
        combined.append((chunk, 0.5 * lexical_unit[chunk["id"]] + 0.5 * dense_unit[chunk["id"]]))
    combined.sort(key=lambda item: (-item[1], item[0]["id"]))
    return [_pack(chunk, score, rank) for rank, (chunk, score) in enumerate(combined[: max(int(k), 1)], start=1)]


def _dense_scored(index: SparseIndex, query: str) -> List[Tuple[dict, float]]:
    query_vec = _vector(tokenize(query))
    return [(chunk, _cosine(query_vec, _vector(tokens))) for chunk, tokens in zip(index.chunks, index.docs)]


def _windows(passages: Sequence[dict]) -> List[dict]:
    windows = []
    for passage in passages:
        words = str(passage.get("text") or "").split()
        if not words:
            continue
        for start in range(0, len(words), WINDOW_WORDS):
            piece = words[start : start + WINDOW_WORDS]
            windows.append(
                {
                    "id": f"{passage['id']}:{start}",
                    "text": " ".join(piece),
                    "passage_id": passage["id"],
                }
            )
    return windows


def _pool(name: str, gold_id: str, passage_ids: Sequence[str]) -> List[str]:
    others = [passage_id for passage_id in passage_ids if passage_id != gold_id]
    rng = random.Random(int(hashlib.sha256(f"pool:{name}".encode()).hexdigest()[:16], 16))
    rng.shuffle(others)
    return [gold_id] + others[:DISTRACTORS]


def _bucket(prefix: str, name: str, labels: Sequence[str]) -> str:
    digest = hashlib.sha256(f"{prefix}:{name}".encode()).hexdigest()
    return labels[int(digest[:8], 16) % len(labels)]


def _vector(tokens: Sequence[str]) -> List[float]:
    vec = [0.0] * DENSE_DIM
    for token in tokens:
        bucket = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % DENSE_DIM
        vec[bucket] += 1.0
    return vec


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _unit(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {key: 0.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _pack(chunk: dict, score: float, rank: int) -> dict:
    return {"id": chunk["id"], "text": chunk.get("text") or "", "score": float(score), "rank": rank}


def _chunk(raw: dict):
    from rag.trace import Chunk

    score = raw.get("score")
    rank = raw.get("rank")
    return Chunk(
        id=str(raw["id"]),
        text=str(raw.get("text") or ""),
        rank=int(rank) if rank is not None else None,
        score=float(score) if score is not None else None,
    )


def _manifest(records: Sequence[dict], dev_ids: Sequence[str], test_ids: Sequence[str]) -> dict:
    by_environment: Dict[str, Dict[str, int]] = {}
    for record in records:
        bucket = by_environment.setdefault(record["environment"], {"failure": 0, "control": 0})
        bucket[record["kind"]] += 1
    return {
        "study_id": STUDY_ID,
        "source": "SQuAD v1.1 dev",
        "source_url": SOURCE_URL,
        "sealed_before_scoring": True,
        "method_scored": False,
        "split_seed": SPLIT_SEED,
        "cases": len(records),
        "dev": len(dev_ids),
        "test": len(test_ids),
        "failures": sum(1 for record in records if record["kind"] == "failure"),
        "controls": sum(1 for record in records if record["kind"] == "control"),
        "by_environment": by_environment,
    }


def main() -> None:
    payload = ensure_squad()
    questions, passages = flatten_squad(payload)
    records = build_records(questions, passages)
    manifest = write_study(records, passages)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
