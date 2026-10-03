from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.check import check_change, format_check
from rag.judge_eval import make_judge_analyzer
from rag.pipelines.sparse import SparseIndex, tokenize
from rag.pipelines.study_set import PASSAGE_PATH, SOURCE_PATH, flatten_squad
from rag.study import dataset_supported
from rag.trace import Trace

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOGGED_K = 1
DENSE_K = 1
BUDGET_K = 6
QUESTION_LIMIT = 150
MIN_ANSWER_TOKENS = 3
SEED = 17
TRACE_DIR = Path(__file__).resolve().parent / "wiki_traces"
CACHE_DIR = Path(__file__).resolve().parent / ".wiki_cache"

_INDEX = None


class DemoIndex:
    """BM25 over the Wikipedia passages, plus optional MiniLM vectors."""

    def __init__(self, passages, vectors=None, model=None):
        self.passages = list(passages)
        self.sparse = SparseIndex(self.passages)
        self.vectors = None if vectors is None else np.asarray(vectors, dtype=np.float32)
        self.model = model

    def bm25(self, query: str, k: int):
        return self.sparse.top(query, k)

    def budget(self, query: str, k: int):
        return self.sparse.budget(query, k)

    def dense(self, query: str, k: int):
        if self.vectors is None or self.model is None:
            raise SystemExit("Dense index is not built. Run: python -m examples.wiki_demo")
        query_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)
        order = np.argsort(-scores)[: max(int(k), 1)]
        hits = []
        for rank, index in enumerate(order, start=1):
            chunk = self.passages[int(index)]
            hits.append(
                {
                    "id": chunk["id"],
                    "text": chunk.get("text") or "",
                    "score": float(scores[int(index)]),
                    "rank": rank,
                }
            )
        return hits


def select_questions(questions, passage_ids, limit=QUESTION_LIMIT, seed=SEED):
    """One question per passage. Answers shorter than three words are skipped."""
    allowed = set(passage_ids)
    pool = [
        question
        for question in questions
        if question["passage_id"] in allowed and len(tokenize(question["answer"])) >= MIN_ANSWER_TOKENS
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)
    picked = []
    seen = set()
    for question in pool:
        if question["passage_id"] in seen:
            continue
        seen.add(question["passage_id"])
        picked.append(question)
        if len(picked) >= limit:
            break
    return picked


def logged_traces(questions, index: DemoIndex, k: int = LOGGED_K):
    """The 'before' run: BM25 at k, saved as traces."""
    rows = []
    for question in questions:
        hits = index.bm25(question["question"], k)
        trace = Trace.from_dict(
            {
                "question": question["question"],
                "answer": question["answer"],
                "retrieved_chunks": hits,
                "metadata": {
                    "retriever": "bm25",
                    "top_k": k,
                    "passage_id": question["passage_id"],
                },
            }
        )
        rows.append((_safe_name(question["id"]), trace))
    return rows


def gold_counts(questions, before_text, after_text):
    """Whether the dataset answer string is in the retrieved text."""
    failing = fixed = working = broken = 0
    for question in questions:
        had = dataset_supported(question["answer"], before_text[question["id"]])
        now = dataset_supported(question["answer"], after_text[question["id"]])
        if had:
            working += 1
            broken += not now
        else:
            failing += 1
            fixed += bool(now)
    return {"failing": failing, "fixed": fixed, "working": working, "broken": broken}


def format_gold(change: str, counts) -> str:
    return "\n".join(
        [
            change,
            "Gold answer string in the retrieved text. This is not the judge.",
            f"Fixed:  {counts['fixed']} / {counts['failing']} questions BM25 missed",
            f"Broken: {counts['broken']} / {counts['working']} questions BM25 already answered",
            "",
        ]
    )


def retrieve(query: str, k: int):
    """MiniLM dense retrieval. Used by `ragflip check --retriever examples.wiki_demo:retrieve`."""
    return _index().dense(query, k)


def retrieve_budget(query: str, k: int):
    """BM25 hits packed into a fixed word budget."""
    return _index().budget(query, k)


def main() -> None:
    passages, questions = _load_squad()
    chosen = select_questions(questions, [passage["id"] for passage in passages])
    if not chosen:
        raise SystemExit(f"No questions found. Expected passages at {PASSAGE_PATH}.")
    print("Wikipedia demo")
    print(f"Corpus: {len(passages)} SQuAD passages")
    print(f"Questions: {len(chosen)} (at least {MIN_ANSWER_TOKENS} words, one per passage)")
    print(f"Logged retrieval: BM25 top-{LOGGED_K}")
    print("This demo is not the sealed study.")
    print()

    index = DemoIndex(passages)
    rows = logged_traces(chosen, index, LOGGED_K)
    _write_traces(rows)
    before = {
        question["id"]: "\n".join(chunk.text for chunk in trace.retrieved_chunks)
        for question, (_, trace) in zip(chosen, rows)
    }

    vectors, model = _build_dense(passages)
    index.vectors = vectors
    index.model = model
    global _INDEX
    _INDEX = index

    analyzer = make_judge_analyzer("overlap")
    dense_after = {
        question["id"]: "\n".join(hit["text"] for hit in index.dense(question["question"], DENSE_K))
        for question in chosen
    }
    print(format_gold(f"Change: {MODEL_NAME} top-{DENSE_K}, replacing BM25", gold_counts(chosen, before, dense_after)))
    dense = check_change(rows, retrieve, analyzer=analyzer, k=DENSE_K)
    print(format_check(dense, judge="overlap", change=f"{MODEL_NAME} at k={DENSE_K}"))

    budget_after = {
        question["id"]: "\n".join(hit["text"] for hit in index.budget(question["question"], BUDGET_K))
        for question in chosen
    }
    print(
        format_gold(
            f"Change: BM25 top-{BUDGET_K} packed into a 90-word budget",
            gold_counts(chosen, before, budget_after),
        )
    )
    packed = check_change(rows, retrieve_budget, analyzer=analyzer, k=BUDGET_K)
    print(format_check(packed, judge="overlap", change=f"bm25 budget packer at k={BUDGET_K}"))
    print(f"Traces written to {TRACE_DIR}")
    print("Re-run the dense change with:")
    print("  ragflip check examples/wiki_traces --retriever examples.wiki_demo:retrieve --k 1 --judge overlap")


def _load_squad():
    if not PASSAGE_PATH.exists() or not SOURCE_PATH.exists():
        raise SystemExit("Missing rag/datasets/study passages. The SQuAD demo files are not in this checkout.")
    passages = json.loads(PASSAGE_PATH.read_text(encoding="utf-8"))
    questions, _ = flatten_squad(json.loads(SOURCE_PATH.read_text(encoding="utf-8")))
    return passages, questions


def _build_dense(passages):
    from examples.minilm import MiniLM, from_cache

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids_path = CACHE_DIR / "ids.json"
    vectors_path = CACHE_DIR / "vectors.npy"
    ids = [passage["id"] for passage in passages]
    cached = ids_path.exists() and vectors_path.exists() and json.loads(ids_path.read_text(encoding="utf-8")) == ids
    print(f"Loading {MODEL_NAME} from the local Hugging Face cache ...")
    model: MiniLM = from_cache(MODEL_NAME)
    if cached:
        return np.load(vectors_path), model
    print(f"Encoding {len(passages)} passages ...")
    vectors = model.encode(
        [passage.get("text") or "" for passage in passages],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    np.save(vectors_path, vectors)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    return vectors, model


def _index() -> DemoIndex:
    global _INDEX
    if _INDEX is None:
        if not (CACHE_DIR / "vectors.npy").exists():
            raise SystemExit("No dense index yet. Run: python -m examples.wiki_demo")
        passages, _ = _load_squad()
        vectors, model = _build_dense(passages)
        _INDEX = DemoIndex(passages, vectors, model)
    return _INDEX


def _write_traces(rows) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    for name, trace in rows:
        (TRACE_DIR / name).write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")


def _safe_name(question_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(question_id))
    return f"{safe}.json"


if __name__ == "__main__":
    main()
