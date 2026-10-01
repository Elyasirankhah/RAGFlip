from __future__ import annotations

import json
import os
import random
from pathlib import Path

from rag.datasets.real_faults import (
    DEV_SEED,
    RETRIEVAL_MISS,
    SEEN_SOURCE_IDS,
    _cases_for,
    _pick_evidence,
    load_qa_sources,
)
from rag.retrieve import dense_retrieve

TRACE_PATH = Path(__file__).resolve().parent / "traces" / "one_real.json"
DEV_LIMIT = 25
HOLDOUT_LIMIT = 100
ALREADY_RUN = frozenset({"15135", "15303"})
DEFAULT_ROOT = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/RAGTruth/dataset")


def retrieve(query: str, k: int):
    """Dense search over the saved article. Signature is (query, k)."""
    if not TRACE_PATH.exists():
        raise SystemExit(f"Missing {TRACE_PATH}. Run: python -m examples.one_real")
    corpus = json.loads(TRACE_PATH.read_text(encoding="utf-8")).get("corpus_chunks") or []
    from rag.local_models import LocalEmbedder

    hits = dense_retrieve(LocalEmbedder(), query, corpus, max(int(k), 1))
    return [{"id": chunk["id"], "text": chunk["text"]} for chunk in hits]


def write(root: Path = DEFAULT_ROOT) -> Path:
    item = _next_unused(root)
    case = next(case for case in _cases_for(item, "") if case["true_cause"] == RETRIEVAL_MISS)
    from rag.local_models import LocalEmbedder

    ranked = dense_retrieve(
        LocalEmbedder(),
        case["question"],
        case["corpus_chunks"],
        k=len(case["corpus_chunks"]),
    )
    for rank, chunk in enumerate(ranked, start=1):
        chunk["rank"] = rank
    trace = {
        "question": case["question"],
        "answer": case["answer"],
        "retrieved_chunks": case["retrieved_chunks"],
        "corpus_chunks": case["corpus_chunks"],
        "metadata": {
            "top_k": len(case["retrieved_chunks"]),
            "retriever": "bge-large-en-v1.5",
            "source_id": item["source_id"],
            "selection": "first eligible QA article after 25 dev and 100 holdout",
        },
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"source_id {item['source_id']}")
    print(f"wrote {TRACE_PATH}")
    return TRACE_PATH


def _next_unused(root: Path):
    root = Path(os.getenv("RAGTRUTH_ROOT", root))
    eligible = []
    for source in load_qa_sources(root):
        if str(source["source_id"]) in SEEN_SOURCE_IDS:
            continue
        picked = _pick_evidence(source)
        if picked is not None:
            eligible.append(picked)
    random.Random(DEV_SEED).shuffle(eligible)
    slot = DEV_LIMIT + HOLDOUT_LIMIT
    for item in eligible[slot:]:
        if str(item["source_id"]) not in ALREADY_RUN:
            return item
    raise SystemExit(f"No unused article after index {slot}.")


if __name__ == "__main__":
    write()
