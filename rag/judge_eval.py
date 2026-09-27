"""Score entailment judges on a small set that is not the frozen repair holdout.

Gold is RAGTruth response-level supported vs hallucination. This measures the
verifier only. It must not change the frozen repair benchmarks.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag.analyzer import TraceAnalyzer
from rag.datasets.ragtruth import OverlapEmbedder, OverlapJudge, iter_examples, predict_response

JUDGE_DEV_PATH = Path(__file__).resolve().parent / "datasets" / "judge_dev_ids.json"
JUDGE_DEV_SEED = 7
JUDGE_DEV_LIMIT = 200


def build_judge_dev_ids(root: Path, limit: int = JUDGE_DEV_LIMIT, seed: int = JUDGE_DEV_SEED) -> List[str]:
    """Pick a fixed response-id slice from RAGTruth test. Write once, then freeze."""
    rows = list(iter_examples(root, split="test"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    ids: List[str] = []
    for row in rows:
        response_id = str((row.get("trace") or {}).get("metadata", {}).get("response_id") or "")
        if response_id and response_id not in ids:
            ids.append(response_id)
        if len(ids) >= limit:
            break
    return ids


def write_judge_dev_ids(root: Path, path: Path = JUDGE_DEV_PATH) -> dict:
    ids = build_judge_dev_ids(root)
    payload = {
        "seed": JUDGE_DEV_SEED,
        "limit": JUDGE_DEV_LIMIT,
        "split": "test",
        "note": (
            "Verifier development only. Not the repair holdout. "
            "Do not retune frozen repair numbers from this."
        ),
        "response_ids": ids,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_judge_dev_examples(root: Path, path: Path = JUDGE_DEV_PATH) -> List[dict]:
    if not path.exists():
        write_judge_dev_ids(root, path)
    sealed = json.loads(path.read_text(encoding="utf-8"))
    wanted = {str(item) for item in sealed["response_ids"]}
    examples = []
    for row in iter_examples(root, split="test"):
        response_id = str((row.get("trace") or {}).get("metadata", {}).get("response_id") or "")
        if response_id in wanted:
            examples.append(row)
        if len(examples) >= len(wanted):
            break
    return examples


def make_judge_analyzer(name: str) -> TraceAnalyzer:
    from rag.local_models import DEFAULT_EMBED_MODEL, LocalEmbedder

    kind = name.lower().strip()
    if kind == "nli":
        from rag.nli_judge import DEFAULT_NLI_MODEL, NLISupportJudge

        print(
            f"NLI judge: {os.getenv('RAG_DEBUGGER_NLI_JUDGE_MODEL', DEFAULT_NLI_MODEL)}\n"
            f"Local embeddings: {os.getenv('RAG_DEBUGGER_LOCAL_EMBED_MODEL', DEFAULT_EMBED_MODEL)}",
            flush=True,
        )
        return TraceAnalyzer(LocalEmbedder(), NLISupportJudge())
    if kind in {"laya", "jev", "systemone"}:
        from rag.systemone import LayaSupportJudge

        print(
            f"SystemOne judge: backend={os.getenv('RAG_DEBUGGER_SYSTEMONE_BACKEND', 'local')} "
            f"model={os.getenv('RAG_DEBUGGER_LAYA_MODEL', 'english')}\n"
            f"Local embeddings: {os.getenv('RAG_DEBUGGER_LOCAL_EMBED_MODEL', DEFAULT_EMBED_MODEL)}",
            flush=True,
        )
        return TraceAnalyzer(LocalEmbedder(), LayaSupportJudge())
    if kind in {"llama", "llama8b", "llama-8b"}:
        os.environ["RAG_DEBUGGER_LOCAL_JUDGE_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
        from rag.datasets.ragtruth import make_analyzer

        return make_analyzer("local")
    if kind in {"qwen", "qwen27b", "qwen-27b"}:
        # Leave the default local judge (Qwen) unless the user already pinned another model.
        if "RAG_DEBUGGER_LOCAL_JUDGE_MODEL" in os.environ:
            del os.environ["RAG_DEBUGGER_LOCAL_JUDGE_MODEL"]
        from rag.datasets.ragtruth import make_analyzer

        return make_analyzer("local")
    if kind == "local":
        from rag.datasets.ragtruth import make_analyzer

        return make_analyzer("local")
    if kind == "overlap":
        return TraceAnalyzer(OverlapEmbedder(), OverlapJudge())
    raise SystemExit(
        f"Unknown judge name: {name}. Use llama, qwen, laya, nli, local, or overlap."
    )


def score_judge_dev(
    analyzer: TraceAnalyzer,
    examples: Sequence[dict],
    checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    saved = _load(checkpoint)
    rows = []
    pending = []
    for example in examples:
        response_id = str((example.get("trace") or {}).get("metadata", {}).get("response_id") or "")
        if response_id in saved:
            rows.append(saved[response_id])
        else:
            pending.append(example)
    total = len(rows) + len(pending)
    for offset, example in enumerate(pending, start=1):
        gold = example["gold"]
        pred = predict_response(example, analyzer)
        meta = (example.get("trace") or {}).get("metadata") or {}
        row = {
            "response_id": str(meta.get("response_id")),
            "gold": gold,
            "pred": pred,
            "task_type": meta.get("task_type") or meta.get("extra", {}).get("task_type")
            if isinstance(meta.get("extra"), dict)
            else meta.get("task_type"),
        }
        rows.append(row)
        _append(checkpoint, row)
        print(f"[{len(saved) + offset}/{total}] gold={gold} pred={pred}", flush=True)
    return _summarize(rows)


def _summarize(rows: Sequence[dict]) -> Dict[str, Any]:
    tp = fp = fn = 0
    correct = 0
    for row in rows:
        gold = row["gold"]
        pred = row["pred"]
        if gold == pred:
            correct += 1
        if gold == "hallucination" and pred == "hallucination":
            tp += 1
        elif gold == "supported" and pred == "hallucination":
            fp += 1
        elif gold == "hallucination" and pred == "supported":
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    majority = max(
        sum(1 for row in rows if row["gold"] == "supported"),
        sum(1 for row in rows if row["gold"] == "hallucination"),
    ) / max(len(rows), 1)
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "majority_baseline": round(majority, 4),
        "hallucination": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "note": "Verifier development set. Frozen repair benchmarks are unchanged.",
    }


def _load(path: Optional[Path]) -> Dict[str, dict]:
    saved: Dict[str, dict] = {}
    if path is None or not path.exists():
        return saved
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        response_id = item.get("response_id")
        if response_id is not None:
            saved[str(response_id)] = item
    return saved


def _append(path: Optional[Path], item: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
