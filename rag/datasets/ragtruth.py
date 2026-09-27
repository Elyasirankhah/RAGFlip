"""Convert RAGTruth into traces and score supported vs hallucination."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from rag.analyzer import TraceAnalyzer
from rag.diagnose import diagnose
from rag.judge import JudgeResult
from rag.trace import Trace

DEFAULT_ROOT = Path("Benchmark/RAGTruth-main/RAGTruth-main/dataset")
DEFAULT_OUT = Path("Benchmark/ragtruth_traces")


def _tokens(text: str) -> set:
    return {word.strip(".,!?;:\"'()[]").lower() for word in text.split() if len(word) > 2}


def source_to_text(info: Any) -> str:
    if isinstance(info, str):
        return info
    if isinstance(info, dict) and "passages" in info:
        passages = info["passages"]
        if isinstance(passages, list):
            return "\n".join(str(item) for item in passages)
        return str(passages)
    return json.dumps(info, ensure_ascii=False)


def question_of(task_type: str, info: Any) -> str:
    if task_type == "QA" and isinstance(info, dict):
        return str(info.get("question") or "")
    if task_type == "Summary":
        return "Summarize the source."
    if task_type == "Data2txt":
        return "Write an objective overview using only the provided data."
    return task_type


def gold_label(labels: List[dict]) -> str:
    return "hallucination" if labels else "supported"


def load_sources(root: Path) -> Dict[str, dict]:
    sources = {}
    with (root / "source_info.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sources[str(row["source_id"])] = row
    return sources


def iter_examples(root: Path, split: str = "test") -> Iterator[dict]:
    sources = load_sources(root)
    with (root / "response.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if split != "all" and row.get("split") != split:
                continue
            source = sources[str(row["source_id"])]
            info = source["source_info"]
            text = source_to_text(info)
            labels = row.get("labels") or []
            trace = Trace(
                question=question_of(source["task_type"], info),
                answer=row["response"],
                retrieved_chunks=[],
                corpus_chunks=[],
            )
            from rag.trace import Chunk

            chunk = Chunk(id=f"src_{row['source_id']}", text=text, doc_id=str(row["source_id"]))
            trace.retrieved_chunks = [chunk]
            trace.corpus_chunks = [chunk]
            trace.metadata.model = row.get("model")
            trace.metadata.top_k = 1
            trace.metadata.retriever = "ragtruth_source"
            trace.metadata.extra = {
                "response_id": row["id"],
                "source_id": row["source_id"],
                "task_type": source["task_type"],
                "gold": gold_label(labels),
                "split": row.get("split"),
                "label_types": sorted({item.get("label_type") for item in labels if item.get("label_type")}),
            }
            yield {"gold": trace.metadata.extra["gold"], "trace": trace.to_dict()}


def export_split(root: Path, out_path: Path, split: str = "test") -> Dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"supported": 0, "hallucination": 0}
    with out_path.open("w", encoding="utf-8") as handle:
        for example in iter_examples(root, split):
            counts[example["gold"]] += 1
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    counts["total"] = counts["supported"] + counts["hallucination"]
    return counts


def read_examples(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_slice(rows: Iterable[dict], limit: int, seed: int = 0) -> List[dict]:
    grouped: Dict[str, List[dict]] = {"supported": [], "hallucination": []}
    for row in rows:
        grouped.setdefault(row["gold"], []).append(row)
    rng = random.Random(seed)
    half = limit // 2
    picked = []
    for label in ("hallucination", "supported"):
        pool = list(grouped.get(label) or [])
        rng.shuffle(pool)
        picked.extend(pool[:half])
    rng.shuffle(picked)
    return picked[:limit]


class OverlapEmbedder:
    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * 128
        for token in _tokens(text):
            vec[hash(token) % 128] += 1.0
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


class OverlapJudge:
    """Local stand-in used when no API key is available."""

    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        claim_tokens = _tokens(claim)
        evidence_tokens = _tokens(evidence)
        if not claim_tokens or not evidence_tokens:
            return JudgeResult("unsupported", "empty claim or evidence")
        overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        if overlap >= 0.55:
            return JudgeResult("supported", f"token overlap {overlap:.2f}")
        return JudgeResult("unsupported", f"token overlap {overlap:.2f}")


def predict_response(example: dict, analyzer: TraceAnalyzer) -> str:
    trace = Trace.from_dict(example["trace"])
    report = diagnose(trace, analyzer=analyzer)
    if not report.claims:
        return "hallucination"
    if any(claim.label != "supported" for claim in report.claims):
        return "hallucination"
    return "supported"


def make_analyzer(judge_name: str) -> TraceAnalyzer:
    if judge_name == "llm":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. The overlap score is already saved; set the key to run the entailment judge.")
        from rag.embeddings import EmbeddingGenerator
        from rag.judge import SupportJudge

        return TraceAnalyzer(EmbeddingGenerator(), SupportJudge())
    if judge_name == "local":
        from rag.local_models import DEFAULT_EMBED_MODEL, DEFAULT_JUDGE_MODEL, LocalEmbedder, LocalSupportJudge

        print(
            f"Local judge: {os.getenv('RAG_DEBUGGER_LOCAL_JUDGE_MODEL', DEFAULT_JUDGE_MODEL)}\n"
            f"Local embeddings: {os.getenv('RAG_DEBUGGER_LOCAL_EMBED_MODEL', DEFAULT_EMBED_MODEL)}",
            flush=True,
        )
        return TraceAnalyzer(LocalEmbedder(), LocalSupportJudge())
    return TraceAnalyzer(OverlapEmbedder(), OverlapJudge())


def _response_id(example: dict) -> Any:
    return example["trace"]["metadata"].get("response_id")


def _load_checkpoint(path: Path) -> Dict[Any, dict]:
    saved: Dict[Any, dict] = {}
    if not path.exists():
        return saved
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            saved[row.get("response_id")] = row
    return saved


def _tally(rows: List[dict]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    by_task: Dict[str, Dict[str, int]] = {}
    for row in rows:
        gold = row["gold"]
        pred = row["pred"]
        task = row.get("task_type") or "unknown"
        bucket = by_task.setdefault(task, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        if gold == "hallucination" and pred == "hallucination":
            tp += 1
            bucket["tp"] += 1
        elif gold == "supported" and pred == "hallucination":
            fp += 1
            bucket["fp"] += 1
        elif gold == "hallucination" and pred == "supported":
            fn += 1
            bucket["fn"] += 1
        else:
            tn += 1
            bucket["tn"] += 1
    n = len(rows)
    return {
        "n": n,
        "hallucination": _prf(tp, fp, fn),
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "by_task": {name: _prf(v["tp"], v["fp"], v["fn"]) for name, v in by_task.items()},
    }


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def score_examples(
    examples: List[dict],
    analyzer: TraceAnalyzer,
    judge_name: str = "overlap",
    checkpoint_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    saved = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    rows: List[dict] = []
    pending: List[dict] = []
    for example in examples:
        response_id = _response_id(example)
        if response_id in saved:
            rows.append(saved[response_id])
        else:
            pending.append(example)
    if saved:
        print(f"Resuming: {len(rows)} already saved, {len(pending)} left", flush=True)

    total = len(examples)
    handle = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path else None
    try:
        for offset, example in enumerate(pending, start=len(rows) + 1):
            gold = example["gold"]
            pred = predict_response(example, analyzer)
            task = example["trace"]["metadata"].get("task_type") or "unknown"
            row = {"response_id": _response_id(example), "gold": gold, "pred": pred, "task_type": task}
            rows.append(row)
            if judge_name == "local":
                print(f"[{offset}/{total}] gold={gold} pred={pred}", flush=True)
            if handle is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if summary_path is not None:
                _write_summary(summary_path, {"judge": judge_name, "complete": False, **_tally(rows)})
    finally:
        if handle is not None:
            handle.close()

    report = {"judge": judge_name, "complete": len(rows) == total, **_tally(rows), "rows": rows}
    if summary_path is not None:
        _write_summary(summary_path, {k: v for k, v in report.items() if k != "rows"})
    return report


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export and score RAGTruth as traces")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--judge", choices=("overlap", "llm", "local"), default="overlap")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args(argv)

    export_path = args.out / f"{args.split}.jsonl"
    if args.skip_export and export_path.exists():
        counts = {"path": str(export_path), "note": "reused existing export"}
    else:
        counts = export_split(args.root, export_path, args.split)
        counts = {"path": str(export_path), **counts}
    rows = read_examples(export_path)
    sample = stratified_slice(rows, args.limit)
    sample_path = args.out / f"{args.split}_slice{args.limit}.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    analyzer = make_analyzer(args.judge)
    report_path = args.out / f"slice{args.limit}_{args.judge}.json"
    checkpoint_path = args.out / f"slice{args.limit}_{args.judge}_partial.jsonl"
    report = score_examples(
        sample,
        analyzer,
        judge_name=args.judge,
        checkpoint_path=checkpoint_path,
        summary_path=report_path,
    )
    report["export"] = counts if isinstance(counts, dict) and "path" in counts else {"path": str(export_path), **counts}
    report["checkpoint"] = str(checkpoint_path)
    _write_summary(report_path, {k: v for k, v in report.items() if k != "rows"})
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
