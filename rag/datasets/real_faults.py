"""Break one stage in real RAGTruth QA text, then replay the suggested fix.

The words come from the dataset. The label is applied afterward and is not
shown to the debugger.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag.analyzer import (
    CHUNKING_MISS,
    EMBEDDING_RETRIEVAL_FAILURE,
    HALLUCINATION,
    K_TOO_SMALL,
    RETRIEVAL_MISS,
    SUPPORTED,
    TraceAnalyzer,
    _content_tokens,
    _covers_claim,
)
from rag.benchmark import _family, _matches
from rag.retrieve import dense_retrieve
from utils.text_utils import split_into_sentences

RETRIEVAL_LABELS = {RETRIEVAL_MISS, K_TOO_SMALL, EMBEDDING_RETRIEVAL_FAILURE}
# Articles already inspected. Development and holdout must not use them.
SEEN_SOURCE_IDS = frozenset({
    "15394",
    "14342",
    "14376",
    "15179",
    "12181",
    "12184",
    "15260",
    "15440",
})
DEV_SEED = 1
STAGE_LABELS = [SUPPORTED, HALLUCINATION, CHUNKING_MISS, RETRIEVAL_MISS]


def load_qa_sources(root: Path) -> List[Dict[str, Any]]:
    path = root / "source_info.jsonl"
    if not path.exists():
        raise SystemExit(f"RAGTruth source file not found: {path}")
    sources = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_type") != "QA":
                continue
            question, passages = _question_and_passages(row.get("source_info"))
            if question and passages:
                sources.append(
                    {
                        "source_id": str(row.get("source_id")),
                        "question": question,
                        "passages": passages,
                    }
                )
    return sources


def build_cases_from_ragtruth(root: Path, limit: int = 8, seed: int = 0) -> List[Dict[str, Any]]:
    sources = [source for source in load_qa_sources(root) if source["source_id"] not in SEEN_SOURCE_IDS]
    return build_real_fault_cases(sources, limit=limit, seed=seed)


def build_dev_from_ragtruth(root: Path, limit: int = 25, seed: int = DEV_SEED) -> Dict[str, Any]:
    cases, dev_ids, holdout_ids = build_dev_cases(load_qa_sources(root), limit=limit, seed=seed)
    return {"cases": cases, "dev_source_ids": dev_ids, "holdout_source_ids_sealed": holdout_ids}


def build_real_fault_cases(
    sources: Sequence[Dict[str, Any]],
    limit: int = 8,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    eligible = []
    for source in sources:
        picked = _pick_evidence(source)
        if picked is not None:
            eligible.append(picked)
    if len(eligible) < 2:
        raise SystemExit("Need at least two QA articles with a splittable evidence sentence.")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[:limit]
    cases = []
    for index, item in enumerate(chosen):
        foreign = _foreign_sentence(item, chosen, index)
        cases.extend(_cases_for(item, foreign))
    return cases


def build_dev_cases(
    sources: Sequence[Dict[str, Any]],
    limit: int = 25,
    seed: int = DEV_SEED,
    holdout_limit: int = 100,
) -> tuple:
    """Return dev cases plus sealed holdout ids. Holdout cases are not built."""
    eligible = []
    for source in sources:
        if str(source["source_id"]) in SEEN_SOURCE_IDS:
            continue
        picked = _pick_evidence(source)
        if picked is not None:
            eligible.append(picked)
    if len(eligible) < 2:
        raise SystemExit("Need at least two new QA articles after excluding the articles already used.")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    dev = eligible[:limit]
    holdout = eligible[limit : limit + holdout_limit]
    cases = []
    for index, item in enumerate(dev):
        foreign = _foreign_sentence(item, dev, index)
        cases.extend(_cases_for(item, foreign))
    return cases, [item["source_id"] for item in dev], [item["source_id"] for item in holdout]


def build_holdout_cases(
    sources: Sequence[Dict[str, Any]],
    source_ids: Sequence[str],
) -> tuple:
    """Build cases for the sealed article ids only. Missing ids are not replaced."""
    wanted = [str(source_id) for source_id in source_ids]
    picked_by_id = {}
    for source in sources:
        source_id = str(source["source_id"])
        if source_id not in wanted or source_id in SEEN_SOURCE_IDS:
            continue
        picked = _pick_evidence(source)
        if picked is not None:
            picked_by_id[source_id] = picked
    chosen = [picked_by_id[source_id] for source_id in wanted if source_id in picked_by_id]
    missing = [source_id for source_id in wanted if source_id not in picked_by_id]
    cases = []
    for index, item in enumerate(chosen):
        foreign = _foreign_sentence(item, chosen, index)
        cases.extend(_cases_for(item, foreign))
    return cases, missing


def replay_chunks(
    case: Dict[str, Any],
    predicted: str,
    embedder: Any = None,
) -> Optional[List[Dict[str, str]]]:
    """Replay the diagnosed experiment. Retrieval is a bounded retriever rerun."""
    if predicted == CHUNKING_MISS:
        merged = " ".join(chunk["text"] for chunk in case["retrieved_chunks"])
        return [{"id": f"{case['name']}-merged", "text": merged}]
    if predicted in RETRIEVAL_LABELS:
        if embedder is None:
            return None
        logged_k = max(len(case["retrieved_chunks"]), 1)
        if predicted == K_TOO_SMALL:
            ranks = [int(chunk["rank"]) for chunk in case.get("corpus_chunks") or [] if chunk.get("rank")]
            target = max(ranks) if ranks else logged_k * 3
            k = min(20, max(logged_k * 3, target))
        else:
            k = logged_k
        return dense_retrieve(embedder, case["question"], case.get("corpus_chunks") or [], k)
    return None


def evaluate_with_repair(
    analyzer: TraceAnalyzer,
    cases: Sequence[Dict[str, Any]],
    checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    saved = _load_checkpoint(checkpoint)
    if saved:
        print(f"Resuming: {len(saved)} already saved", flush=True)
    rejected = []
    rows = []
    pending = []
    for case in cases:
        prior = saved.get(case["name"])
        if prior and prior.get("kind") == "rejected":
            rejected.append(case["name"])
            continue
        if prior and prior.get("kind") == "scored":
            rows.append(prior["row"])
            continue
        if _structure_ok(case) and _judge_accepts(analyzer.judge, case):
            _annotate_ranks(analyzer.embedder, case)
            pending.append(case)
        else:
            rejected.append(case["name"])
            _append_checkpoint(checkpoint, {"kind": "rejected", "name": case["name"]})
    total = len(rows) + len(pending)
    already = len(rows)
    for offset, case in enumerate(pending, start=1):
        row = _score_case(analyzer, case)
        rows.append(row)
        _append_checkpoint(checkpoint, {"kind": "scored", "name": case["name"], "row": row})
        print(
            f"[{already + offset}/{total}] {case['name']} true={case['true_cause']} "
            f"pred={row['predicted']} repair={row['repair_after']}",
            flush=True,
        )
    return _report_from_rows(rows, rejected)


def _score_case(analyzer: TraceAnalyzer, case: Dict[str, Any]) -> Dict[str, Any]:
    report = analyzer.analyze(
        case["question"],
        case["answer"],
        case["retrieved_chunks"],
        case.get("corpus_chunks"),
    )
    predicted_fault = report.sentences[0].fault if report.sentences else ""
    predicted = report.root_cause or predicted_fault
    predicted_family = _family(predicted)
    family_ok = _family(case["true_cause"]) == predicted_family
    repair_after = None
    improved = False
    replay = replay_chunks(case, predicted, embedder=analyzer.embedder)
    if replay is not None:
        after = analyzer.analyze(
            case["question"],
            case["answer"],
            replay,
            case.get("corpus_chunks"),
        )
        repair_after = after.root_cause or (after.sentences[0].fault if after.sentences else "")
        improved = repair_after == SUPPORTED and predicted != SUPPORTED
    return {
        "name": case["name"],
        "true_cause": case["true_cause"],
        "predicted": predicted,
        "predicted_family": predicted_family,
        "correct": _matches(case["true_cause"], predicted_fault, report.root_cause or ""),
        "family_correct": family_ok,
        "repair_after": repair_after,
        "repair_improved": improved,
    }


def _report_from_rows(rows: Sequence[Dict[str, Any]], rejected: Sequence[str]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    family_correct = sum(bool(row["family_correct"]) for row in rows)
    repairable_rows = [row for row in rows if row["true_cause"] in {CHUNKING_MISS, RETRIEVAL_MISS}]
    repair_attempted = sum(row["repair_after"] is not None for row in rows)
    repair_improved = sum(bool(row["repair_improved"]) for row in rows)
    e2e_success = sum(
        bool(row["family_correct"] and row["repair_improved"]) for row in repairable_rows
    )
    matrix = _confusion(rows)
    return {
        "total": total,
        "rejected_ambiguous": len(rejected),
        "rejected_names": list(rejected),
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "family_correct": family_correct,
        "family_accuracy": round(family_correct / total, 3) if total else 0.0,
        "macro_f1": matrix["macro_f1"],
        "per_class": matrix["per_class"],
        "confusion_matrix": matrix["matrix"],
        "repair_attempted": repair_attempted,
        "repair_improved": repair_improved,
        "repairable": len(repairable_rows),
        "end_to_end_success": e2e_success,
        "end_to_end_rate": round(e2e_success / len(repairable_rows), 3) if repairable_rows else 0.0,
        "cases": list(rows),
    }


def _load_checkpoint(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    saved: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.exists():
        return saved
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = item.get("name")
        if name:
            saved[str(name)] = item
    return saved


def _append_checkpoint(path: Optional[Path], item: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _annotate_ranks(embedder: Any, case: Dict[str, Any]) -> None:
    corpus = case.get("corpus_chunks") or []
    if not corpus:
        return
    ranked = dense_retrieve(embedder, case["question"], corpus, k=len(corpus))
    for rank, chunk in enumerate(ranked, start=1):
        chunk["rank"] = rank
        chunk["score"] = chunk.get("score")


def _structure_ok(case: Dict[str, Any]) -> bool:
    claim = case["answer"]
    retrieved = " ".join(chunk["text"] for chunk in case["retrieved_chunks"])
    cause = case["true_cause"]
    if cause == RETRIEVAL_MISS:
        return not _covers_claim(claim, retrieved) and any(
            _covers_claim(claim, chunk["text"]) for chunk in case["corpus_chunks"]
        )
    if cause == CHUNKING_MISS:
        halves = case["retrieved_chunks"][:2]
        if len(halves) < 2:
            return False
        return all(not _covers_claim(claim, chunk["text"]) for chunk in halves) and _covers_claim(
            claim, " ".join(chunk["text"] for chunk in halves)
        )
    if cause == HALLUCINATION:
        corpus = " ".join(chunk["text"] for chunk in case["corpus_chunks"])
        return not _covers_claim(claim, corpus)
    if cause == SUPPORTED:
        return any(_covers_claim(claim, chunk["text"]) for chunk in case["retrieved_chunks"])
    return True


def _judge_accepts(judge: Any, case: Dict[str, Any]) -> bool:
    claim = case["answer"]
    question = case["question"]
    retrieved = "\n\n".join(chunk["text"] for chunk in case["retrieved_chunks"])
    cause = case["true_cause"]
    if cause == SUPPORTED:
        return any(
            judge.judge(claim, chunk["text"], question).label == "supported"
            for chunk in case["retrieved_chunks"]
        )
    if cause == HALLUCINATION:
        corpus = "\n\n".join(chunk["text"] for chunk in case["corpus_chunks"])
        return judge.judge(claim, corpus, question).label == "unsupported"
    if cause == CHUNKING_MISS:
        halves = case["retrieved_chunks"][:2]
        if any(judge.judge(claim, chunk["text"], question).label == "supported" for chunk in halves):
            return False
        joined = "\n\n".join(chunk["text"] for chunk in halves)
        return judge.judge(claim, joined, question).label == "supported"
    if cause == RETRIEVAL_MISS:
        if judge.judge(claim, retrieved, question).label in {"supported", "partial"}:
            return False
        evidence = next(
            (chunk for chunk in case["corpus_chunks"] if str(chunk["id"]).endswith("-evidence")),
            None,
        )
        return evidence is not None and judge.judge(claim, evidence["text"], question).label == "supported"
    return True


def _confusion(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = list(STAGE_LABELS)
    seen = {row["true_cause"] for row in rows} | {row["predicted_family"] for row in rows}
    labels.extend(sorted(seen - set(labels)))
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for row in rows:
        true = row["true_cause"]
        pred = row["predicted_family"]
        matrix.setdefault(true, {label: 0 for label in labels})
        matrix[true][pred] = matrix[true].get(pred, 0) + 1
    per_class = {}
    f1s = []
    for label in STAGE_LABELS:
        tp = matrix.get(label, {}).get(label, 0)
        pred_count = sum(matrix.get(true, {}).get(label, 0) for true in matrix)
        gold_count = sum(matrix.get(label, {}).values())
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": gold_count,
        }
        if gold_count:
            f1s.append(f1)
    return {
        "matrix": matrix,
        "per_class": per_class,
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
    }


def _question_and_passages(info: Any) -> tuple:
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return "", []
    if not isinstance(info, dict):
        return "", []
    question = str(info.get("question") or "").strip()
    raw = info.get("passages") or []
    if isinstance(raw, str):
        raw = [raw]
    passages = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("passage") or item.get("content") or "").strip()
        else:
            text = ""
        if text:
            passages.append(text)
    return question, passages


def _pick_evidence(source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = source["question"]
    question_tokens = _content_tokens(question)
    sentences = []
    for passage in source["passages"]:
        sentences.extend(split_into_sentences(passage))
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    best = None
    best_overlap = 0
    for sentence in sentences:
        if _split_claim(sentence) is None:
            continue
        overlap = len(question_tokens & _content_tokens(sentence))
        if overlap > best_overlap:
            best = sentence
            best_overlap = overlap
    if best is None or best_overlap < 1:
        return None
    distractors = []
    for sentence in sentences:
        if sentence == best or _covers_claim(best, sentence):
            continue
        if _covers_claim(best, " ".join(distractors + [sentence])):
            continue
        distractors.append(sentence)
    if len(distractors) < 4:
        return None
    return {
        "source_id": source["source_id"],
        "question": question,
        "evidence": best,
        "left": _split_claim(best)[0],
        "right": _split_claim(best)[1],
        "distractors": distractors[:6],
    }


def _split_claim(sentence: str) -> Optional[tuple]:
    words = sentence.split()
    if len(words) < 6 or len(_content_tokens(sentence)) < 3:
        return None
    best = None
    best_distance = len(words)
    midpoint = len(words) // 2
    for cut in range(2, len(words) - 1):
        left = " ".join(words[:cut])
        right = " ".join(words[cut:])
        if _covers_claim(sentence, left) or _covers_claim(sentence, right):
            continue
        distance = abs(cut - midpoint)
        if distance < best_distance:
            best = (left, right)
            best_distance = distance
    return best


def _foreign_sentence(item: Dict[str, Any], chosen: Sequence[Dict[str, Any]], index: int) -> str:
    claim_tokens = _content_tokens(item["evidence"])
    best = None
    best_overlap = None
    for other_index, other in enumerate(chosen):
        if other_index == index:
            continue
        overlap = len(claim_tokens & _content_tokens(other["evidence"]))
        if best_overlap is None or overlap < best_overlap:
            best = other["evidence"]
            best_overlap = overlap
    return best or ""


def _cases_for(item: Dict[str, Any], foreign: str) -> List[Dict[str, Any]]:
    name = item["source_id"]
    evidence = {"id": f"{name}-evidence", "text": item["evidence"]}
    distractors = [
        {"id": f"{name}-d{i}", "text": text} for i, text in enumerate(item["distractors"])
    ]
    left = {"id": f"{name}-left", "text": item["left"]}
    right = {"id": f"{name}-right", "text": item["right"]}
    return [
        _case(f"{name}-supported", SUPPORTED, item, item["evidence"], [evidence] + distractors[:2], [evidence] + distractors[:2]),
        _case(f"{name}-hallucination", HALLUCINATION, item, foreign, distractors[:4], distractors[:4] + [evidence]),
        _case(f"{name}-chunking_miss", CHUNKING_MISS, item, item["evidence"], [left, right] + distractors[:2], [left, right] + distractors[:2]),
        _case(f"{name}-retrieval_miss", RETRIEVAL_MISS, item, item["evidence"], distractors[:4], distractors[:4] + [evidence]),
    ]


def _case(
    name: str,
    true_cause: str,
    item: Dict[str, Any],
    answer: str,
    retrieved: List[Dict[str, str]],
    corpus: List[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "name": name,
        "true_cause": true_cause,
        "question": item["question"],
        "answer": answer,
        "retrieved_chunks": retrieved,
        "corpus_chunks": corpus,
    }
