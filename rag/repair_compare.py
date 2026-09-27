"""Compare repair policies on the same traces.

The planted label selects which failures are repairable. It is not given to
any policy. RECTIFY is not included: its sandbox reruns RAGVue through an LLM
endpoint and does not call this retriever hook.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from rag.analyzer import CHUNKING_MISS, RETRIEVAL_MISS
from rag.diagnose import diagnose
from rag.experiment import _experiment, choose_experiment, execute_experiment, retriever_from_corpus
from rag.trace import Chunk, Trace, TraceMetadata

POLICIES = ("ragfix", "always_increase_k", "always_rerun", "llm_choice")
Suggester = Callable[[Trace], str]


def compare_repairs(
    analyzer: Any,
    cases: Sequence[Dict[str, Any]],
    checkpoint: Optional[Path] = None,
    suggester: Optional[Suggester] = None,
) -> Dict[str, Any]:
    saved = _load(checkpoint)
    if saved:
        print(f"Resuming comparison: {len(saved)} already saved", flush=True)
    rows = []
    pending = []
    for case in cases:
        prior = saved.get(case["name"])
        if prior:
            rows.append(prior)
        else:
            pending.append(case)
    total = len(rows) + len(pending)
    for offset, case in enumerate(pending, start=1):
        row = _compare_case(analyzer, case, suggester)
        rows.append(row)
        _append(checkpoint, row)
        flags = " ".join(
            f"{name}={int(bool(row['policies'][name]['verified']))}" for name in POLICIES if name in row["policies"]
        )
        print(f"[{len(saved) + offset}/{total}] {case['name']} {flags}", flush=True)
    return _summarize(rows)


def repairable_cases(
    cases: Sequence[Dict[str, Any]],
    scored_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Keep scored retrieval and chunking failures. Order follows the sealed list."""
    wanted = set(scored_names)
    return [
        case
        for case in cases
        if case["name"] in wanted and case["true_cause"] in {RETRIEVAL_MISS, CHUNKING_MISS}
    ]


def compare_mixed(
    analyzer: Any,
    cases: Sequence[Dict[str, Any]],
    checkpoint: Optional[Path] = None,
    suggester: Optional[Suggester] = None,
) -> Dict[str, Any]:
    """Score saved pipeline traces. The observed label is not given to a policy."""
    from rag.pipelines.mixed import REPAIR_OBSERVED

    saved = _load(checkpoint)
    if saved:
        print(f"Resuming comparison: {len(saved)} already saved", flush=True)
    rows = []
    pending = [case for case in cases if case["name"] not in saved]
    rows.extend(saved[case["name"]] for case in cases if case["name"] in saved)
    total = len(rows) + len(pending)
    for offset, case in enumerate(pending, start=1):
        row = _compare_saved(analyzer, case, suggester)
        rows.append(row)
        _append(checkpoint, row)
        flags = " ".join(
            f"{name}={int(bool(row['policies'][name]['verified']))}" for name in POLICIES
        )
        print(f"[{len(saved) + offset}/{total}] {case['name']} {case['observed']} {flags}", flush=True)
    failures = [row for row in rows if row["observed"] in REPAIR_OBSERVED]
    controls = [row for row in rows if row["observed"] not in REPAIR_OBSERVED]
    present = list(POLICIES)
    return {
        "cases": len(rows),
        "failures": len(failures),
        "controls": len(controls),
        "verified_repair_rate": {name: _rate(failures, name) for name in present},
        "control_regressions": {name: _regressions(controls, name) for name in present},
        "failure_regressions": {name: _regressions(failures, name) for name in present},
        "rectify": "not run; its sandbox calls RAGVue through an LLM endpoint, not this retriever hook",
        "results": rows,
    }


def scored_names(path: Path) -> List[str]:
    names = []
    for item in _load(path).values():
        row = item.get("row") or item
        if item.get("kind") == "scored" and row.get("true_cause") in {RETRIEVAL_MISS, CHUNKING_MISS}:
            names.append(str(item["name"]))
    return names


def _compare_saved(analyzer: Any, case: Dict[str, Any], suggester: Optional[Suggester]) -> Dict[str, Any]:
    trace = case["trace"]
    return _policy_row(analyzer, case["name"], case["observed"], trace, case["retriever"], suggester)


def _compare_case(analyzer: Any, case: Dict[str, Any], suggester: Optional[Suggester]) -> Dict[str, Any]:
    trace = _trace(case)
    retriever = retriever_from_corpus(analyzer.embedder, trace.corpus_chunks)
    return _policy_row(analyzer, case["name"], case["true_cause"], trace, retriever, suggester, cause_field="true_cause")


def _policy_row(
    analyzer: Any,
    name: str,
    observed: str,
    trace: Trace,
    retriever: Callable,
    suggester: Optional[Suggester],
    cause_field: str = "observed",
) -> Dict[str, Any]:
    before = diagnose(trace, analyzer=analyzer)
    experiments = {
        "ragfix": choose_experiment(trace, before),
        "always_increase_k": _forced(trace, "increase_k"),
        "always_rerun": _forced(trace, "rerun_retriever"),
        "llm_choice": _from_suggestion(trace, _suggestion(analyzer, trace, suggester)),
    }
    cache: Dict[str, Dict[str, Any]] = {}
    policies = {}
    for policy, experiment in experiments.items():
        key = _key(experiment)
        if key not in cache:
            cache[key] = execute_experiment(trace, retriever, analyzer, experiment, before)
        result = cache[key]
        policies[policy] = {
            "experiment": (result.get("experiment") or {}).get("name"),
            "k": (result.get("experiment") or {}).get("k"),
            "runnable": bool((result.get("experiment") or {}).get("runnable")),
            "verified": bool(result.get("verified")),
            "regressions": result.get("regressions", 0),
        }
    return {"name": name, cause_field: observed, "policies": policies}


def _suggestion(analyzer: Any, trace: Trace, suggester: Optional[Suggester]) -> str:
    if suggester is not None:
        return suggester(trace)
    judge = getattr(analyzer, "judge", None)
    suggest = getattr(judge, "suggest_action", None)
    if not callable(suggest):
        return "no_retrieval_change"
    retrieved = "\n".join(chunk.text for chunk in trace.retrieved_chunks)
    return str(suggest(trace.question, trace.answer, retrieved))


def _from_suggestion(trace: Trace, name: str) -> Dict[str, Any]:
    if name == "increase_k":
        return _forced(trace, "increase_k")
    if name == "rerun_retriever":
        return _forced(trace, "rerun_retriever")
    k = trace.metadata.top_k or len(trace.retrieved_chunks) or 1
    return _experiment(
        "no_retrieval_change",
        trace.question,
        k,
        "The suggestion was not a retriever change.",
        runnable=False,
    )


def _forced(trace: Trace, name: str) -> Dict[str, Any]:
    k = trace.metadata.top_k or len(trace.retrieved_chunks) or 1
    if name == "increase_k":
        return _experiment(name, trace.question, min(20, max(k * 3, k + 5)), "Always retrieve a larger set.")
    return _experiment(name, trace.question, k, "Always call the retriever again at the same k.")


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    present = [name for name in POLICIES if rows and name in rows[0]["policies"]]
    return {
        "repairable": len(rows),
        "verified_repair_rate": {name: _rate(rows, name) for name in present},
        "retrieval_only_rate": {
            name: _rate([row for row in rows if row["true_cause"] == RETRIEVAL_MISS], name) for name in present
        },
        "chunking_only_rate": {
            name: _rate([row for row in rows if row["true_cause"] == CHUNKING_MISS], name) for name in present
        },
        "cases": list(rows),
    }


def _regressions(rows: Sequence[Dict[str, Any]], policy: str) -> int:
    return sum(int(row["policies"][policy]["regressions"] or 0) > 0 for row in rows)


def _rate(rows: Sequence[Dict[str, Any]], policy: str) -> Dict[str, Any]:
    verified = sum(bool(row["policies"][policy]["verified"]) for row in rows)
    return {
        "verified": verified,
        "total": len(rows),
        "rate": round(verified / len(rows), 3) if rows else 0.0,
    }


def _trace(case: Dict[str, Any]) -> Trace:
    return Trace(
        question=case["question"],
        answer=case["answer"],
        retrieved_chunks=[_chunk(item, index) for index, item in enumerate(case["retrieved_chunks"])],
        corpus_chunks=[_chunk(item, index) for index, item in enumerate(case["corpus_chunks"])],
        metadata=TraceMetadata(top_k=len(case["retrieved_chunks"]), retriever="dense"),
    )


def _chunk(raw: Dict[str, Any], index: int) -> Chunk:
    return Chunk(id=str(raw.get("id") or f"chunk_{index}"), text=str(raw.get("text") or ""))


def _key(experiment: Optional[Dict[str, Any]]) -> str:
    if not experiment:
        return "none"
    return json.dumps(
        [experiment.get("name"), experiment.get("query"), experiment.get("k"), experiment.get("runnable")],
        sort_keys=True,
    )


def _load(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
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


def _append(path: Optional[Path], item: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
