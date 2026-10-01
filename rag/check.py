"""Run one retrieval change over many logged traces and count what it fixes and breaks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rag.diagnose import diagnose
from rag.experiment import Retriever, _as_chunks, _claim_rows
from rag.trace import Trace

OUTCOMES = ("fixed", "broken", "mixed", "unchanged")


def load_traces(path: Union[str, Path]) -> Tuple[List[Tuple[str, Trace]], List[str]]:
    """A folder of .json traces, a .jsonl file, or one .json file holding a trace or a list."""
    path = Path(path)
    loaded: List[Tuple[str, Trace]] = []
    skipped: List[str] = []

    def add(name: str, raw: Any) -> None:
        if isinstance(raw, dict) and raw.get("question") and raw.get("answer"):
            loaded.append((name, Trace.from_dict(raw)))
        else:
            skipped.append(name)

    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            try:
                add(file.name, json.loads(file.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                skipped.append(file.name)
        return loaded, skipped
    if path.suffix == ".jsonl":
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                raw = json.loads(line)
                add(str(raw.get("id") or f"{path.name}:{index}") if isinstance(raw, dict) else f"{path.name}:{index}", raw)
        return loaded, skipped
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for index, raw in enumerate(data):
            add(str(raw.get("id") or f"{path.name}[{index}]") if isinstance(raw, dict) else f"{path.name}[{index}]", raw)
    else:
        add(path.name, data)
    return loaded, skipped


def check_trace(trace: Trace, retriever: Retriever, analyzer: Any = None, k: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve again with the new retriever or k, then compare every claim with the logged run."""
    use_k = int(k or trace.metadata.top_k or len(trace.retrieved_chunks) or 1)
    before = diagnose(trace, analyzer=analyzer)
    retrieved = _as_chunks(retriever(trace.question, use_k))
    after_trace = Trace(
        question=trace.question,
        answer=trace.answer,
        retrieved_chunks=retrieved,
        corpus_chunks=list(trace.corpus_chunks),
        candidates=list(trace.candidates),
        metadata=trace.metadata,
    )
    after = diagnose(after_trace, analyzer=analyzer)
    claims = _claim_rows(before, after)
    fixed = sum(1 for row in claims if row["improved"])
    broken = sum(1 for row in claims if row["regressed"])
    if fixed and broken:
        outcome = "mixed"
    elif fixed:
        outcome = "fixed"
    elif broken:
        outcome = "broken"
    else:
        outcome = "unchanged"
    return {
        "k": use_k,
        "outcome": outcome,
        "working_before": bool(claims) and all(row["before"] == "supported" for row in claims),
        "claims": claims,
        "claims_fixed": fixed,
        "claims_broken": broken,
        "retrieved_ids": [chunk.id for chunk in retrieved],
    }


def check_change(
    traces: Sequence[Tuple[str, Trace]],
    retriever: Retriever,
    analyzer: Any = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    rows = []
    for name, trace in traces:
        row = check_trace(trace, retriever, analyzer=analyzer, k=k)
        row["name"] = name
        rows.append(row)
    return {"summary": summarize(rows), "traces": rows}


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    failing = [row for row in rows if not row["working_before"]]
    working = [row for row in rows if row["working_before"]]
    counts = {outcome: sum(1 for row in rows if row["outcome"] == outcome) for outcome in OUTCOMES}
    unsupported = sum(1 for row in rows for claim in row["claims"] if claim["before"] != "supported")
    supported = sum(1 for row in rows for claim in row["claims"] if claim["before"] == "supported")
    return {
        "traces": len(rows),
        "failing_before": len(failing),
        "failing_fixed": sum(1 for row in failing if row["claims_fixed"] and not row["claims_broken"]),
        "working_before": len(working),
        "working_broken": sum(1 for row in working if row["claims_broken"]),
        "outcomes": counts,
        "claims_unsupported_before": unsupported,
        "claims_fixed": sum(row["claims_fixed"] for row in rows),
        "claims_supported_before": supported,
        "claims_broken": sum(row["claims_broken"] for row in rows),
        "recommendation": recommendation(rows),
    }


def recommendation(rows: Sequence[Dict[str, Any]]) -> str:
    broken = sum(row["claims_broken"] for row in rows)
    fixed = sum(row["claims_fixed"] for row in rows)
    if broken:
        return "REVIEW BEFORE SHIPPING"
    if fixed:
        return "NO REGRESSIONS FOUND"
    return "NO EFFECT"


def format_check(result: Dict[str, Any], judge: str = "", change: str = "", limit: int = 10) -> str:
    summary = result["summary"]
    rows = result["traces"]
    lines = ["ragfix check"]
    if judge:
        lines.append(f"Judge: {judge}")
    if change:
        lines.append(f"Change: {change}")
    lines.append("")
    lines.append(f"Traces checked: {summary['traces']}")
    lines.append(f"Fixed:     {summary['failing_fixed']} / {summary['failing_before']} failing traces")
    lines.append(f"Broken:    {summary['working_broken']} / {summary['working_before']} working traces")
    lines.append(f"Mixed:     {summary['outcomes']['mixed']} traces fixed one claim and broke another")
    lines.append(f"Unchanged: {summary['outcomes']['unchanged']}")
    lines.append("")
    lines.append(
        f"Claims: {summary['claims_fixed']} of {summary['claims_unsupported_before']} unsupported became supported, "
        f"{summary['claims_broken']} of {summary['claims_supported_before']} supported were lost"
    )
    for title, wanted in (("Broken", {"broken", "mixed"}), ("Fixed", {"fixed"})):
        picked = [row for row in rows if row["outcome"] in wanted]
        if not picked:
            continue
        lines.append("")
        lines.append(title)
        for row in picked[:limit]:
            key = "regressed" if title == "Broken" else "improved"
            claim = next((c for c in row["claims"] if c[key]), None)
            text = (claim or {}).get("claim", "")
            if len(text) > 70:
                text = text[:67] + "..."
            suffix = " (also fixed a claim)" if row["outcome"] == "mixed" else ""
            lines.append(f"  {row['name']}  \"{text}\"{suffix}")
        if len(picked) > limit:
            lines.append(f"  ... {len(picked) - limit} more")
    lines.append("")
    lines.append("Recommendation")
    lines.append(summary["recommendation"])
    lines.append("")
    lines.append("Labels come from the judge. They are not ground truth.")
    return "\n".join(lines) + "\n"
