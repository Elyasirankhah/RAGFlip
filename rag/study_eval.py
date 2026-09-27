"""Score the sealed development split.

The reported label is the SQuAD answer string, not the internal judge.
The final-test ids are refused. This does not change the selection rule.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag.experiment import choose_experiment, execute_experiment
from rag.pipelines.study_set import DEV_IDS_PATH, ROOT, load_study_cases
from rag.selective import candidate_actions, choose_scored, select_repair
from rag.study import dataset_supported, summarize_outcomes

DEV_POLICIES = (
    "always_increase_k",
    "always_rerun",
    "one_shot",
    "selective_no_guard",
    "selective",
    "oracle",
)
CHECKPOINT = ROOT / "dev_score.jsonl"


def score_development(
    root: Path = ROOT,
    checkpoint: Optional[Path] = None,
    limit: Optional[int] = None,
    analyzer: Any = None,
    judge: str = "overlap",
) -> Dict[str, Any]:
    """Run development cases only. Raises if asked to use the final test."""
    dev_ids = set(json.loads((root / "dev_ids.json").read_text(encoding="utf-8")))
    test_ids = set(json.loads((root / "test_ids.json").read_text(encoding="utf-8")))
    if dev_ids & test_ids:
        raise ValueError("Development and final-test ids overlap.")
    cases = [
        case
        for case in load_study_cases(root)
        if case["split"] == "dev" and case["name"] in dev_ids and case["name"] not in test_ids
    ]
    if limit is not None:
        cases = cases[:limit]
    if checkpoint is None:
        checkpoint = CHECKPOINT if judge == "overlap" else root / f"dev_score_{judge}.jsonl"
    saved = _load(checkpoint)
    if analyzer is None:
        analyzer = study_analyzer(judge)
    pending = [case for case in cases if case["name"] not in saved]
    total = len(cases)
    for offset, case in enumerate(pending, start=1):
        row = _score_case(case, analyzer)
        saved[case["name"]] = row
        _append(checkpoint, row)
        if offset % 25 == 0 or offset == len(pending):
            print(f"[{len(saved)}/{total}] {case['name']} {case['environment']} {case['kind']}", flush=True)
    rows = [saved[case["name"]] for case in cases if case["name"] in saved]
    summary = {policy: summarize_outcomes(_policy_rows(rows, policy)) for policy in DEV_POLICIES}
    summary["split"] = "dev"
    summary["internal_judge"] = judge
    summary["label"] = "squad_answer_string"
    summary["final_test_scored"] = False
    summary["cases"] = len(rows)
    return summary


def _score_case(case: dict, analyzer: Any) -> dict:
    policies = {
        "always_increase_k": _forced(case, "increase_k"),
        "always_rerun": _forced(case, "rerun_retriever"),
        "one_shot": _from_result(case, _one_shot(case, analyzer)),
        "selective_no_guard": _from_result(case, select_repair(case["trace"], case["retriever"], analyzer, mode="no_guard")),
        "selective": _from_result(case, select_repair(case["trace"], case["retriever"], analyzer, mode="full")),
        "oracle": _oracle(case),
    }
    return {
        "name": case["name"],
        "split": "dev",
        "environment": case["environment"],
        "kind": case["kind"],
        "policies": policies,
    }


def _one_shot(case: dict, analyzer: Any) -> dict:
    from rag.diagnose import diagnose

    before = diagnose(case["trace"], analyzer=analyzer)
    experiment = choose_experiment(case["trace"], before)
    return execute_experiment(case["trace"], case["retriever"], analyzer, experiment, before)


def _forced(case: dict, name: str) -> dict:
    trace = case["trace"]
    k = trace.metadata.top_k or len(trace.retrieved_chunks) or 1
    if name == "increase_k":
        action = {"name": name, "query": trace.question, "k": min(20, max(k * 3, k + 5)), "runnable": True}
    else:
        action = {"name": name, "query": trace.question, "k": k, "runnable": True}
    return _outcome(case, action, calls=1)


def _from_result(case: dict, result: dict) -> dict:
    experiment = result.get("experiment") or {}
    name = experiment.get("name")
    abstained = name in {None, "abstain"} or not experiment.get("runnable", False)
    if abstained:
        text = _original(case)
        action = "abstain"
    else:
        text = _context_for(case, experiment)
        action = str(name)
    if "retriever_calls" in result:
        calls = int(result.get("retriever_calls") or 0)
    elif abstained or name in {None, "abstain", "merge_retrieved", "rerank_candidates"}:
        calls = 0
    else:
        calls = 1
    return _labeled(case, text, action, calls, abstained)


def _oracle(case: dict) -> dict:
    rows = []
    contexts = {}
    for action in candidate_actions(case["trace"]):
        if action["name"] == "abstain":
            continue
        text = _context_for(case, action)
        contexts[action["name"]] = text
        supported = dataset_supported(case["trace"].answer, text)
        calls = 0 if action["name"] in {"merge_retrieved", "rerank_candidates"} else 1
        if case["kind"] == "failure":
            improvement = 1.0 if supported else 0.0
            regressions = 0
        else:
            improvement = 0.0
            regressions = 0 if supported else 1
        rows.append(
            {
                "name": action["name"],
                "improvement": improvement,
                "regressions": regressions,
                "calls": calls,
            }
        )
    chosen = choose_scored(rows, "full")
    if chosen == "abstain":
        return _labeled(case, _original(case), "abstain", sum(row["calls"] for row in rows), True)
    return _labeled(case, contexts[chosen], chosen, sum(row["calls"] for row in rows), False)


def _outcome(case: dict, action: dict, calls: int) -> dict:
    return _labeled(case, _context_for(case, action), action["name"], calls, False)


def _labeled(case: dict, text: str, action: str, calls: int, abstained: bool) -> dict:
    supported = dataset_supported(case["trace"].answer, text)
    failure = case["kind"] == "failure"
    return {
        "action": action,
        "abstained": abstained,
        "repaired": bool(failure and supported and not abstained),
        "regressed": bool(case["kind"] == "control" and not supported),
        "calls": calls,
    }


def _context_for(case: dict, action: dict) -> str:
    name = action.get("name")
    trace = case["trace"]
    if name in {None, "abstain"} or not action.get("runnable", True):
        return _original(case)
    if name == "merge_retrieved":
        return "\n".join(chunk.text for chunk in trace.retrieved_chunks)
    if name == "rerank_candidates":
        ranked = sorted(trace.candidates, key=lambda chunk: chunk.score or 0.0, reverse=True)
        return "\n".join(chunk.text for chunk in ranked[: max(int(action.get("k") or 1), 1)])
    hits = case["retriever"](str(action.get("query") or trace.question), int(action["k"]))
    return "\n".join(hit.get("text") or "" for hit in hits)


def _original(case: dict) -> str:
    return "\n".join(chunk.text for chunk in case["trace"].retrieved_chunks)


def _policy_rows(rows: Sequence[dict], policy: str) -> List[dict]:
    flattened = []
    for row in rows:
        outcome = row["policies"][policy]
        flattened.append(
            {
                "kind": row["kind"],
                "repaired": outcome["repaired"],
                "regressed": outcome["regressed"],
                "abstained": outcome["abstained"],
                "calls": outcome["calls"],
            }
        )
    return flattened


def _load(path: Optional[Path]) -> Dict[str, dict]:
    saved: Dict[str, dict] = {}
    if path is None or not path.exists():
        return saved
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        saved[str(item["name"])] = item
    return saved


def _append(path: Optional[Path], item: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def study_analyzer(judge: str) -> Any:
    """Build the internal judge. Reported labels stay the answer string."""
    from rag.study import STUDY_JUDGE_MODELS

    if judge == "overlap":
        from rag.judge_eval import make_judge_analyzer

        return make_judge_analyzer("overlap")
    if judge not in STUDY_JUDGE_MODELS:
        names = ", ".join(("overlap", *STUDY_JUDGE_MODELS))
        raise SystemExit(f"Unknown study judge: {judge}. Use {names}.")
    os.environ["RAG_DEBUGGER_LOCAL_JUDGE_MODEL"] = STUDY_JUDGE_MODELS[judge]
    print(f"Study judge {judge}: {STUDY_JUDGE_MODELS[judge]}", flush=True)
    from rag.datasets.ragtruth import make_analyzer

    return make_analyzer("local")


def main() -> None:
    import argparse

    from rag.study import ROBUSTNESS_JUDGES

    parser = argparse.ArgumentParser(description="Score the development split only.")
    parser.add_argument("--judge", required=True, choices=("overlap", *ROBUSTNESS_JUDGES))
    args = parser.parse_args()
    summary = score_development(judge=args.judge)
    name = "dev_summary.json" if args.judge == "overlap" else f"dev_summary_{args.judge}.json"
    out = ROOT / name
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
