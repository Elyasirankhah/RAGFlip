"""Controlled RAG fault injection. The planted cause is known."""
from __future__ import annotations

from typing import Any, Dict, List

from rag.analyzer import (
    CHUNKING_MISS,
    EMBEDDING_RETRIEVAL_FAILURE,
    HALLUCINATION,
    K_TOO_SMALL,
    RETRIEVAL_MISS,
    TraceAnalyzer,
)

QUESTION = "What is the capital of France?"
GOLD = "Paris is the capital of France."
ANSWER = "Paris is the capital of France."
HALLUCINATED = "Bananas grow on Mars."


def _fillers(n: int) -> List[Dict[str, str]]:
    return [{"id": f"d{i}", "text": f"Unrelated weather note number {i}."} for i in range(n)]


def planted_cases() -> List[Dict[str, Any]]:
    gold = {"id": "gold", "text": GOLD}
    return [
        {
            "name": "k_too_small",
            "true_cause": K_TOO_SMALL,
            "question": QUESTION,
            "answer": ANSWER,
            "retrieved_chunks": _fillers(5),
            "corpus_chunks": _fillers(5) + [{"id": "gold", "text": GOLD, "rank": 9, "score": 0.15}] + _fillers(3),
        },
        {
            "name": "chunking_miss",
            "true_cause": CHUNKING_MISS,
            "question": "When was CERN founded?",
            "answer": "CERN was founded in 1961.",
            "retrieved_chunks": [
                {"id": "r1", "text": "CERN was founded"},
                {"id": "r2", "text": "in 1961 near Geneva."},
            ],
            "corpus_chunks": [
                {"id": "r1", "text": "CERN was founded"},
                {"id": "r2", "text": "in 1961 near Geneva."},
            ],
        },
        {
            "name": "hallucination",
            "true_cause": HALLUCINATION,
            "question": QUESTION,
            "answer": HALLUCINATED,
            "retrieved_chunks": [gold],
            "corpus_chunks": [gold],
        },
        {
            "name": "embedding_retrieval_failure",
            "true_cause": EMBEDDING_RETRIEVAL_FAILURE,
            "question": QUESTION,
            "answer": ANSWER,
            "retrieved_chunks": _fillers(5),
            "corpus_chunks": _fillers(40) + [gold],
        },
    ]


FACTS = [
    ("france", "What is the capital of France?", "Paris is the capital of France.", "Paris is the capital", "of France.", "Bananas grow on Mars."),
    ("cern", "When was CERN founded?", "CERN was founded in 1961.", "CERN was founded", "in 1961.", "CERN was founded in 1492."),
    ("museum", "When did the museum on Broad Street open?", "The museum opened in 1884 on Broad Street.", "The museum opened in", "1884 on Broad Street.", "The museum opened in 2020 on King Street."),
    ("library", "When did the Oak Street library open?", "The Oak Street library opened in 1998.", "The Oak Street library opened", "in 1998.", "The Oak Street library opened in 1801."),
    ("bridge", "How long is the Harbor Bridge?", "The Harbor Bridge is 900 meters long.", "The Harbor Bridge is", "900 meters long.", "The Harbor Bridge is 12 meters long."),
    ("treaty", "Where was the treaty signed?", "The treaty was signed in Geneva.", "The treaty was signed", "in Geneva.", "The treaty was signed in Cairo."),
    ("prize", "Who won the 2014 city prize?", "Mina Cole won the 2014 city prize.", "Mina Cole won", "the 2014 city prize.", "Jonah Hale won the 2014 city prize."),
    ("dose", "What dose does the label recommend?", "The label recommends a dose of 50 mg.", "The label recommends", "a dose of 50 mg.", "The label recommends a dose of 500 mg."),
    ("flight", "What time does flight 214 depart?", "Flight 214 departs at 18:40.", "Flight 214 departs", "at 18:40.", "Flight 214 departs at 06:05."),
    ("population", "What was the town population in 2020?", "The town population in 2020 was 12000.", "The town population in 2020", "was 12000.", "The town population in 2020 was 900000."),
    ("alloy", "What is the melting point of the alloy?", "The alloy melts at 640 degrees.", "The alloy melts", "at 640 degrees.", "The alloy melts at 20 degrees."),
    ("park", "How many acres is Cedar Park?", "Cedar Park covers 80 acres.", "Cedar Park covers", "80 acres.", "Cedar Park covers 8000 acres."),
]


def _fillers_named(prefix: str, n: int) -> List[Dict[str, str]]:
    return [{"id": f"{prefix}-w{i}", "text": f"Unrelated weather note number {i} about rainfall."} for i in range(n)]


def build_expanded_cases() -> List[Dict[str, Any]]:
    """One known cause per construction. Retrieval subtypes are scored together."""
    cases = []
    for name, question, answer, left, right, false_answer in FACTS:
        full = {"id": f"{name}-full", "text": answer}
        fillers = _fillers_named(name, 8)
        cases.append(
            {
                "name": f"{name}-supported",
                "true_cause": "supported",
                "question": question,
                "answer": answer,
                "retrieved_chunks": [full] + fillers[:2],
                "corpus_chunks": [full] + fillers[:2],
            }
        )
        cases.append(
            {
                "name": f"{name}-hallucination",
                "true_cause": HALLUCINATION,
                "question": question,
                "answer": false_answer,
                "retrieved_chunks": [full],
                "corpus_chunks": [full],
            }
        )
        cases.append(
            {
                "name": f"{name}-chunking_miss",
                "true_cause": CHUNKING_MISS,
                "question": question,
                "answer": answer,
                "retrieved_chunks": [
                    {"id": f"{name}-left", "text": left},
                    {"id": f"{name}-right", "text": right},
                ],
                "corpus_chunks": [
                    {"id": f"{name}-left", "text": left},
                    {"id": f"{name}-right", "text": right},
                ],
            }
        )
        cases.append(
            {
                "name": f"{name}-retrieval_miss",
                "true_cause": RETRIEVAL_MISS,
                "question": question,
                "answer": answer,
                "retrieved_chunks": fillers[:5],
                "corpus_chunks": fillers + [full],
            }
        )
    return cases


def _family(label: str) -> str:
    if label in {K_TOO_SMALL, EMBEDDING_RETRIEVAL_FAILURE, RETRIEVAL_MISS}:
        return RETRIEVAL_MISS
    return label


def _matches(true_cause: str, predicted_fault: str, predicted_component: str) -> bool:
    predicted = predicted_component or predicted_fault
    if true_cause == predicted:
        return True
    if true_cause in {K_TOO_SMALL, EMBEDDING_RETRIEVAL_FAILURE} and predicted_fault == RETRIEVAL_MISS:
        return predicted == true_cause
    return False


def evaluate_analyzer(analyzer: TraceAnalyzer, cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    cases = cases or planted_cases()
    rows = []
    correct = 0
    family_correct = 0
    for case in cases:
        report = analyzer.analyze(
            case["question"],
            case["answer"],
            case["retrieved_chunks"],
            case.get("corpus_chunks"),
        )
        predicted_component = report.root_cause or ""
        predicted_fault = report.sentences[0].fault if report.sentences else ""
        predicted = predicted_component or predicted_fault
        ok = _matches(case["true_cause"], predicted_fault, predicted_component)
        family_ok = _family(case["true_cause"]) == _family(predicted)
        correct += int(ok)
        family_correct += int(family_ok)
        rows.append(
            {
                "name": case["name"],
                "true_cause": case["true_cause"],
                "predicted": predicted,
                "correct": ok,
                "family_correct": family_ok,
            }
        )
    total = len(cases)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "family_correct": family_correct,
        "family_accuracy": round(family_correct / total, 3) if total else 0.0,
        "cases": rows,
    }


def main() -> None:
    import argparse
    import json
    import os
    from pathlib import Path

    from rag.datasets.ragtruth import make_analyzer

    parser = argparse.ArgumentParser(description="Score planted RAG faults with a known cause")
    parser.add_argument(
        "--set",
        choices=("small", "expanded", "real", "dev", "holdout", "compare", "mixed", "judge-dev"),
        default="expanded",
    )
    parser.add_argument("--root", type=Path, default=None, help="RAGTruth dataset directory")
    parser.add_argument("--limit", type=int, default=None, help="How many real QA articles to break")
    parser.add_argument("--out", type=Path, default=None, help="Checkpoint jsonl. A rerun continues from this file.")
    parser.add_argument("--from-checkpoint", type=Path, default=None, help="Scored holdout jsonl used to choose repairable cases")
    parser.add_argument(
        "--judge-kind",
        choices=("llama", "qwen", "laya", "jev", "nli", "local", "overlap"),
        default="laya",
        help="Only for --set judge-dev. Default laya. Does not change frozen repair numbers.",
    )
    args = parser.parse_args()
    judge = os.getenv("RAG_DEBUGGER_BENCH_JUDGE", "local")
    if args.set == "judge-dev":
        if args.root is None:
            raise SystemExit("Pass --root pointing at the RAGTruth dataset directory.")
        from rag.judge_eval import load_judge_dev_examples, make_judge_analyzer, score_judge_dev

        print(
            "Judge-dev only. Frozen repair holdout and mixed results are not changed.",
            flush=True,
        )
        analyzer = make_judge_analyzer(args.judge_kind)
        examples = load_judge_dev_examples(args.root)
        print(f"Judge-dev examples: {len(examples)}", flush=True)
        report = score_judge_dev(analyzer, examples, checkpoint=args.out)
        report["judge_kind"] = args.judge_kind
    else:
        analyzer = make_analyzer(judge)
        if args.set == "compare":
            if args.root is None or args.from_checkpoint is None:
                raise SystemExit("Pass --root and --from-checkpoint for --set compare.")
            from rag.datasets.real_faults import build_holdout_cases, load_qa_sources
            from rag.repair_compare import compare_repairs, repairable_cases, scored_names

            id_path = Path(__file__).resolve().parent / "datasets" / "holdout_ids.json"
            sealed = json.loads(id_path.read_text(encoding="utf-8"))
            cases, _missing = build_holdout_cases(load_qa_sources(args.root), sealed["source_ids"])
            selected = repairable_cases(cases, scored_names(args.from_checkpoint))
            print(f"Repairable scored cases: {len(selected)}", flush=True)
            report = compare_repairs(analyzer, selected, checkpoint=args.out)
        elif args.set == "mixed":
            from rag.pipelines.mixed import CONTROL_QUOTAS, FAILURE_QUOTAS, load_mixed_cases
            from rag.repair_compare import compare_mixed

            cases = load_mixed_cases()
            counts: Dict[str, int] = {}
            pipelines: Dict[str, int] = {}
            for case in cases:
                counts[case["observed"]] = counts.get(case["observed"], 0) + 1
                pipelines[case["pipeline"]] = pipelines.get(case["pipeline"], 0) + 1
            print(f"Mixed pipeline cases: {len(cases)}", flush=True)
            print(f"Observed: {counts}", flush=True)
            print(f"Pipelines: {pipelines}", flush=True)
            print(
                "RECTIFY is not in this run. Its sandbox calls RAGVue through an LLM endpoint, "
                "not this retriever hook.",
                flush=True,
            )
            report = compare_mixed(analyzer, cases, checkpoint=args.out)
            report["observed_counts"] = counts
            report["pipeline_counts"] = pipelines
            report["failure_quotas"] = FAILURE_QUOTAS
            report["control_quotas"] = CONTROL_QUOTAS
        elif args.set in {"real", "dev", "holdout"}:
            if args.root is None:
                raise SystemExit("Pass --root pointing at the RAGTruth dataset directory.")
            from rag.datasets.real_faults import (
                SEEN_SOURCE_IDS,
                build_cases_from_ragtruth,
                build_dev_from_ragtruth,
                build_holdout_cases,
                evaluate_with_repair,
                load_qa_sources,
            )

            if args.set == "holdout":
                id_path = Path(__file__).resolve().parent / "datasets" / "holdout_ids.json"
                sealed = json.loads(id_path.read_text(encoding="utf-8"))
                source_ids = [str(source_id) for source_id in sealed["source_ids"]]
                cases, missing = build_holdout_cases(load_qa_sources(args.root), source_ids)
                print(
                    f"Holdout articles: {len(source_ids)} sealed, "
                    f"built: {len(cases) // 4}, missing: {len(missing)}, cases: {len(cases)}",
                    flush=True,
                )
                report = evaluate_with_repair(analyzer, cases, checkpoint=args.out)
                report["holdout_source_ids"] = source_ids
                report["holdout_missing_source_ids"] = missing
            elif args.set == "dev":
                limit = 25 if args.limit is None else args.limit
                built = build_dev_from_ragtruth(args.root, limit=limit)
                cases = built["cases"]
                print(
                    f"Dev articles: {len(built['dev_source_ids'])}, cases: {len(cases)}, "
                    f"sealed holdout ids: {len(built['holdout_source_ids_sealed'])}",
                    flush=True,
                )
                report = evaluate_with_repair(analyzer, cases, checkpoint=args.out)
                report["excluded_seen_source_ids"] = sorted(SEEN_SOURCE_IDS)
                report["dev_source_ids"] = built["dev_source_ids"]
                report["holdout_source_ids_sealed"] = built["holdout_source_ids_sealed"]
            else:
                limit = 8 if args.limit is None else args.limit
                cases = build_cases_from_ragtruth(args.root, limit=limit)
                print(f"Real articles: {limit} requested, cases: {len(cases)}", flush=True)
                report = evaluate_with_repair(analyzer, cases, checkpoint=args.out)
        else:
            cases = planted_cases() if args.set == "small" else build_expanded_cases()
            report = evaluate_analyzer(analyzer, cases)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
