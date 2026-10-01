from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _prepare_imports() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _configure_stdio() -> None:
    if sys.platform != "win32":
        return
    import codecs

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_report(data) -> bool:
    return isinstance(data, dict) and "claims" in data and "pipeline" in data


def resolve_judge(name: str | None) -> str:
    """Named judges, or module:function. Omit the flag to use OpenAI when a key is set."""
    if name:
        return name
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "overlap"


def _analyzer(judge: str):
    if judge == "openai":
        return None
    if ":" in judge:
        from rag.analyzer import TraceAnalyzer
        from rag.datasets.ragtruth import OverlapEmbedder
        from rag.judge import JudgeResult

        fn = _load_retriever(judge)

        class _CallableJudge:
            def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
                raw = fn(claim, evidence)
                if isinstance(raw, JudgeResult):
                    return raw
                if isinstance(raw, bool):
                    return JudgeResult("supported" if raw else "unsupported", "")
                label = str(raw).strip().lower()
                if label not in {"supported", "partial", "unsupported"}:
                    label = "unsupported"
                return JudgeResult(label, "")

        return TraceAnalyzer(OverlapEmbedder(), _CallableJudge())
    from rag.judge_eval import make_judge_analyzer

    return make_judge_analyzer(judge)


def run_analyze(path: str, as_json: bool = False, judge: str = "openai") -> str:
    from rag.diagnose import diagnose
    from rag.trace import Trace

    report = diagnose(Trace.load(path), analyzer=_analyzer(judge))
    if as_json:
        return json.dumps(report.to_dict(), indent=2) + "\n"
    text = report.format()
    return f"ragfix\nJudge: {judge}\n\n{text}"


def run_repair(path: str, retriever_spec: str, as_json: bool = False, judge: str = "openai") -> str:
    from rag.experiment import format_repair, run_verified_repair
    from rag.trace import Trace

    result = run_verified_repair(Trace.load(path), _load_retriever(retriever_spec), analyzer=_analyzer(judge))
    result["judge"] = judge
    if as_json:
        public = {key: value for key, value in result.items() if key not in {"before", "after"}}
        public["before_primary"] = ((result.get("before") or {}).get("pipeline") or {}).get("primary_failure")
        public["after_primary"] = ((result.get("after") or {}).get("pipeline") or {}).get("primary_failure")
        public["judge_supported"] = bool(result.get("judge_supported"))
        return json.dumps(public, indent=2) + "\n"
    return format_repair(result)


def run_check(
    path: str,
    retriever_spec: str,
    k: int | None = None,
    as_json: bool = False,
    judge: str = "openai",
    out: str | None = None,
) -> str:
    from rag.check import check_change, format_check, load_traces

    traces, skipped = load_traces(path)
    if not traces:
        raise SystemExit(f"No traces found in {path}.")
    result = check_change(traces, _load_retriever(retriever_spec), analyzer=_analyzer(judge), k=k)
    result["judge"] = judge
    result["retriever"] = retriever_spec
    result["k"] = k
    result["skipped"] = skipped
    if out:
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if as_json:
        return json.dumps(result, indent=2) + "\n"
    change = retriever_spec + (f" at k={k}" if k else " at each trace's logged k")
    text = format_check(result, judge=judge, change=change)
    if skipped:
        text += f"Skipped {len(skipped)} files that are not traces.\n"
    return text


def _load_retriever(spec: str):
    import importlib

    if ":" not in spec:
        raise SystemExit("Pass --retriever as module:function. The function takes (query, k) and returns chunks.")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module_name, func_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    retriever = getattr(module, func_name, None)
    if not callable(retriever):
        raise SystemExit(f"{spec} is not a callable retriever.")
    return retriever


def run_compare(before_path: str, after_path: str) -> str:
    from rag.diagnose import diagnose
    from rag.report import compare_reports
    from rag.trace import Trace

    def load(path: str):
        data = _load_json(path)
        if _is_report(data):
            return data
        return diagnose(Trace.from_dict(data)).to_dict()

    return compare_reports(load(before_path), load(after_path))


def _serve(host: str, port: int, reload: bool) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        print("  Windows:  $env:OPENAI_API_KEY='your-key-here'")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    try:
        from app import app  # noqa: F401
    except Exception as exc:
        print(f"ERROR: could not import app: {exc}")
        sys.exit(1)

    print("Starting ragfix...")
    print(f"  API:  http://{host}:{port}")
    print("  Stop: Ctrl+C")

    import uvicorn

    uvicorn.run("app:app", host=host, port=port, reload=reload)


def main(argv=None) -> None:
    _configure_stdio()
    _prepare_imports()

    parser = argparse.ArgumentParser(
        description="Test a RAG fix against your real retriever before you ship it."
    )
    sub = parser.add_subparsers(dest="command")

    analyze = sub.add_parser("analyze", help="Diagnose one trace JSON file")
    analyze.add_argument("trace")
    analyze.add_argument(
        "--judge",
        default=None,
        help="llama, qwen, openai, overlap, local, or module:function. Default: openai if OPENAI_API_KEY is set, otherwise overlap.",
    )
    analyze.add_argument("--json", action="store_true", help="Print the DiagnosisReport as JSON")

    compare = sub.add_parser("compare", help="Compare two traces or two diagnosis reports")
    compare.add_argument("before")
    compare.add_argument("after")

    repair = sub.add_parser(
        "repair",
        help="Propose one change, call your retriever, accept only if no supported claim regresses",
    )
    repair.add_argument("trace")
    repair.add_argument("--retriever", required=True, help="module:function taking (query, k) and returning chunks")
    repair.add_argument(
        "--judge",
        default=None,
        help="llama, qwen, openai, overlap, local, or module:function. Default: openai if OPENAI_API_KEY is set, otherwise overlap.",
    )
    repair.add_argument("--json", action="store_true")

    check = sub.add_parser(
        "check",
        help="Run a retriever change over many traces and list what it fixes and breaks",
    )
    check.add_argument("traces", help="Folder of .json traces, a .jsonl file, or a .json list")
    check.add_argument("--retriever", required=True, help="The changed retriever, module:function taking (query, k)")
    check.add_argument("--k", type=int, default=None, help="Retrieve at this k. Default: each trace's logged top_k")
    check.add_argument(
        "--judge",
        default=None,
        help="llama, qwen, openai, overlap, local, or module:function. Default: openai if OPENAI_API_KEY is set, otherwise overlap.",
    )
    check.add_argument("--json", action="store_true")
    check.add_argument("--out", default=None, help="Also write the full JSON result to this file")

    serve = sub.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "analyze":
        judge = resolve_judge(args.judge)
        if judge == "openai" and not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set. Use --judge overlap, llama, qwen, or module:function.")
            sys.exit(1)
        print(run_analyze(args.trace, as_json=args.json, judge=judge), end="")
        return
    if args.command == "repair":
        judge = resolve_judge(args.judge)
        if judge == "openai" and not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set. Use --judge overlap, llama, qwen, or module:function.")
            sys.exit(1)
        print(run_repair(args.trace, args.retriever, as_json=args.json, judge=judge), end="")
        return
    if args.command == "check":
        judge = resolve_judge(args.judge)
        if judge == "openai" and not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set. Use --judge overlap, llama, qwen, or module:function.")
            sys.exit(1)
        print(run_check(args.traces, args.retriever, k=args.k, as_json=args.json, judge=judge, out=args.out), end="")
        return
    if args.command == "compare":
        before = _load_json(args.before)
        after = _load_json(args.after)
        if not (_is_report(before) and _is_report(after)) and not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set. Compare saved reports, or set the key to diagnose traces.")
            sys.exit(1)
        print(run_compare(args.before, args.after), end="")
        return
    if args.command == "serve":
        _serve(args.host, args.port, args.reload)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
