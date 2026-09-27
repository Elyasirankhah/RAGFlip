import json
import shutil

from examples.demo_retriever import retrieve
from rag.check import check_change, format_check, load_traces
from rag.judge_eval import make_judge_analyzer


def _folder(tmp_path):
    for name in ("accept_increase_k.json", "reject_increase_k.json"):
        shutil.copy(f"examples/traces/{name}", tmp_path / name)
    (tmp_path / "notes.json").write_text(json.dumps({"not": "a trace"}), encoding="utf-8")
    return tmp_path


def test_check_counts_fixed_and_mixed(tmp_path):
    traces, skipped = load_traces(_folder(tmp_path))
    assert [name for name, _ in traces] == ["accept_increase_k.json", "reject_increase_k.json"]
    assert skipped == ["notes.json"]

    result = check_change(traces, retrieve, analyzer=make_judge_analyzer("overlap"), k=6)
    outcomes = {row["name"]: row["outcome"] for row in result["traces"]}
    assert outcomes == {"accept_increase_k.json": "fixed", "reject_increase_k.json": "mixed"}
    summary = result["summary"]
    assert summary["failing_before"] == 2
    assert summary["failing_fixed"] == 1
    assert summary["claims_broken"] == 1
    assert summary["recommendation"] == "REVIEW BEFORE SHIPPING"

    text = format_check(result, judge="overlap", change="demo at k=6")
    assert "Fixed:     1 / 2 failing traces" in text
    assert "The desk lamp uses 40 watts." in text
    assert "REVIEW BEFORE SHIPPING" in text


def test_check_reads_jsonl_and_uses_logged_k(tmp_path):
    raw = json.loads(open("examples/traces/accept_increase_k.json", encoding="utf-8").read())
    path = tmp_path / "logged.jsonl"
    path.write_text(json.dumps({**raw, "id": "q1"}) + "\n", encoding="utf-8")
    traces, _ = load_traces(path)
    assert [name for name, _ in traces] == ["q1"]

    result = check_change(traces, retrieve, analyzer=make_judge_analyzer("overlap"))
    row = result["traces"][0]
    assert row["k"] == 1
    assert row["outcome"] == "unchanged"
    assert result["summary"]["recommendation"] == "NO EFFECT"


def test_cli_check_prints_report(tmp_path, capsys):
    from ragfix.cli import main

    main(["check", str(_folder(tmp_path)), "--retriever", "examples.demo_retriever:retrieve", "--k", "6", "--judge", "overlap"])
    out = capsys.readouterr().out
    assert "RAG Debugger check" in out
    assert "Skipped 1 files that are not traces." in out
