"""Fixed-budget hybrids on the saved BM25 and BGE top-50 lists.

Every rule was chosen before looking at these scores. Each one returns
exactly k passages. Reciprocal rank fusion uses the standard constant 60.
No encoding.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/negative_flips")
KS = (1, 5, 10, 20, 50)
TOP = 50
RRF_C = 60


def main() -> None:
    import pyarrow.parquet as pq

    runs = (
        (
            "beir_nq_full",
            ROOT / "nq_full_ids.npy",
            ROOT / "nq_full_bm25_top50.npy",
            ROOT / "nq_full_bge_top50.npy",
            ROOT / "queries.parquet",
            ROOT / "qrels.tsv",
        ),
        (
            "beir_hotpotqa_full",
            ROOT / "hotpot_full_ids.npy",
            ROOT / "hotpot_full_bm25_top50.npy",
            ROOT / "hotpot_full_bge_top50.npy",
            ROOT / "hotpot_queries.parquet",
            ROOT / "hotpot_qrels.tsv",
        ),
    )
    summary = {
        "label": "judged_relevant_passage_in_top_k",
        "rrf_constant": RRF_C,
        "rules": [
            "bge_only",
            "alternate_bge_first",
            "rrf",
            "keep_bm25_top1_then_bge",
            "bge_when_top1_agrees_else_alternate",
        ],
        "runs": [],
    }
    for name, ids_path, bm25_path, bge_path, queries_path, qrels_path in runs:
        for path in (ids_path, bm25_path, bge_path, queries_path, qrels_path):
            if not path.exists():
                raise SystemExit(f"Missing {path}")
        relevant = _qrels(qrels_path)
        queries = _queries(pq.read_table(queries_path))
        query_ids = sorted(query_id for query_id in queries if relevant.get(query_id))
        ids = np.load(ids_path, allow_pickle=True)
        bm25_top = np.load(bm25_path)
        bge_top = np.load(bge_path)
        if len(query_ids) != len(bm25_top) or len(query_ids) != len(bge_top):
            raise SystemExit(f"{name}: query count does not match the saved rankings")
        run = _evaluate(name, ids, relevant, query_ids, bm25_top, bge_top)
        summary["runs"].append(run)
        _print_table(run)
    out = ROOT / "hybrid_k.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


def _qrels(path: Path):
    relevant = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[2].lstrip("-").isdigit():
            continue
        if int(parts[2]) > 0:
            relevant[parts[0]].add(parts[1])
    return relevant


def _queries(table):
    data = table.to_pydict()
    return {str(query_id): str(text) for query_id, text in zip(data["_id"], data["text"]) if text}


def _take(rows, k: int):
    chosen = []
    seen = set()
    for doc in rows:
        doc = int(doc)
        if doc < 0 or doc in seen:
            continue
        seen.add(doc)
        chosen.append(doc)
        if len(chosen) == k:
            break
    return chosen


def _alternate(left, right, k: int):
    stream = []
    for index in range(TOP):
        stream.append(int(left[index]))
        stream.append(int(right[index]))
    return _take(stream, k)


def _rrf(bm25_row, bge_row, k: int):
    scores = {}
    for rank, doc in enumerate(bm25_row):
        doc = int(doc)
        if doc < 0:
            continue
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_C + rank + 1)
    for rank, doc in enumerate(bge_row):
        doc = int(doc)
        if doc < 0:
            continue
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_C + rank + 1)
    ranked = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return ranked[:k]


def _keep_bm25_top1(bm25_row, bge_row, k: int):
    return _take(list(bm25_row[:1]) + list(bge_row) + list(bm25_row), k)


def _agree_or_alternate(bm25_row, bge_row, k: int):
    if int(bm25_row[0]) == int(bge_row[0]) and int(bge_row[0]) >= 0:
        return _take(bge_row, k)
    return _alternate(bge_row, bm25_row, k)


def _hit(ids, gold, docs) -> bool:
    return bool(gold & {str(ids[doc]) for doc in docs})


def _counts(hits):
    both = fixed = broken = bm25_hit = mixed_hit = 0
    for had, now in hits:
        bm25_hit += had
        mixed_hit += now
        both += had and now
        fixed += (not had) and now
        broken += had and (not now)
    return {
        "bm25_supported": bm25_hit,
        "supported": mixed_hit,
        "both_correct": both,
        "fixed": fixed,
        "broken": broken,
        "negative_flip_rate": round(broken / bm25_hit, 4) if bm25_hit else None,
        "net_gain": fixed - broken,
    }


def _evaluate(name, ids, relevant, query_ids, bm25_top, bge_top):
    by_k = {}
    for k in KS:
        buckets = {rule: [] for rule in (
            "bge_only",
            "alternate_bge_first",
            "rrf",
            "keep_bm25_top1_then_bge",
            "bge_when_top1_agrees_else_alternate",
        )}
        for row, query_id in enumerate(query_ids):
            gold = relevant[query_id]
            bm25_row = bm25_top[row]
            bge_row = bge_top[row]
            had = _hit(ids, gold, _take(bm25_row, k))
            picks = {
                "bge_only": _take(bge_row, k),
                "alternate_bge_first": _alternate(bge_row, bm25_row, k),
                "rrf": _rrf(bm25_row, bge_row, k),
                "keep_bm25_top1_then_bge": _keep_bm25_top1(bm25_row, bge_row, k),
                "bge_when_top1_agrees_else_alternate": _agree_or_alternate(bm25_row, bge_row, k),
            }
            for rule, docs in picks.items():
                buckets[rule].append((had, _hit(ids, gold, docs)))
        by_k[str(k)] = {rule: _counts(hits) for rule, hits in buckets.items()}
    return {"dataset": name, "questions": len(query_ids), "by_k": by_k}


def _print_table(run) -> None:
    print(f"\n{run['dataset']}  questions={run['questions']}", flush=True)
    print(f"{'k':>4}  {'rule':<40} {'supported':>9} {'broken':>7} {'flip':>7} {'net':>6}", flush=True)
    for k, rules in run["by_k"].items():
        for rule, counts in rules.items():
            print(
                f"{k:>4}  {rule:<40} {counts['supported']:9d} {counts['broken']:7d} "
                f"{counts['negative_flip_rate']:7.4f} {counts['net_gain']:6d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
