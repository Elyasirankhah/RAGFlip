"""Fixed-budget mix of the saved BM25 and BGE rankings.

Reads the top-50 lists already written by the full Natural Questions and
HotpotQA sweeps. For each k, compare BM25 alone, BGE alone, an alternating
mix that still returns k passages, and the union of both lists. No encoding.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/negative_flips")
KS = (1, 5, 10, 20, 50)
TOP = 50


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
    summary = {"label": "judged_relevant_passage_in_top_k", "runs": []}
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
        print(json.dumps(run, indent=2), flush=True)
    out = ROOT / "merge_k.json"
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


def _pick(left, right, k: int):
    chosen = []
    seen = set()
    for index in range(TOP):
        for doc in (int(left[index]), int(right[index])):
            if doc < 0 or doc in seen:
                continue
            seen.add(int(doc))
            chosen.append(int(doc))
            if len(chosen) == k:
                return chosen
    return chosen


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
        "mixed_supported": mixed_hit,
        "both_correct": both,
        "fixed": fixed,
        "broken": broken,
        "negative_flip_rate": round(broken / bm25_hit, 4) if bm25_hit else None,
        "net_gain": fixed - broken,
    }


def _evaluate(name, ids, relevant, query_ids, bm25_top, bge_top):
    by_k = {}
    for k in KS:
        bge_hits = []
        bge_first_hits = []
        bm25_first_hits = []
        union_hits = []
        for row, query_id in enumerate(query_ids):
            gold = relevant[query_id]
            bm25_docs = [int(doc) for doc in bm25_top[row, :k] if doc >= 0]
            bge_docs = [int(doc) for doc in bge_top[row, :k] if doc >= 0]
            had = _hit(ids, gold, bm25_docs)
            bge_hits.append((had, _hit(ids, gold, bge_docs)))
            bge_first_hits.append((had, _hit(ids, gold, _pick(bge_top[row], bm25_top[row], k))))
            bm25_first_hits.append((had, _hit(ids, gold, _pick(bm25_top[row], bge_top[row], k))))
            union_hits.append((had, _hit(ids, gold, bm25_docs + bge_docs)))
        by_k[str(k)] = {
            "passages": k,
            "bge_only": _counts(bge_hits),
            "alternate_bge_first": _counts(bge_first_hits),
            "alternate_bm25_first": _counts(bm25_first_hits),
            "union_up_to_twice_k": _counts(union_hits),
        }
    return {"dataset": name, "questions": len(query_ids), "by_k": by_k}


if __name__ == "__main__":
    main()
