"""Intervals, paired tests, and where negative flips sit.

Uses the top-50 lists already saved by the measurement sweeps. No encoding.
Definitions live in rag.flip_metrics.
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("flip_metrics", ROOT_DIR / "rag" / "flip_metrics.py")
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
mcnemar_p = _metrics.mcnemar_p
summarize = _metrics.summarize
with_intervals = _metrics.with_intervals
wilson_interval = _metrics.wilson_interval

ROOT = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/negative_flips")
KS = (1, 5, 10, 20, 50)
RANK_BINS = ((1, 1, "1"), (2, 5, "2-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"))
RUNS = (
    {
        "dataset": "beir_nq_full",
        "ids": "nq_full_ids.npy",
        "bm25": "nq_full_bm25_top50.npy",
        "queries": "queries.parquet",
        "qrels": "qrels.tsv",
        "models": {
            "bge": ("nq_full_bge_top50.npy", "nq_full_bge_scores.npy"),
            "e5": ("nq_e5_top50.npy", "nq_e5_scores.npy"),
        },
    },
    {
        "dataset": "beir_hotpotqa_full",
        "ids": "hotpot_full_ids.npy",
        "bm25": "hotpot_full_bm25_top50.npy",
        "queries": "hotpot_queries.parquet",
        "qrels": "hotpot_qrels.tsv",
        "models": {
            "bge": ("hotpot_full_bge_top50.npy", "hotpot_full_bge_scores.npy"),
            "e5": ("hotpot_e5_top50.npy", "hotpot_e5_scores.npy"),
        },
    },
    {
        "dataset": "beir_fiqa",
        "ids": "fiqa_ids.npy",
        "bm25": "fiqa_bm25_top50.npy",
        "queries": "fiqa_queries.parquet",
        "qrels": "fiqa_qrels.tsv",
        "models": {
            "bge": ("fiqa_bge_top50.npy", None),
            "e5": ("fiqa_e5_top50.npy", None),
        },
    },
)


def main() -> None:
    import pyarrow.parquet as pq

    summary = {"study": "retriever_swap_measurement", "runs": []}
    for spec in RUNS:
        missing = [name for name in (spec["ids"], spec["bm25"], spec["queries"], spec["qrels"]) if not (ROOT / name).exists()]
        if missing:
            print(f"Skipping {spec['dataset']}: missing {missing[0]}", flush=True)
            continue
        relevant = _qrels(ROOT / spec["qrels"])
        queries = _queries(pq.read_table(ROOT / spec["queries"]))
        query_ids = sorted(query_id for query_id in queries if relevant.get(query_id))
        ids = __import__("numpy").load(ROOT / spec["ids"], allow_pickle=True)
        bm25_top = __import__("numpy").load(ROOT / spec["bm25"])
        if len(bm25_top) != len(query_ids):
            raise SystemExit(f"{spec['dataset']}: BM25 rows do not match the questions")
        lengths = [_token_count(queries[query_id]) for query_id in query_ids]
        run = {"dataset": spec["dataset"], "questions": len(query_ids), "by_model": {}, "paired": {}}
        hits = {}
        ranks = _gold_ranks(ids, relevant, query_ids, bm25_top)
        for model, (top_name, score_name) in spec["models"].items():
            if not (ROOT / top_name).exists():
                print(f"Skipping {spec['dataset']} {model}: missing {top_name}", flush=True)
                continue
            dense_top = __import__("numpy").load(ROOT / top_name)
            scores = __import__("numpy").load(ROOT / score_name) if score_name and (ROOT / score_name).exists() else None
            model_report = _model_report(ids, relevant, query_ids, bm25_top, dense_top, ranks, lengths, scores)
            run["by_model"][model] = model_report
            hits[model] = model_report.pop("_hits")
            _print_model(spec["dataset"], model, model_report)
        if "bge" in hits and "e5" in hits:
            run["paired"] = _paired(hits["bge"], hits["e5"])
            _print_paired(spec["dataset"], run["paired"])
        summary["runs"].append(run)
    out = ROOT / "flip_analysis.json"
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


def _token_count(text: str) -> int:
    return len([token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token])


def _gold_ids(ids, gold, docs) -> set:
    return gold & {str(ids[int(doc)]) for doc in docs if int(doc) >= 0}


def _gold_ranks(ids, relevant, query_ids, bm25_top):
    ranks = []
    for row, query_id in enumerate(query_ids):
        gold = relevant[query_id]
        found = None
        for rank, doc in enumerate(bm25_top[row], start=1):
            if int(doc) >= 0 and str(ids[int(doc)]) in gold:
                found = rank
                break
        ranks.append(found)
    return ranks


def _model_report(ids, relevant, query_ids, bm25_top, dense_top, ranks, lengths, scores):
    by_k = {}
    hit_rows = {}
    for k in KS:
        pairs = []
        buckets = {label: [] for _lo, _hi, label in RANK_BINS}
        length_rows = []
        margins = []
        for row, query_id in enumerate(query_ids):
            gold = relevant[query_id]
            had = ranks[row] is not None and ranks[row] <= k
            now = bool(_gold_ids(ids, gold, dense_top[row, :k]))
            pairs.append((had, now))
            if had:
                for low, high, label in RANK_BINS:
                    if low <= ranks[row] <= high:
                        buckets[label].append((True, now))
                length_rows.append((lengths[row], had and not now))
            if scores is not None and had:
                margins.append((_margin(ids, gold, dense_top[row], scores[row]), had and not now))
        summary = with_intervals(summarize(pairs, questions=len(query_ids)))
        summary["by_bm25_rank"] = {
            label: _bucket_summary(rows) for label, rows in buckets.items() if rows
        }
        summary["by_query_length"] = _length_summary(length_rows)
        if margins:
            summary["by_dense_margin"] = _margin_summary(margins)
        by_k[str(k)] = summary
        hit_rows[k] = pairs
    return {"by_k": by_k, "_hits": hit_rows}


def _bucket_summary(rows):
    broken = sum(1 for had, now in rows if had and not now)
    interval = wilson_interval(broken, len(rows))
    return {
        "bm25_supported": len(rows),
        "broken": broken,
        "negative_flip_rate": round(broken / len(rows), 4) if rows else None,
        "negative_flip_rate_ci": [round(bound, 4) for bound in interval] if interval else None,
    }


def _length_summary(rows):
    if not rows:
        return {}
    ordered = sorted(rows, key=lambda item: item[0])
    count = len(ordered)
    report = {}
    for quartile in range(4):
        chunk = ordered[quartile * count // 4 : (quartile + 1) * count // 4]
        if not chunk:
            continue
        broken = sum(1 for _length, flipped in chunk if flipped)
        interval = wilson_interval(broken, len(chunk))
        label = f"Q{quartile + 1} ({chunk[0][0]}-{chunk[-1][0]} tokens)"
        report[label] = {
            "bm25_supported": len(chunk),
            "broken": broken,
            "negative_flip_rate": round(broken / len(chunk), 4),
            "negative_flip_rate_ci": [round(bound, 4) for bound in interval],
        }
    return report


def _margin(ids, gold, dense_row, score_row):
    top_score = float(score_row[0])
    best_gold = None
    for doc, score in zip(dense_row, score_row):
        if int(doc) >= 0 and str(ids[int(doc)]) in gold:
            best_gold = float(score)
            break
    if best_gold is None:
        return None
    return top_score - best_gold


def _margin_summary(margins):
    outside = [flipped for gap, flipped in margins if gap is None]
    known = [(gap, flipped) for gap, flipped in margins if gap is not None]
    report = {
        "gap_is_top1_score_minus_best_gold_in_top_50": True,
        "gold_outside_top_50": _flip_group(outside),
    }
    if not known:
        return report
    ordered = sorted(gap for gap, _flipped in known)
    cuts = [ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))] for q in (0.33, 0.66)]
    groups = {
        "small_gap": [flipped for gap, flipped in known if gap <= cuts[0]],
        "middle_gap": [flipped for gap, flipped in known if cuts[0] < gap <= cuts[1]],
        "large_gap": [flipped for gap, flipped in known if gap > cuts[1]],
    }
    report["groups"] = {label: _flip_group(flips) for label, flips in groups.items() if flips}
    return report


def _flip_group(flips):
    if not flips:
        return {"bm25_supported": 0, "broken": 0, "negative_flip_rate": None, "negative_flip_rate_ci": None}
    broken = sum(flips)
    interval = wilson_interval(broken, len(flips))
    return {
        "bm25_supported": len(flips),
        "broken": broken,
        "negative_flip_rate": round(broken / len(flips), 4),
        "negative_flip_rate_ci": [round(bound, 4) for bound in interval],
    }


def _paired(bge_hits, e5_hits):
    paired = {}
    for k in KS:
        only_bge = only_e5 = 0
        for (bge_had, bge_now), (e5_had, e5_now) in zip(bge_hits[k], e5_hits[k]):
            if not bge_had:
                continue
            bge_flip = not bge_now
            e5_flip = not e5_now
            only_bge += int(bge_flip and not e5_flip)
            only_e5 += int(e5_flip and not bge_flip)
        paired[str(k)] = {
            "bge_flips_e5_keeps": only_bge,
            "e5_flips_bge_keeps": only_e5,
            "mcnemar_p": mcnemar_p(only_bge, only_e5),
        }
    return paired


def _print_model(dataset, model, report) -> None:
    print(f"\n{dataset} {model}", flush=True)
    print(f"{'k':>4} {'flip':>8} {'ci':>22} {'fix':>8} {'compat':>8} {'net':>8}", flush=True)
    for k, summary in report["by_k"].items():
        flip_ci = _fmt_ci(summary["negative_flip_rate_ci"])
        print(
            f"{k:>4} {_fmt(summary['negative_flip_rate']):>8} {flip_ci:>22} "
            f"{_fmt(summary['fix_rate']):>8} {_fmt(summary['compatibility']):>8} {summary['net_gain']:8d}",
            flush=True,
        )
    k10 = report["by_k"]["10"]["by_bm25_rank"]
    print("  k=10 flip rate by BM25 rank of the first relevant passage", flush=True)
    for label, bucket in k10.items():
        print(
            f"    {label:>6}  n={bucket['bm25_supported']:5d}  flip={bucket['negative_flip_rate']:.4f}  "
            f"ci={_fmt_ci(bucket['negative_flip_rate_ci'])}",
            flush=True,
        )


def _print_paired(dataset, paired) -> None:
    print(f"\n{dataset} paired BGE vs E5 flips on BM25 successes", flush=True)
    for k, row in paired.items():
        print(
            f"  k={k:>2}  only BGE flips {row['bge_flips_e5_keeps']:5d}  "
            f"only E5 flips {row['e5_flips_bge_keeps']:5d}  p={row['mcnemar_p']:.4g}",
            flush=True,
        )


def _fmt(value) -> str:
    return "   n/a" if value is None else f"{value:.4f}"


def _fmt_ci(interval) -> str:
    if not interval:
        return "n/a"
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


if __name__ == "__main__":
    main()
