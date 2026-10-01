"""BM25 versus E5-large-v2 on the full Natural Questions and HotpotQA corpora.

Reuses the BM25 top-50 lists from the BGE sweeps. Encodes passages with the
required E5 prefix and saves a new embedding file, so the BGE files stay in
place. A killed job continues from the last saved batch.
"""
from __future__ import annotations

import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/negative_flips")
MODEL = "intfloat/e5-large-v2"
DIM = 1024
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
KS = (1, 5, 10, 20, 50)
TOP = 50
RRF_C = 60
RUNS = (
    {
        "dataset": "beir_nq_full",
        "prefix": "nq_e5",
        "ids": "nq_full_ids.npy",
        "bm25": "nq_full_bm25_top50.npy",
        "corpus": "corpus.parquet",
        "queries": "queries.parquet",
        "qrels": "qrels.tsv",
    },
    {
        "dataset": "beir_hotpotqa_full",
        "prefix": "hotpot_e5",
        "ids": "hotpot_full_ids.npy",
        "bm25": "hotpot_full_bm25_top50.npy",
        "corpus": "hotpot_corpus.parquet",
        "queries": "hotpot_queries.parquet",
        "qrels": "hotpot_qrels.tsv",
    },
)


def main() -> None:
    import pyarrow.parquet as pq

    for spec in RUNS:
        _run(spec, pq)


def _run(spec, pq) -> None:
    summary_path = ROOT / f"{spec['prefix']}_k_sweep.json"
    if summary_path.exists():
        print(f"{spec['dataset']} already saved.", flush=True)
        return
    for name in (spec["ids"], spec["bm25"], spec["corpus"], spec["queries"], spec["qrels"]):
        if not (ROOT / name).exists():
            raise SystemExit(f"Missing {ROOT / name}")
    relevant = _qrels(ROOT / spec["qrels"])
    queries = _queries(pq.read_table(ROOT / spec["queries"]))
    query_ids = sorted(query_id for query_id in queries if relevant.get(query_id))
    ids = np.load(ROOT / spec["ids"], allow_pickle=True)
    n_docs = len(ids)
    print(f"{spec['dataset']}: {n_docs} passages, {len(query_ids)} questions", flush=True)
    _encode(pq.ParquetFile(ROOT / spec["corpus"]), n_docs, spec["prefix"])
    e5_top = _e5_top(queries, query_ids, n_docs, spec["prefix"])
    bm25_top = np.load(ROOT / spec["bm25"])
    if len(bm25_top) != len(query_ids) or len(e5_top) != len(query_ids):
        raise SystemExit(f"{spec['dataset']}: ranking rows do not match the questions")
    summary = _summarize(spec["dataset"], ids, relevant, query_ids, bm25_top, e5_top, n_docs)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {summary_path}", flush=True)


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


def _docs(parquet, start: int = 0):
    index = 0
    for batch in parquet.iter_batches(batch_size=20000, columns=["_id", "title", "text"]):
        count = batch.num_rows
        if index + count <= start:
            index += count
            continue
        rows = batch.to_pydict()
        for title, text in zip(rows["title"], rows["text"]):
            if index >= start:
                yield index, PASSAGE_PREFIX + f"{title}. {text}".strip()
            index += 1


def _encode(parquet, n_docs: int, prefix: str) -> None:
    path = ROOT / f"{prefix}_emb.f32"
    progress_path = ROOT / f"{prefix}_encode.txt"
    done = int(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else 0
    mode = "r+" if path.exists() and path.stat().st_size == n_docs * DIM * 4 else "w+"
    if mode == "w+":
        done = 0
    if done == 0 and not progress_path.exists():
        print(f"Creating a {n_docs * DIM * 4 / 1e9:.1f} GB embedding file.", flush=True)
    emb = np.memmap(path, dtype=np.float32, mode=mode, shape=(n_docs, DIM))
    if done >= n_docs:
        print("E5 embeddings already saved.", flush=True)
        return
    print(f"Encoding from passage {done} of {n_docs} ...", flush=True)
    model = _model()
    batch_ids = []
    batch_text = []
    for index, text in _docs(parquet, done):
        batch_ids.append(index)
        batch_text.append(text)
        if len(batch_text) == 64:
            _write_batch(emb, model, batch_ids, batch_text, progress_path)
            batch_ids, batch_text = [], []
    if batch_text:
        _write_batch(emb, model, batch_ids, batch_text, progress_path)
    emb.flush()
    _drop_model()


def _write_batch(emb, model, batch_ids, batch_text, progress_path: Path) -> None:
    vectors = model.encode(batch_text, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    for row, vector in zip(batch_ids, vectors):
        emb[row] = vector
    emb.flush()
    done = batch_ids[-1] + 1
    _save_text(progress_path, str(done))
    if done % 20480 < 64:
        print(f"  encoded {done}", flush=True)


def _e5_top(queries, query_ids, n_docs: int, prefix: str):
    out = ROOT / f"{prefix}_top50.npy"
    if out.exists():
        print("E5 top-50 already saved.", flush=True)
        return np.load(out)
    print("Scoring E5 ...", flush=True)
    import torch

    model = _model()
    query_vecs = model.encode(
        [QUERY_PREFIX + queries[query_id] for query_id in query_ids],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    _drop_model()
    emb = np.memmap(ROOT / f"{prefix}_emb.f32", dtype=np.float32, mode="r", shape=(n_docs, DIM))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    partial = out.with_suffix(".partial.npy")
    scores_path = ROOT / f"{prefix}_scores.npy"
    block_path = ROOT / f"{prefix}_search.done"
    if partial.exists() and scores_path.exists() and block_path.exists():
        top = np.load(partial)
        best_scores = torch.tensor(np.load(scores_path), dtype=torch.float32, device=device)
        resume = int(block_path.read_text(encoding="utf-8"))
        print(f"Resuming E5 search at passage {resume}", flush=True)
    else:
        top = np.full((len(query_ids), TOP), -1, dtype=np.int32)
        best_scores = torch.full((len(query_ids), TOP), -2.0, device=device)
        resume = 0
    block = 100000
    query_tensor = torch.tensor(query_vecs, dtype=torch.float32, device=device)
    for start in range(resume, n_docs, block):
        stop = min(start + block, n_docs)
        chunk = torch.tensor(np.asarray(emb[start:stop]), dtype=torch.float32, device=device)
        scores = query_tensor @ chunk.T
        scores, local = torch.topk(scores, min(TOP, stop - start), dim=1)
        local = local + start
        merged_scores = torch.cat([best_scores, scores], dim=1)
        merged_index = torch.cat(
            [torch.tensor(top, dtype=torch.long, device=device), local.long()],
            dim=1,
        )
        pick = torch.topk(merged_scores, TOP, dim=1)
        best_scores = pick.values
        top = merged_index.gather(1, pick.indices).cpu().numpy().astype(np.int32)
        print(f"  e5 passages {stop}/{n_docs}", flush=True)
        _save_npy(partial, top)
        _save_npy(scores_path, best_scores.detach().cpu().numpy())
        _save_text(block_path, str(stop))
        del chunk, scores, local, merged_scores, merged_index, pick
        _empty_cuda()
    _save_npy(out, top)
    return top


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


def _rrf(bm25_row, e5_row, k: int):
    scores = {}
    for rank, doc in enumerate(bm25_row):
        doc = int(doc)
        if doc < 0:
            continue
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_C + rank + 1)
    for rank, doc in enumerate(e5_row):
        doc = int(doc)
        if doc < 0:
            continue
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_C + rank + 1)
    ranked = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return ranked[:k]


def _hit(ids, gold, docs) -> bool:
    return bool(gold & {str(ids[doc]) for doc in docs})


def _counts(hits, supported_name: str):
    both = fixed = broken = bm25_hit = now_hit = 0
    for had, now in hits:
        bm25_hit += had
        now_hit += now
        both += had and now
        fixed += (not had) and now
        broken += had and (not now)
    return {
        "bm25_supported": bm25_hit,
        supported_name: now_hit,
        "both_correct": both,
        "fixed": fixed,
        "broken": broken,
        "negative_flip_rate": round(broken / bm25_hit, 4) if bm25_hit else None,
        "net_gain": fixed - broken,
    }


def _summarize(dataset, ids, relevant, query_ids, bm25_top, e5_top, n_docs: int):
    by_k = {}
    for k in KS:
        e5_hits = []
        alternate_hits = []
        rrf_hits = []
        for row, query_id in enumerate(query_ids):
            gold = relevant[query_id]
            had = _hit(ids, gold, _take(bm25_top[row], k))
            e5_hits.append((had, _hit(ids, gold, _take(e5_top[row], k))))
            alternate_hits.append((had, _hit(ids, gold, _alternate(e5_top[row], bm25_top[row], k))))
            rrf_hits.append((had, _hit(ids, gold, _rrf(bm25_top[row], e5_top[row], k))))
        by_k[str(k)] = {
            "e5_only": _counts(e5_hits, "e5_supported"),
            "alternate_e5_first": _counts(alternate_hits, "mixed_supported"),
            "rrf": _counts(rrf_hits, "mixed_supported"),
        }
    return {
        "dataset": dataset,
        "corpus_size": n_docs,
        "questions": len(query_ids),
        "model": MODEL,
        "query_prefix": QUERY_PREFIX,
        "passage_prefix": PASSAGE_PREFIX,
        "max_length": 512,
        "bm25_reused": True,
        "ks": list(KS),
        "stopwords_removed": True,
        "label": "judged_relevant_passage_in_top_k",
        "rrf_constant": RRF_C,
        "by_k": by_k,
    }


def _save_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _save_npy(path: Path, array) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _model():
    global _MODEL
    try:
        return _MODEL
    except NameError:
        from sentence_transformers import SentenceTransformer

        print(f"Loading {MODEL} ...", flush=True)
        _MODEL = SentenceTransformer(MODEL)
        _MODEL.max_seq_length = 512
        return _MODEL


def _drop_model() -> None:
    global _MODEL
    try:
        del _MODEL
    except NameError:
        pass
    gc.collect()
    _empty_cuda()


def _empty_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


if __name__ == "__main__":
    main()
