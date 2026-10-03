from __future__ import annotations

import gc
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/negative_flips")
FILES = {
    "fiqa_corpus.parquet": (
        "https://huggingface.co/datasets/BeIR/fiqa/resolve/main/corpus/corpus-00000-of-00001.parquet",
        27700817,
    ),
    "fiqa_queries.parquet": (
        "https://huggingface.co/datasets/BeIR/fiqa/resolve/main/queries/queries-00000-of-00001.parquet",
        321680,
    ),
    "fiqa_qrels.tsv": (
        "https://huggingface.co/datasets/BeIR/fiqa-qrels/resolve/main/test.tsv",
        25256,
    ),
}
MODELS = {
    "bge": {
        "name": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "e5": {
        "name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}
DIM = 1024
KS = (1, 5, 10, 20, 50)
TOP = 50
RRF_C = 60
STOP = {
    "the", "a", "an", "of", "in", "to", "and", "is", "are", "was", "were", "for", "on",
    "that", "this", "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "did", "does", "do", "with", "from", "by", "or", "as", "at", "be", "been",
    "it", "its", "their", "his", "her", "they", "them", "there",
}


def main() -> None:
    import pyarrow.parquet as pq

    summary_path = ROOT / "fiqa_k_sweep.json"
    if summary_path.exists():
        print(f"Already saved: {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    _download()
    relevant = _qrels(ROOT / "fiqa_qrels.tsv")
    queries = _queries(pq.read_table(ROOT / "fiqa_queries.parquet"))
    query_ids = sorted(query_id for query_id in queries if relevant.get(query_id))
    parquet = pq.ParquetFile(ROOT / "fiqa_corpus.parquet")
    ids, doc_len, vocab, df = _meta(parquet)
    n_docs = len(ids)
    print(f"FiQA: {n_docs} passages, {len(query_ids)} questions", flush=True)
    bm25_top = _bm25_top(parquet, ids, doc_len, vocab, df, queries, query_ids)
    by_model = {}
    for key, spec in MODELS.items():
        _encode(parquet, n_docs, key, spec)
        dense_top = _dense_top(queries, query_ids, n_docs, key, spec)
        by_model[key] = _summarize(spec, ids, relevant, query_ids, bm25_top, dense_top)
        _print_table(key, by_model[key])
    summary = {
        "dataset": "beir_fiqa",
        "domain": "finance",
        "corpus_size": n_docs,
        "questions": len(query_ids),
        "ks": list(KS),
        "stopwords_removed": True,
        "label": "judged_relevant_passage_in_top_k",
        "rrf_constant": RRF_C,
        "by_model": by_model,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {summary_path}", flush=True)


def _download() -> None:
    import urllib.request

    for name, (url, expected) in FILES.items():
        path = ROOT / name
        if path.exists() and path.stat().st_size == expected:
            continue
        if path.exists():
            path.unlink()
        print(f"Downloading {name} ...", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "ragflip"})
        with urllib.request.urlopen(request, timeout=600) as response, path.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        if path.stat().st_size != expected:
            raise SystemExit(f"{name} is {path.stat().st_size} bytes, expected {expected}")


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


def _body(title, text) -> str:
    return f"{title}. {text}".strip()


def _tokens(text: str):
    return [token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token and token not in STOP]


def _docs(parquet, start: int = 0):
    index = 0
    for batch in parquet.iter_batches(batch_size=20000, columns=["_id", "title", "text"]):
        count = batch.num_rows
        if index + count <= start:
            index += count
            continue
        rows = batch.to_pydict()
        for doc_id, title, text in zip(rows["_id"], rows["title"], rows["text"]):
            if index >= start:
                yield index, str(doc_id), _body(title, text)
            index += 1


def _meta(parquet):
    ids_path = ROOT / "fiqa_ids.npy"
    if ids_path.exists() and (ROOT / "fiqa_vocab.txt").exists() and (ROOT / "fiqa_df.npy").exists():
        ids = np.load(ids_path, allow_pickle=True)
        doc_len = np.load(ROOT / "fiqa_doclen.npy")
        df = np.load(ROOT / "fiqa_df.npy")
        vocab = [line for line in (ROOT / "fiqa_vocab.txt").read_text(encoding="utf-8").split("\n") if line]
        print(f"Corpus metadata already saved ({len(ids)} passages).", flush=True)
        return ids, doc_len, vocab, df
    ids, lengths, counts = [], [], Counter()
    print("Counting FiQA ...", flush=True)
    for _index, doc_id, text in _docs(parquet):
        ids.append(doc_id)
        tokens = _tokens(text)
        lengths.append(len(tokens))
        counts.update(set(tokens))
    vocab = sorted(counts)
    df = np.asarray([counts[token] for token in vocab], dtype=np.int32)
    doc_len = np.asarray(lengths, dtype=np.int32)
    _save_npy(ids_path, np.asarray(ids, dtype=object))
    _save_npy(ROOT / "fiqa_doclen.npy", doc_len)
    _save_npy(ROOT / "fiqa_df.npy", df)
    _save_text(ROOT / "fiqa_vocab.txt", "\n".join(vocab))
    return np.asarray(ids, dtype=object), doc_len, vocab, df


def _bm25_top(parquet, ids, doc_len, vocab, df, queries, query_ids):
    out = ROOT / "fiqa_bm25_top50.npy"
    if out.exists():
        print("BM25 top-50 already saved.", flush=True)
        return np.load(out)
    print("Scoring BM25 ...", flush=True)
    n_docs = len(ids)
    token_id = {token: index for index, token in enumerate(vocab)}
    avgdl = float(doc_len.mean()) if len(doc_len) else 1.0
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    postings = [[] for _ in vocab]
    for index, _doc_id, text in _docs(parquet):
        counts = Counter(token for token in _tokens(text) if token in token_id)
        for token, freq in counts.items():
            postings[token_id[token]].append((index, min(freq, 65535)))
    top = np.full((len(query_ids), TOP), -1, dtype=np.int32)
    values = np.zeros(n_docs, dtype=np.float32)
    seen = np.zeros(n_docs, dtype=np.int32)
    for query_number, query_id in enumerate(query_ids, start=1):
        touched = []
        for token in _tokens(queries[query_id]):
            term = token_id.get(token)
            if term is None:
                continue
            weight = float(idf[term])
            for doc, freq in postings[term]:
                if seen[doc] != query_number:
                    seen[doc] = query_number
                    values[doc] = 0.0
                    touched.append(doc)
                denom = freq + 1.5 * (0.25 + 0.75 * doc_len[doc] / avgdl)
                values[doc] += weight * freq * 2.5 / denom
        ranked = sorted(touched, key=lambda doc: (-values[doc], doc))[:TOP]
        top[query_number - 1, : len(ranked)] = ranked
    _save_npy(out, top)
    return top


def _encode(parquet, n_docs: int, key: str, spec) -> None:
    path = ROOT / f"fiqa_{key}_emb.f32"
    progress_path = ROOT / f"fiqa_{key}_encode.txt"
    done = int(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else 0
    mode = "r+" if path.exists() and path.stat().st_size == n_docs * DIM * 4 else "w+"
    if mode == "w+":
        done = 0
    emb = np.memmap(path, dtype=np.float32, mode=mode, shape=(n_docs, DIM))
    if done >= n_docs:
        print(f"{key} embeddings already saved.", flush=True)
        return
    print(f"Encoding {spec['name']} from passage {done} of {n_docs} ...", flush=True)
    model = _model(spec["name"])
    batch_ids, batch_text = [], []
    prefix = spec["passage_prefix"]
    for index, _doc_id, text in _docs(parquet, done):
        batch_ids.append(index)
        batch_text.append(prefix + text if prefix else text)
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
    if done % 20480 < 64 or done == emb.shape[0]:
        print(f"  encoded {done}", flush=True)


def _dense_top(queries, query_ids, n_docs: int, key: str, spec):
    out = ROOT / f"fiqa_{key}_top50.npy"
    if out.exists():
        print(f"{key} top-50 already saved.", flush=True)
        return np.load(out)
    print(f"Scoring {spec['name']} ...", flush=True)
    import torch

    model = _model(spec["name"])
    query_vecs = model.encode(
        [spec["query_prefix"] + queries[query_id] for query_id in query_ids],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    _drop_model()
    emb = np.memmap(ROOT / f"fiqa_{key}_emb.f32", dtype=np.float32, mode="r", shape=(n_docs, DIM))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_tensor = torch.tensor(query_vecs, dtype=torch.float32, device=device)
    chunk = torch.tensor(np.asarray(emb), dtype=torch.float32, device=device)
    scores = query_tensor @ chunk.T
    _scores, local = torch.topk(scores, min(TOP, n_docs), dim=1)
    top = local.cpu().numpy().astype(np.int32)
    _save_npy(out, top)
    del chunk, scores, query_tensor
    _empty_cuda()
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


def _rrf(bm25_row, dense_row, k: int):
    scores = {}
    for rank, doc in enumerate(bm25_row):
        doc = int(doc)
        if doc < 0:
            continue
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_C + rank + 1)
    for rank, doc in enumerate(dense_row):
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


def _summarize(spec, ids, relevant, query_ids, bm25_top, dense_top):
    by_k = {}
    for k in KS:
        dense_hits, alternate_hits, rrf_hits = [], [], []
        for row, query_id in enumerate(query_ids):
            gold = relevant[query_id]
            had = _hit(ids, gold, _take(bm25_top[row], k))
            dense_hits.append((had, _hit(ids, gold, _take(dense_top[row], k))))
            alternate_hits.append((had, _hit(ids, gold, _alternate(dense_top[row], bm25_top[row], k))))
            rrf_hits.append((had, _hit(ids, gold, _rrf(bm25_top[row], dense_top[row], k))))
        by_k[str(k)] = {
            "dense_only": _counts(dense_hits, "dense_supported"),
            "alternate_dense_first": _counts(alternate_hits, "mixed_supported"),
            "rrf": _counts(rrf_hits, "mixed_supported"),
        }
    return {
        "model": spec["name"],
        "query_prefix": spec["query_prefix"],
        "passage_prefix": spec["passage_prefix"],
        "max_length": 512,
        "by_k": by_k,
    }


def _print_table(key: str, result) -> None:
    print(f"\n{key}  {result['model']}", flush=True)
    print(f"{'k':>4}  {'rule':<24} {'supported':>9} {'broken':>7} {'flip':>7} {'net':>6}", flush=True)
    for k, rules in result["by_k"].items():
        for rule, counts in rules.items():
            supported = counts.get("dense_supported", counts.get("mixed_supported"))
            print(
                f"{k:>4}  {rule:<24} {supported:9d} {counts['broken']:7d} "
                f"{counts['negative_flip_rate']:7.4f} {counts['net_gain']:6d}",
                flush=True,
            )


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


def _model(name: str):
    global _MODEL, _MODEL_NAME
    try:
        if _MODEL_NAME == name:
            return _MODEL
    except NameError:
        pass
    _drop_model()
    from sentence_transformers import SentenceTransformer

    print(f"Loading {name} ...", flush=True)
    _MODEL = SentenceTransformer(name)
    _MODEL.max_seq_length = 512
    _MODEL_NAME = name
    return _MODEL


def _drop_model() -> None:
    global _MODEL, _MODEL_NAME
    try:
        del _MODEL
        del _MODEL_NAME
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
