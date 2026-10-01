from __future__ import annotations

import gc
import json
import os
import pickle
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/negative_flips")
FILES = {
    "hotpot_corpus.parquet": (
        "https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus/corpus-00000-of-00001.parquet",
        975977704,
    ),
    "hotpot_queries.parquet": (
        "https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/queries/queries-00000-of-00001.parquet",
        8454589,
    ),
    "hotpot_qrels.tsv": (
        "https://huggingface.co/datasets/BeIR/hotpotqa-qrels/resolve/main/test.tsv",
        532620,
    ),
}
MODEL = "BAAI/bge-large-en-v1.5"
DIM = 1024
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
KS = (1, 5, 10, 20, 50)
TOP = 50
PREFIX = "hotpot_full"
STOP = {
    "the", "a", "an", "of", "in", "to", "and", "is", "are", "was", "were", "for", "on",
    "that", "this", "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "did", "does", "do", "with", "from", "by", "or", "as", "at", "be", "been",
    "it", "its", "their", "his", "her", "they", "them", "there",
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _download()
    import pyarrow.parquet as pq

    relevant = _qrels(ROOT / "hotpot_qrels.tsv")
    queries = _queries(pq.read_table(ROOT / "hotpot_queries.parquet"))
    query_ids = sorted(query_id for query_id in queries if relevant.get(query_id))
    n_docs = _meta(pq.ParquetFile(ROOT / "hotpot_corpus.parquet"))
    print(f"Corpus: {n_docs}  Queries: {len(query_ids)}", flush=True)
    _encode(pq.ParquetFile(ROOT / "hotpot_corpus.parquet"), n_docs)
    bm25_top = _bm25_top(pq.ParquetFile(ROOT / "hotpot_corpus.parquet"), n_docs, queries, query_ids)
    bge_top = _bge_top(queries, query_ids, n_docs)
    ids = np.load(ROOT / f"{PREFIX}_ids.npy", allow_pickle=True)
    summary = _summarize(ids, relevant, query_ids, bm25_top, bge_top, n_docs)
    out = ROOT / f"{PREFIX}_k_sweep.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


def _download() -> None:
    for name, (url, expected) in FILES.items():
        path = ROOT / name
        if path.exists() and path.stat().st_size == expected:
            continue
        if path.exists():
            path.unlink()
        print(f"Downloading {name} ...", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "ragfix"})
        with urllib.request.urlopen(request, timeout=600) as response, path.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                handle.flush()
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
                yield index, str(doc_id), f"{title}. {text}".strip()
            index += 1


def _tokens(text: str):
    return [token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token and token not in STOP]


def _meta(parquet) -> int:
    flag = ROOT / f"{PREFIX}_meta.done"
    checkpoint = ROOT / f"{PREFIX}_meta.pkl"
    if flag.exists():
        ids = np.load(ROOT / f"{PREFIX}_ids.npy", allow_pickle=True)
        print(f"Corpus metadata already saved ({len(ids)} passages).", flush=True)
        return len(ids)
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        ids, lengths, df = state["ids"], state["lengths"], Counter(state["df"])
        print(f"Resuming corpus count at {len(ids)}", flush=True)
    else:
        ids, lengths, df = [], [], Counter()
        print("Counting the corpus ...", flush=True)
    start = len(ids)
    for index, doc_id, text in _docs(parquet, start):
        ids.append(doc_id)
        tokens = _tokens(text)
        lengths.append(len(tokens))
        df.update(set(tokens))
        if index and index % 100000 == 0:
            _save_pickle(checkpoint, {"ids": ids, "lengths": lengths, "df": dict(df)})
            print(f"  counted {index}", flush=True)
    vocab = sorted(df)
    _save_npy(ROOT / f"{PREFIX}_ids.npy", np.asarray(ids, dtype=object))
    _save_npy(ROOT / f"{PREFIX}_doclen.npy", np.asarray(lengths, dtype=np.int32))
    _save_npy(ROOT / f"{PREFIX}_df.npy", np.asarray([df[token] for token in vocab], dtype=np.int32))
    _save_text(ROOT / f"{PREFIX}_vocab.txt", "\n".join(vocab))
    _save_text(flag, str(len(ids)))
    print(f"  counted {len(ids)}", flush=True)
    return len(ids)


def _encode(parquet, n_docs: int) -> None:
    path = ROOT / f"{PREFIX}_emb.f32"
    progress_path = ROOT / f"{PREFIX}_encode.txt"
    done = int(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else 0
    mode = "r+" if path.exists() and path.stat().st_size == n_docs * DIM * 4 else "w+"
    if mode == "w+":
        done = 0
    emb = np.memmap(path, dtype=np.float32, mode=mode, shape=(n_docs, DIM))
    if done >= n_docs:
        print("Embeddings already saved.", flush=True)
        return
    print(f"Encoding from passage {done} of {n_docs} ...", flush=True)
    model = _model()
    batch_ids = []
    batch_text = []
    for index, _doc_id, text in _docs(parquet, done):
        batch_ids.append(index)
        batch_text.append(text)
        if len(batch_text) == 64:
            _write_batch(emb, model, batch_ids, batch_text, progress_path)
            batch_ids, batch_text = [], []
    if batch_text:
        _write_batch(emb, model, batch_ids, batch_text, progress_path)
    emb.flush()
    del model
    gc.collect()
    _empty_cuda()


def _write_batch(emb, model, batch_ids, batch_text, progress_path: Path) -> None:
    vectors = model.encode(batch_text, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    for row, vector in zip(batch_ids, vectors):
        emb[row] = vector
    emb.flush()
    done = batch_ids[-1] + 1
    _save_text(progress_path, str(done))
    if done % 20480 < 64:
        print(f"  encoded {done}", flush=True)


def _bm25_top(parquet, n_docs: int, queries, query_ids):
    out = ROOT / f"{PREFIX}_bm25_top50.npy"
    if out.exists():
        print("BM25 top-50 already saved.", flush=True)
        return np.load(out)
    print("Building BM25 postings ...", flush=True)
    vocab = [line for line in (ROOT / f"{PREFIX}_vocab.txt").read_text(encoding="utf-8").split("\n") if line]
    token_id = {token: index for index, token in enumerate(vocab)}
    df = np.load(ROOT / f"{PREFIX}_df.npy")
    doc_len = np.load(ROOT / f"{PREFIX}_doclen.npy")
    avgdl = float(doc_len.mean()) if len(doc_len) else 1.0
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    np.cumsum(df, out=offsets[1:])
    postings_path = ROOT / f"{PREFIX}_postings.u32"
    posted_path = ROOT / f"{PREFIX}_postings.done"
    cursor_path = ROOT / f"{PREFIX}_postings_cursor.npy"
    shape = (int(offsets[-1]), 2)
    expected = int(offsets[-1]) * 2 * 4
    if postings_path.exists() and postings_path.stat().st_size == expected and posted_path.exists():
        posted = int(posted_path.read_text(encoding="utf-8"))
        cursor = np.load(cursor_path)
        postings = np.memmap(postings_path, dtype=np.uint32, mode="r+", shape=shape)
        print(f"Resuming BM25 postings at {posted}", flush=True)
    else:
        posted = 0
        cursor = offsets[:-1].copy()
        postings = np.memmap(postings_path, dtype=np.uint32, mode="w+", shape=shape)
    if posted < n_docs:
        for index, _doc_id, text in _docs(parquet, posted):
            counts = Counter(token for token in _tokens(text) if token in token_id)
            for token, freq in counts.items():
                slot = int(cursor[token_id[token]])
                postings[slot, 0] = index
                postings[slot, 1] = min(freq, 65535)
                cursor[token_id[token]] += 1
            if index and index % 50000 == 0:
                postings.flush()
                _save_npy(cursor_path, cursor)
                _save_text(posted_path, str(index + 1))
                print(f"  posted {index}", flush=True)
        postings.flush()
        _save_npy(cursor_path, cursor)
        _save_text(posted_path, str(n_docs))
    print("Scoring BM25 ...", flush=True)
    partial = out.with_suffix(".partial.npy")
    score_path = ROOT / f"{PREFIX}_bm25.done"
    if partial.exists() and score_path.exists():
        top = np.load(partial)
        scored = int(score_path.read_text(encoding="utf-8"))
        print(f"Resuming BM25 scoring at query {scored}", flush=True)
    else:
        top = np.full((len(query_ids), TOP), -1, dtype=np.int32)
        scored = 0
    values = np.zeros(n_docs, dtype=np.float32)
    seen = np.zeros(n_docs, dtype=np.int32)
    for query_number, query_id in enumerate(query_ids, start=1):
        if query_number <= scored:
            continue
        epoch = query_number
        touched = []
        for token in _tokens(queries[query_id]):
            term = token_id.get(token)
            if term is None:
                continue
            start, stop = int(offsets[term]), int(offsets[term + 1])
            weight = float(idf[term])
            for slot in range(start, stop):
                doc = int(postings[slot, 0])
                freq = int(postings[slot, 1])
                if seen[doc] != epoch:
                    seen[doc] = epoch
                    values[doc] = 0.0
                    touched.append(doc)
                denom = freq + 1.5 * (0.25 + 0.75 * doc_len[doc] / avgdl)
                values[doc] += weight * freq * 2.5 / denom
        ranked = sorted(touched, key=lambda doc: (-values[doc], doc))[:TOP]
        top[query_number - 1, : len(ranked)] = ranked
        if query_number % 50 == 0:
            print(f"  bm25 {query_number}/{len(query_ids)}", flush=True)
            _save_npy(partial, top)
            _save_text(score_path, str(query_number))
    _save_npy(out, top)
    return top


def _bge_top(queries, query_ids, n_docs: int):
    out = ROOT / f"{PREFIX}_bge_top50.npy"
    if out.exists():
        print("BGE top-50 already saved.", flush=True)
        return np.load(out)
    print("Scoring BGE ...", flush=True)
    import torch

    model = _model()
    query_vecs = model.encode(
        [QUERY_PROMPT + queries[query_id] for query_id in query_ids],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    del model
    gc.collect()
    _empty_cuda()
    emb = np.memmap(ROOT / f"{PREFIX}_emb.f32", dtype=np.float32, mode="r", shape=(n_docs, DIM))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    partial = out.with_suffix(".partial.npy")
    scores_path = ROOT / f"{PREFIX}_bge_scores.npy"
    block_path = ROOT / f"{PREFIX}_bge.done"
    if partial.exists() and scores_path.exists() and block_path.exists():
        top = np.load(partial)
        best_scores = torch.tensor(np.load(scores_path), dtype=torch.float32, device=device)
        resume = int(block_path.read_text(encoding="utf-8"))
        print(f"Resuming BGE search at passage {resume}", flush=True)
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
        print(f"  bge passages {stop}/{n_docs}", flush=True)
        _save_npy(partial, top)
        _save_npy(scores_path, best_scores.detach().cpu().numpy())
        _save_text(block_path, str(stop))
    _save_npy(out, top)
    return top


def _summarize(ids, relevant, query_ids, bm25_top, bge_top, n_docs: int):
    by_k = {}
    for k in KS:
        both = fixed = broken = bm25_hit = bge_hit = 0
        for row, query_id in enumerate(query_ids):
            gold = relevant[query_id]
            had = bool(gold & {str(ids[int(doc)]) for doc in bm25_top[row, :k] if doc >= 0})
            now = bool(gold & {str(ids[int(doc)]) for doc in bge_top[row, :k] if doc >= 0})
            bm25_hit += had
            bge_hit += now
            both += had and now
            fixed += (not had) and now
            broken += had and (not now)
        by_k[str(k)] = {
            "bm25_supported": bm25_hit,
            "bge_supported": bge_hit,
            "both_correct": both,
            "fixed": fixed,
            "broken": broken,
            "negative_flip_rate": round(broken / bm25_hit, 4) if bm25_hit else None,
            "net_gain": fixed - broken,
        }
    return {
        "dataset": "beir_hotpotqa_full",
        "corpus_size": n_docs,
        "questions": len(query_ids),
        "model": MODEL,
        "ks": list(KS),
        "stopwords_removed": True,
        "label": "judged_relevant_passage_in_top_k",
        "full_5233329_corpus": n_docs >= 5233329,
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


def _save_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)
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
        return _MODEL


def _empty_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


if __name__ == "__main__":
    main()
