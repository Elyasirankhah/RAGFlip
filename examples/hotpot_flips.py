from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

URLS = (
    "https://hotpotqa.github.io/data/hotpot_dev_distractor_v1.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
)
MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
K = 2
MIN_ANSWER_TOKENS = 3
TOKEN = re.compile(r"[a-z0-9]+")
SKIP_ANSWERS = {"yes", "no"}


def _download() -> bytes:
    last_error = None
    for url in URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ragflip"})
            return urllib.request.urlopen(request, timeout=180).read()
        except Exception as exc:
            last_error = exc
            print(f"Download failed: {url}", flush=True)
    raise SystemExit(f"Could not download HotpotQA: {last_error}")


def main() -> None:
    root = Path("/nfs/roberts/scratch/pi_sjf37/ei235/ragdbg/negative_flips")
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "hotpot_dev_distractor_v1.json"
    if not raw_path.exists():
        print("Downloading HotpotQA distractor dev ...", flush=True)
        raw_path.write_bytes(_download())
    rows = [row for row in json.loads(raw_path.read_text(encoding="utf-8")) if _keep(row)]
    print(f"Questions: {len(rows)}", flush=True)

    before_hit = after_hit = 0
    for index, row in enumerate(rows, start=1):
        if index % 200 == 0 or index == 1:
            print(f"[{index}/{len(rows)}]", flush=True)
        paragraphs = _paragraphs(row)
        before = "\n".join(text for _, text in _bm25(row["question"], paragraphs, K))
        after = "\n".join(text for _, text in _bge(row["question"], paragraphs, K))
        had = _supported(row["answer"], before)
        now = _supported(row["answer"], after)
        before_hit += had
        after_hit += now
        row["_had"] = had
        row["_now"] = now
    fixed = sum(1 for row in rows if not row["_had"] and row["_now"])
    broken = sum(1 for row in rows if row["_had"] and not row["_now"])
    missed = len(rows) - before_hit
    summary = {
        "dataset": "hotpot_dev_distractor_v1",
        "model": MODEL,
        "k": K,
        "questions": len(rows),
        "bm25_supported": before_hit,
        "bge_supported": after_hit,
        "fixed": fixed,
        "missed_by_bm25": missed,
        "broken": broken,
        "working_under_bm25": before_hit,
        "label": "answer_string_in_retrieved_text",
    }
    out = root / "hotpot_bge_k2.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


def _keep(row: dict) -> bool:
    answer = " ".join(str(row.get("answer") or "").casefold().split())
    return bool(answer) and answer not in SKIP_ANSWERS and len(TOKEN.findall(answer)) >= MIN_ANSWER_TOKENS


def _paragraphs(row: dict):
    paragraphs = []
    for title, sentences in row.get("context") or []:
        text = " ".join(str(sentence) for sentence in sentences)
        if text.strip():
            paragraphs.append((str(title), text))
    return paragraphs


def _supported(answer: str, text: str) -> bool:
    needle = " ".join(answer.casefold().split())
    haystack = " ".join(text.casefold().split())
    return bool(needle) and needle in haystack


def _tokens(text: str):
    return TOKEN.findall(text.lower())


def _bm25(query: str, paragraphs, k: int):
    docs = [_tokens(text) for _, text in paragraphs]
    avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
    df = Counter(token for doc in docs for token in set(doc))
    count = len(docs)
    idf = {token: math.log(1.0 + (count - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}
    query_tokens = _tokens(query)
    scores = []
    for index, doc in enumerate(docs):
        counts = Counter(doc)
        score = 0.0
        for token in query_tokens:
            freq = counts.get(token)
            if not freq:
                continue
            denom = freq + 1.5 * (0.25 + 0.75 * len(doc) / avgdl)
            score += idf.get(token, 0.0) * freq * 2.5 / denom
        scores.append(score)
    order = sorted(range(len(paragraphs)), key=lambda index: (-scores[index], index))
    return [paragraphs[index] for index in order[:k]]


_VECTORS = {}


def _bge(query: str, paragraphs, k: int):
    model = _model()
    keys = []
    missing = []
    for title, text in paragraphs:
        key = title + "\n" + text
        keys.append(key)
        if key not in _VECTORS:
            missing.append(text)
    if missing:
        encoded = model.encode(missing, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        cursor = 0
        for key in keys:
            if key not in _VECTORS:
                _VECTORS[key] = np.asarray(encoded[cursor], dtype=np.float32)
                cursor += 1
    matrix = np.vstack([_VECTORS[key] for key in keys])
    query_vec = model.encode([QUERY_PROMPT + query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = matrix @ np.asarray(query_vec, dtype=np.float32)
    order = np.argsort(-scores)[:k]
    return [paragraphs[int(index)] for index in order]


def _model():
    global _MODEL
    try:
        return _MODEL
    except NameError:
        from sentence_transformers import SentenceTransformer

        print(f"Loading {MODEL} ...", flush=True)
        _MODEL = SentenceTransformer(MODEL)
        return _MODEL


if __name__ == "__main__":
    main()
