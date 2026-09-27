# RAG Debugger

**Alpha.** Test a RAG fix against your real retriever before you ship it.

```bash
pip install ragfix
ragfix analyze trace.json
ragfix repair trace.json --retriever myapp.search:retrieve
ragfix check traces/ --retriever myapp.search:retrieve --k 8
```

Package: https://pypi.org/project/ragfix/

The tool loads one failed trace, picks one small retrieval change, calls your retriever, and keeps the change only when the failed claim becomes supported and no previously supported claim gets worse.

`judge_supported` is that judge's label. It is not ground truth. On a 200-example slice, Llama, Qwen, and Laya all scored below the 0.635 majority baseline. See [`RESULTS.md`](RESULTS.md).

## Repair

Four changes only: `increase_k`, `restore_original_query`, `rerank_candidates`, `merge_retrieved`.

```bash
ragfix repair failure.json --retriever myproject.retrieval:search --judge qwen
```

`--judge` is `llama`, `qwen`, `openai`, `overlap`, `local`, or `module:function`. With no flag, OpenAI is used when `OPENAI_API_KEY` is set. Otherwise the lexical `overlap` judge runs, so the command works with no key and no GPU.

A run that should be kept looks like this:

```text
RAG Debugger
Judge: overlap

Failed claim
"Paris is the capital of France."

Diagnosis
Likely retrieval failure

Experiment
increase_k: 1 → 6

Before
unsupported

After
supported

Regression check
0 previously supported claims checked
0 regressions

judge_supported=true

Recommendation
ACCEPT EXPERIMENT
```

A run that should be thrown away looks like this. The failed claim may improve, and a claim that was already supported gets worse:

```text
Recommendation
REJECT EXPERIMENT
```

Try that case:

```bash
ragfix repair examples/traces/accept_increase_k.json \
  --retriever examples.demo_retriever:retrieve --judge overlap

ragfix repair examples/traces/reject_increase_k.json \
  --retriever examples.demo_retriever:retrieve --judge overlap
```

The reject trace already supports “The desk lamp uses 40 watts.” Wider k returns Paris and drops the lamp, so the recommendation is `REJECT EXPERIMENT`.

## Check a change across many traces

Before you change `k`, the chunk size, the embedding model, or the reranker, run the new retriever over your logged traces:

```bash
ragfix check logged_traces/ --retriever myapp.search:retrieve_v2 --judge overlap
ragfix check logged.jsonl --retriever myapp.search:retrieve --k 8 --out report.json
```

```text
Traces checked: 26
Fixed:     17 / 19 failing traces
Broken:    4 / 7 working traces
Mixed:     1 traces fixed one claim and broke another
Unchanged: 4

Recommendation
REVIEW BEFORE SHIPPING
```

The logged `retrieved_chunks` are the "before". The retriever you pass is the "after". Each broken trace is listed by name with the claim that lost its support. `--k` overrides every trace's logged `top_k`.

## Wikipedia example

`python -m examples.wiki_demo` logs BM25 top-1 over 2,067 SQuAD Wikipedia passages (150 questions), then checks two changes. The gold line is whether the dataset answer string is in the retrieved text. It is not the sealed study.

Replacing BM25 with `all-MiniLM-L6-v2` at the same k:

| | Fixed | Broken |
|---|---:|---:|
| Gold answer string | 13 / 33 missed | 33 / 117 already answered |
| Overlap judge | 12 / 31 failing traces | 31 / 119 working traces |

Packing BM25 top-6 into a 90-word budget fixed 6 and broke 93 of the 117 answers BM25 already had. The tool's recommendation on both changes is `REVIEW BEFORE SHIPPING`.

## Your retriever

```python
def retrieve(query: str, k: int):
    return [{"id": "1", "text": "..."}]
```

LangChain:

```python
from ragfix.integrations.langchain import as_retriever, trace

failure = trace("What does the desk lamp use?", chain_result, top_k=1)
failure.save("failure.json")
# ragfix repair failure.json --retriever myapp:retrieve
# or, in Python, pass as_retriever(vectorstore.as_retriever()) to run_verified_repair
```

`as_retriever` sets `k` and returns `{id, text}` chunks.

## Trace

```json
{
  "question": "...",
  "answer": "...",
  "retrieved_chunks": [{"id": "r1", "text": "..."}],
  "corpus_chunks": [{"id": "c47", "text": "..."}],
  "metadata": {"retriever": "faiss", "top_k": 4}
}
```

## Frozen results

The sealed holdout and the mixed Hotpot comparison are frozen. Diagnosis did not beat blind increase-k: 66/76 versus 67/76, and 11/35 versus 12/35. On 12 controls, blind increase-k regressed 4 working answers. The guarded path regressed 0. Do not retune those sets.

On the SQuAD development half, selective repair also stays behind increase-k at the 5% regression budget (overlap 354 vs 382, Llama 333 vs 382, Qwen 274 vs 382, Mistral 247 vs 382). The guard matched no-guard on all four. The final test is not scored. Details are in [`RESULTS.md`](RESULTS.md).

`ragfix` with no subcommand prints help. `ragfix serve` is the older demo server.
