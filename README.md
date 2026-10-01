# ragfix

Swapping BM25 for MiniLM on 150 Wikipedia questions fixed 13 missed answers and broke 33 that were already right.

```bash
pip install ragfix
ragfix check traces/ --retriever myapp.search:retrieve --k 8
```

The logged traces are the before. The retriever you pass is the after. Keep a change only when the failed claim improves and a previously supported claim does not get worse.

Package: https://pypi.org/project/ragfix/

`judge_supported` is that judge's label. It is not ground truth. On a 200-example slice, Llama, Qwen, and Laya all scored below the 0.635 majority baseline. See [`RESULTS.md`](RESULTS.md).

## Repair

Four changes only: `increase_k`, `restore_original_query`, `rerank_candidates`, `merge_retrieved`.

```bash
ragfix repair failure.json --retriever myproject.retrieval:search --judge qwen
```

`--judge` is `llama`, `qwen`, `openai`, `overlap`, `local`, or `module:function`. With no flag, OpenAI is used when `OPENAI_API_KEY` is set. Otherwise the lexical `overlap` judge runs, so the command works with no key and no GPU.

A run that should be kept looks like this:

```text
ragfix
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

## Retriever-swap measurement

This section is a separate study from the repair experiment at the bottom of this file. It does not use the sealed SQuAD split, the repair actions, or the judges.

On the full BEIR collections, replacing BM25 with BGE-large or E5-large-v2 raises the number of questions that have a judged relevant passage in the top k. It also drops some questions BM25 already had right. That drop is largest at k=1 and smaller at k=50. BM25 dropped a fixed stopword list so the large indexes would fit in memory.

Definitions, 95% intervals, and the reproduction path are in [`MEASUREMENT.md`](MEASUREMENT.md). A flip below is a question BM25 had right and the dense retriever lost. Net is newly fixed questions minus flips.

Natural Questions, 2,681,468 passages, 3,452 questions:

| k | BGE broken | BGE flip | BGE net | E5 broken | E5 flip | E5 net |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 194 | 37.5% | +708 | 148 | 28.6% | +959 |
| 5 | 149 | 12.0% | +1,157 | 105 | 8.5% | +1,406 |
| 10 | 97 | 6.1% | +1,177 | 63 | 3.9% | +1,359 |
| 20 | 89 | 4.6% | +1,056 | 47 | 2.4% | +1,184 |
| 50 | 55 | 2.3% | +864 | 37 | 1.6% | +916 |

HotpotQA, 5,233,329 passages, 7,405 questions:

| k | BGE broken | BGE flip | BGE net | E5 broken | E5 flip | E5 net |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 498 | 10.2% | +1,175 | 492 | 10.1% | +1,200 |
| 5 | 205 | 3.3% | +810 | 187 | 3.1% | +829 |
| 10 | 143 | 2.2% | +604 | 156 | 2.4% | +589 |
| 20 | 105 | 1.6% | +428 | 106 | 1.6% | +418 |
| 50 | 61 | 0.9% | +257 | 66 | 0.9% | +245 |

FiQA, 57,638 financial passages, 648 questions:

| k | BGE broken | BGE flip | BGE net | E5 broken | E5 flip | E5 net |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27 | 19.4% | +149 | 32 | 23.0% | +130 |
| 5 | 25 | 10.2% | +172 | 38 | 15.5% | +132 |
| 10 | 28 | 9.2% | +156 | 38 | 12.4% | +127 |
| 20 | 16 | 4.4% | +150 | 35 | 9.6% | +112 |
| 50 | 10 | 2.4% | +142 | 24 | 5.8% | +116 |

At a fixed budget of 10 passages, alternating the two rankings, or fusing them with reciprocal rank fusion (constant 60), cuts the breaks. On Natural Questions the mix gives up some of the net gain. On HotpotQA the mix keeps the gain.

| Corpus and model | Dense broken | Dense net | Alternate broken | Alternate net | Fusion broken | Fusion net |
|---|---:|---:|---:|---:|---:|---:|
| NQ, BGE | 97 | +1,177 | 48 | +1,024 | 23 | +873 |
| NQ, E5 | 63 | +1,359 | 32 | +1,205 | 12 | +997 |
| HotpotQA, BGE | 143 | +604 | 40 | +669 | 24 | +677 |
| HotpotQA, E5 | 156 | +589 | 37 | +671 | 31 | +645 |
| FiQA, BGE | 28 | +156 | 17 | +143 | 6 | +137 |
| FiQA, E5 | 38 | +127 | 20 | +115 | 15 | +120 |

Keeping both full lists removes the breaks and can use up to 20 passages.

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

## Earlier repair study

This is not the retriever-swap measurement above. The numbers here come from the frozen repair experiment in [`STUDY.md`](STUDY.md) and [`RESULTS.md`](RESULTS.md). Do not mix them with the flip tables.

The sealed holdout and the mixed Hotpot comparison are frozen. Diagnosis did not beat blind increase-k: 66/76 versus 67/76, and 11/35 versus 12/35. On 12 controls, blind increase-k regressed 4 working answers. The guarded path regressed 0. Do not retune those sets.

On the SQuAD development half, selective repair also stays behind increase-k at the 5% regression budget (overlap 354 vs 382, Llama 333 vs 382, Qwen 274 vs 382, Mistral 247 vs 382). The guard matched no-guard on all four. The final test is not scored. Details are in [`RESULTS.md`](RESULTS.md).

`ragfix` with no subcommand prints help. `ragfix serve` is the older demo server.
