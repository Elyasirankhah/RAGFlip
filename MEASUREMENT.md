# Retriever-swap measurement

This study asks what happens to individual questions when BM25 is replaced by BGE-large or E5-large-v2. It is not the repair study in [`STUDY.md`](STUDY.md). That study scores repair actions on a sealed SQuAD split. This one does not. Do not retune the repair rule from these tables, and do not score the sealed test.

A question is supported when at least one judged relevant passage is inside the top k. BM25 removed a fixed stopword list. The same list is used on every corpus.

## Metrics

For one question, write \(B\) when BM25 is supported and \(D\) when the dense retriever is supported.

| Name | Definition |
|---|---|
| Fixed | \(D\) and not \(B\) |
| Broken | \(B\) and not \(D\) |
| Fix rate | Fixed questions divided by questions BM25 missed |
| Negative-flip rate | Broken questions divided by questions BM25 got right |
| Net gain | Fixed minus broken, over all questions |
| Compatibility | Questions both retrievers got right, divided by questions BM25 got right. This is 1 minus the negative-flip rate |

Rates use a 95% Wilson interval. Net gain scores each question +1 if fixed, −1 if broken, and 0 otherwise, and uses a 95% Wald interval on the sum. Questions are treated as independent. Recompute every row with:

```bash
python -m rag.flip_metrics
```

The definitions are locked by `tests/test_flip_metrics.py`.

## Intervals

Natural Questions, 3,452 questions. Flip rate and net gain:

| k | BGE flip | E5 flip | BGE net | E5 net |
|---:|---|---|---|---|
| 1 | 37.5% [33.5, 41.8] | 28.6% [24.9, 32.7] | +708 [648, 768] | +959 [897, 1021] |
| 5 | 12.0% [10.3, 13.9] | 8.5% [7.0, 10.2] | +1,157 [1,093, 1,221] | +1,406 [1,343, 1,469] |
| 10 | 6.1% [5.0, 7.3] | 3.9% [3.1, 5.0] | +1,177 [1,116, 1,238] | +1,359 [1,299, 1,419] |
| 20 | 4.6% [3.7, 5.6] | 2.4% [1.8, 3.2] | +1,056 [997, 1,115] | +1,184 [1,126, 1,242] |
| 50 | 2.3% [1.8, 3.0] | 1.6% [1.1, 2.2] | +864 [810, 918] | +916 [862, 970] |

HotpotQA, 7,405 questions:

| k | BGE flip | E5 flip | BGE net | E5 net |
|---:|---|---|---|---|
| 1 | 10.2% [9.4, 11.1] | 10.1% [9.3, 11.0] | +1,175 [1,088, 1,262] | +1,200 [1,113, 1,287] |
| 5 | 3.3% [2.9, 3.8] | 3.1% [2.7, 3.5] | +810 [744, 876] | +829 [764, 894] |
| 10 | 2.2% [1.9, 2.6] | 2.4% [2.1, 2.8] | +604 [547, 661] | +589 [532, 646] |
| 20 | 1.6% [1.3, 1.9] | 1.6% [1.3, 1.9] | +428 [380, 477] | +418 [370, 466] |
| 50 | 0.9% [0.7, 1.1] | 0.9% [0.7, 1.2] | +257 [219, 295] | +245 [207, 283] |

FiQA, 648 questions:

| k | BGE flip | E5 flip | BGE net | E5 net |
|---:|---|---|---|---|
| 1 | 19.4% [13.7, 26.8] | 23.0% [16.8, 30.7] | +149 [124, 175] | +130 [105, 155] |
| 5 | 10.2% [7.0, 14.6] | 15.5% [11.5, 20.5] | +172 [146, 198] | +132 [106, 158] |
| 10 | 9.2% [6.4, 12.9] | 12.4% [9.2, 16.6] | +156 [130, 182] | +127 [101, 153] |
| 20 | 4.4% [2.7, 7.0] | 9.6% [7.0, 13.0] | +150 [126, 174] | +112 [87, 137] |
| 50 | 2.4% [1.3, 4.4] | 5.8% [3.9, 8.5] | +142 [120, 164] | +116 [93, 140] |

At k=10 the other two rates are:

| Corpus and model | Fix rate | Compatibility |
|---|---|---|
| NQ, BGE | 68.8% [66.6, 70.9] | 93.9% [92.7, 95.0] |
| NQ, E5 | 76.8% [74.8, 78.7] | 96.1% [95.0, 96.9] |
| HotpotQA, BGE | 81.1% [78.5, 83.5] | 97.8% [97.4, 98.1] |
| HotpotQA, E5 | 80.9% [78.2, 83.3] | 97.6% [97.2, 97.9] |
| FiQA, BGE | 53.8% [48.5, 59.0] | 90.8% [87.1, 93.6] |
| FiQA, E5 | 48.2% [43.0, 53.5] | 87.6% [83.4, 90.8] |

On HotpotQA the BGE and E5 flip intervals overlap at every depth. On Natural Questions they are separated at k=1 and k=10. On FiQA the intervals are wide because only 648 questions are scored.

## Where the flips sit

At k=10, the flip rate by the BM25 rank of the first relevant passage:

| Corpus and model | Rank 1 | Ranks 2–5 | Ranks 6–10 |
|---|---|---|---|
| NQ, BGE | 4.6% [3.1, 6.8] | 7.2% [5.5, 9.3] | 5.8% [3.9, 8.8] |
| NQ, E5 | 3.7% [2.4, 5.7] | 3.9% [2.7, 5.5] | 4.4% [2.8, 7.1] |
| HotpotQA, BGE | 1.5% [1.2, 1.9] | 3.4% [2.5, 4.5] | 7.6% [5.3, 10.9] |
| HotpotQA, E5 | 1.6% [1.3, 2.0] | 4.0% [3.0, 5.2] | 7.6% [5.3, 10.9] |
| FiQA, BGE | 3.6% [1.6, 8.1] | 10.3% [5.8, 17.5] | 20.0% [11.8, 31.8] |
| FiQA, E5 | 4.3% [2.0, 9.1] | 15.9% [10.2, 24.0] | 25.0% [15.8, 37.2] |

Rank 1 is the lowest flip rate in every row. On HotpotQA and FiQA the rate rises as BM25's relevant passage sits deeper, and the rank-1 interval does not overlap the rank 6–10 interval. On Natural Questions the rise is small and the intervals overlap.

Paired McNemar test on questions BM25 already got right. "Only BGE" means BGE flipped that question and E5 kept it.

| Corpus | k=10 only BGE | k=10 only E5 | k=10 p | Reading |
|---|---:|---:|---:|---|
| Natural Questions | 60 | 26 | 0.0004 | E5 keeps more BM25 successes at every depth, p < 0.05 from k=1 through k=50 |
| HotpotQA | 64 | 77 | 0.31 | No detectable difference at any depth. The smallest p is 0.22, at k=5 |
| FiQA | 10 | 20 | 0.10 | E5 flips more. The paired test is significant at k=5, k=20, and k=50, and not at k=1 or k=10 |

Query-length quartiles and the dense score gap are in `negative_flips/flip_analysis.json`. The script that wrote them is `examples/flip_analysis.py`. It reads the saved top-50 lists and does not encode again.

## Reproduction

| Script | What it writes |
|---|---|
| `examples/nq_flips.py` | BM25 and BGE on full Natural Questions, then the k sweep |
| `examples/hotpot_full_flips.py` | The same on full HotpotQA |
| `examples/e5_flips.py` | E5 on both corpora, reusing the saved BM25 lists |
| `examples/fiqa_flips.py` | BM25, BGE, and E5 on FiQA |
| `examples/merge_topk.py` | Same-budget alternate mix and the union of both lists |
| `examples/hybrid_k.py` | Reciprocal rank fusion and the other fixed-budget rules |
| `examples/flip_analysis.py` | Intervals, paired tests, and the rank and length slices |

Corpora and qrels come from the Hugging Face BEIR parquet releases named in those scripts: `BeIR/nq`, `BeIR/hotpotqa`, `BeIR/fiqa`, and the matching `*-qrels` test files. Dense models are `BAAI/bge-large-en-v1.5` and `intfloat/e5-large-v2`. E5 uses the prefixes `query: ` and `passage: `. Depths are 1, 5, 10, 20, and 50. A killed job continues from the last saved batch. Leave the `nq_`, `hotpot_`, and `fiqa_` files in place.
