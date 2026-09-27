# Current results

**Frozen.** Do not retune the repair policy, prompts, or thresholds against these numbers. The sealed holdout and the mixed Hotpot comparison stay as written below.

## Decision

Diagnosis does not beat always-increase-k. That is closed for two independent sets: sealed holdout 66/76 vs 67/76, mixed pipelines 11/35 vs 12/35.

The product direction is no longer "we diagnose RAG failures better." It is: try a retrieval change on the user's retriever, rerun, and **reject** the change if a previously supported claim gets worse.

The result that motivates that:

| Control set (12 budget / supported cases) | Regressions |
|---|---|
| Always increase k | 4 |
| Guarded RAG Debugger path | 0 |

The SQuAD development study is scored below. The final test is not. Do not use any of these numbers to change the accept rule.

Judge for the development benchmark and the repair runs below: `meta-llama/Llama-3.1-8B-Instruct`. Embeddings: `BAAI/bge-large-en-v1.5`.

## Development benchmark

25 RAGTruth QA articles, 100 planted cases. The same judge rejected 39 cases as ambiguous before scoring. 61 cases were scored.

| Metric | Result |
|---|---|
| Accuracy | 0.951 (58/61) |
| Macro F1 | 0.877 |
| Repair attempted | 14 |
| Repair improved | 12 |
| End-to-end | 0.786 (11/14) |

Per class on the 61 scored cases:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| supported | 22 | 1.000 | 1.000 | 1.000 |
| hallucination | 25 | 1.000 | 1.000 | 1.000 |
| chunking_miss | 4 | 0.600 | 0.750 | 0.667 |
| retrieval_miss | 10 | 0.889 | 0.800 | 0.842 |

The 0.951 accuracy is only on cases the same judge already accepted. The repair number to keep is 11/14. Three cases missed:

- `15444-retrieval_miss` was called chunking. The rerun did not verify.
- `14418-retrieval_miss` was called chunking. The rerun became supported, but the stage was wrong, so it is not in the 11.
- `15512-chunking_miss` was called retrieval. The rerun stayed a retrieval miss.

Development article IDs: 15215, 15243, 12397, 15444, 15145, 15331, 14418, 14429, 15512, 15222, 15374, 15148, 12161, 12192, 15551, 15419, 12274, 15556, 12185, 15458, 14404, 15275, 15285, 15105, 12278.

## Command test

20 hand-written traces in `examples/traces`, judged by Llama 3.1 8B, retrieved by `examples.my_retriever`. All 20 matched `examples/traces/expected.json`.

| Outcome | Count |
|---|---|
| `rerun_retriever`, Verified True | 8 |
| `increase_k` at k=6, Verified True | 4 |
| hallucination, retriever not called, Verified False | 4 |
| chunking, retriever not called, Verified False | 2 |
| already supported, no experiment, Verified False | 2 |

This checks the repair command. It is not a validation set.

## Single real articles

Both are RAGTruth passages with a planted retrieval miss and the dense retriever. They are outside the 25 development articles and the sealed 100.

| Article | Result |
|---|---|
| 15135 | Verified False. The debugger called chunking and did not call the retriever. The claim was several sentences glued together. |
| 15303 | Verified True. `rerun_retriever` at k=4. The claim became supported. The saved claim still starts with a stray `2` from the source text. |

## External retriever

One TF-IDF retriever over six documents, `examples/tfidf_retriever.py`. At k=1 it returned `catalog_notice.txt`. The supporting document, `special_collections.txt`, was rank 2. RAG Debugger called that retriever again at k=6.

Result: `increase_k` at k=6, 1/1 failed claims improved, 0 regressions, Verified True. The printed family line is `retrieval_miss -> supported`. `increase_k` is the `k_too_small` experiment.

## Sealed holdout

100 article IDs in `rag/datasets/holdout_ids.json`, scored once with Llama 3.1 8B. Two IDs did not build (`15220`, `12307`). The other 98 produced 392 planted cases. The same judge rejected 126 as ambiguous. 266 cases were scored.

| Metric | Result |
|---|---|
| Accuracy | 0.951 (253/266) |
| Family accuracy | 0.955 (254/266) |
| Macro F1 | 0.843 |
| Repair attempted | 77 |
| Repair improved | 69 |
| End-to-end | 0.882 (67/76) |

Per class on the 266 scored cases:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| supported | 93 | 1.000 | 1.000 | 1.000 |
| hallucination | 97 | 0.979 | 0.969 | 0.974 |
| chunking_miss | 9 | 0.500 | 0.444 | 0.471 |
| retrieval_miss | 67 | 0.913 | 0.940 | 0.926 |

The 0.951 accuracy is only on cases the same judge already accepted. Most planted chunking cases were rejected, so chunking support is 9. When the stage family was right, the rerun made the claim supported: the 9 misses in the 67/76 are diagnosis misses, not failed reruns after a correct family. Thirteen exact misses:

- Chunking called retrieval, rerun stayed a retrieval miss: `15532`, `14344`, `12345`.
- Chunking called hallucination, no rerun: `12285`, `15350`.
- Retrieval called chunking, rerun did not verify: `12290`, `14392`.
- Retrieval called chunking, rerun became supported, but the stage was wrong: `12374`, `15494`.
- Retrieval called `k_too_small`, family correct, rerun verified: `15540`.
- Hallucination called retrieval, rerun returned to hallucination: `12258`, `15310`, `12443`.

This is a planted-fault holdout with the same judge used as a filter.

## Policy comparison

The 76 scored repairable holdout cases were given to four policies. Each policy could call the same dense retriever. The planted label was not shown to them. RECTIFY was not run.

| Policy | All 76 | Retrieval only (67) | Chunking only (9) |
|---|---|---|---|
| RAG Debugger | 0.868 (66/76) | 0.925 (62/67) | 0.444 (4/9) |
| Always increase k | 0.882 (67/76) | 1.000 (67/67) | 0.000 (0/9) |
| Always rerun at the same k | 0.868 (66/76) | 0.985 (66/67) | 0.000 (0/9) |
| Llama suggestion, then that repair | 0.882 (67/76) | 1.000 (67/67) | 0.000 (0/9) |

After `chunking_miss` runs `merge_retrieved`, RAG Debugger is the only policy that repairs chunking: 4/9 (`15107`, `14297`, `15526`, `12167`). The other five chunking cases were diagnosed as retrieval or hallucination, so the merge never ran.

Retrieval is unchanged. Always-increase-k still repairs all 67. RAG Debugger misses five: `12290`, `14392`, `12374`, and `15494` were called chunking and the merge did not verify; `15540` was rerun at k=4, and only the larger k verified. The suggestion policy still chose `increase_k`, so it ties that baseline. Overall, RAG Debugger is 66/76, one case behind always-increase-k.

The holdout is frozen. On this planted set the product does not outperform always-increase-k. The result to keep is that RAG Debugger repaired 4 split-chunk failures that every retrieval-only baseline missed. The next development set is `examples/traces/hetero`. It is not scored against this holdout.

## RAGTruth supported vs hallucination

Side check only. The full RAGTruth response test set is 2700 examples: 1757 supported and 943 hallucination. Always answering "supported" is 0.651 accuracy. This does not measure repair, and it is not mixed into the holdout checkpoint.

| Judge | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Llama 3.1 8B | 0.443 | 0.836 | 0.579 | 0.576 |
| Qwen3.8 27B | 0.506 | 0.951 | 0.661 | 0.659 |

Qwen counts: tp 897, fp 876, fn 46. It finds almost every hallucination and also marks 876 supported answers as hallucinations, so accuracy is 0.8 points above always-supported. By task, Data2txt F1 is 0.821. Summary F1 is 0.521 and QA F1 is 0.468. Checkpoint: `slice4000_local_partial.jsonl`.

## Heterogeneous repair check

Seven scripted traces in `examples/traces/hetero`, judged once by Llama 3.1 8B, retrieved by `examples.hetero_retriever`. Log: `hetero_llama.log`. This set was written so each repair is the one that can succeed. It is not a sealed comparison, and it does not replace the 66/76 holdout result.

| Trace | Repair | Verified |
|---|---|---|
| `wider_k_works` | wider retrieval; `retrieval_miss` became supported | True |
| `hallucination` | `no_retrieval_change` | False |
| `wider_k_regresses` | `no_retrieval_change`; the supported lamp claim stayed supported | False |
| `merge_works` | `merge_retrieved`; `chunking_miss` became supported | True |
| `query_rewrite` | `restore_original_query` | True |
| `rerank` | `rerank_candidates` | True |
| `supported` | no experiment | False |

## Mixed pipeline set

Scored once with Llama 3.1 8B. Log: `mixed_llama.jsonl`. Built from the first 160 HotpotQA validation examples and five retrieval pipelines: BM25, BM25 after a keyword rewrite, BM25 with a logged TF-IDF score, 16-word windows over that question's own paragraphs, and a BM25 packer with a fixed 90-word budget. A case was kept from the retriever's rank or split. The repair policy was not used to choose it. The sealed holdout was not used. RECTIFY was not run.

35 failures and 12 controls.

| Policy | Verified repairs (35 failures) | Control regressions (12) |
|---|---|---|
| RAG Debugger | 0.314 (11/35) | 0 |
| Always increase k | 0.343 (12/35) | 4 |
| Always rerun at the same k | 0.000 (0/35) | 0 |
| Llama suggestion, then that repair | 0.200 (7/35) | 0 |

RAG Debugger is one failure behind always-increase-k. It beats the Llama suggestion and the same-k rerun. Nine of the 35 were already supported under Llama, so neither policy could verify them. On the other 26, the score is 11 versus 12.

Always-increase-k verified 3 failures the debugger missed (`rerank_logged-4`, `wider_k_reaches-6`, `wider_k_reaches-7`). The debugger verified 2 that always-increase-k missed, both keyword-rewrite traces: `query_rewrite-1` restored the original query, and `query_rewrite-3` merged the retrieved chunks. It did not choose merge on any of the 5 window splits. The 4 control regressions are all always-increase-k on the budget packer, where a larger k dropped the supporting paragraph. The debugger left those claims alone.

## Judge-dev (verifier only)

Same 200 RAGTruth response IDs. Frozen repair numbers are unchanged. None of these beat the majority baseline, so none is ready as a trusted `Verified` signal.

| Judge | Accuracy | Majority | Hall. P | Hall. R | Hall. F1 |
|---|---|---|---|---|---|
| Laya | 0.400 | 0.635 | 0.376 | 0.973 | 0.542 |
| Llama 3.1 8B | 0.535 | 0.635 | 0.423 | 0.753 | 0.542 |
| Qwen 27B | 0.585 | 0.635 | 0.467 | 0.973 | 0.631 |

Qwen is the least bad on this slice. Laya is worst on accuracy and over-calls hallucination. Do not run Laya on the full 2700. Do not make Laya the default verifier. Keep the product as the guarded repair loop; treat every `Verified` as provisional until a judge clears the majority baseline.

## Selective repair, SQuAD development

Development half only. Protocol: [`STUDY.md`](STUDY.md). Source is SQuAD v1.1 dev, not RAGTruth and not the mixed Hotpot set. 5,045 development questions: 1,073 failures and 3,972 controls. The reported label is whether the gold answer string is in the served text. The internal judge only chooses which repair to keep. The final test (5,097 ids) is not scored. Gemma 2 9B, DeepSeek-R1-Distill-Llama 8B, and Phi-3.5-mini were not run.

Increase-k, rerun, and the oracle do not depend on the internal judge, so they are the same on every row: always increase-k repairs 382/1073 and regresses 95/3972 (2.4%); always rerun repairs 0 and regresses 0; the oracle repairs 382/1073 and regresses 0. Increase-k is inside the 5% regression budget (382) and is ineligible at 0%, 1%, and 2%.

| Internal judge | One-shot repairs | Selective repairs | Selective control regressions | Selective at 0% | Selective at 5% | Guard vs no-guard |
|---|---:|---:|---:|---:|---:|---|
| Overlap | 299/1073 | 354/1073 | 0/3972 | 354 | 354 | same (354) |
| Llama 3.1 8B | 270/1073 | 333/1073 | 1/3972 | none | 333 | same (333) |
| Qwen2.5 7B | 234/1073 | 274/1073 | 0/3972 | 274 | 274 | same (274) |
| Mistral 7B Instruct v0.3 | 178/1073 | 247/1073 | 0/3972 | 247 | 247 | same (247) |

Selective retriever calls: overlap 2,371, Llama 2,913, Qwen 3,955, Mistral 5,079. The oracle uses 10,857. One-shot calls: overlap 830, Llama 765, Qwen 684, Mistral 546.

On the pre-registered 5% budget, increase-k still ranks first on every judge. Selective stays legal at a 0% cutoff except under Llama, where the single regression makes it ineligible. The regression guard matched the no-guard ablation on all four judges, so the measured effect is abstention, not the guard. Do not edit `select_repair` from this table.

182 development controls contain the answer string inside a longer word. Nobody labeled them. They are all controls. Four of the 95 increase-k regressions are in that set. None of the selective regressions are.
