# Selective repair study

This file is the earlier repair experiment. It is not the retriever-swap measurement in [`MEASUREMENT.md`](MEASUREMENT.md). Do not score the sealed test, and do not retune this study from the flip tables.

**Preregistered. Development is scored with overlap, Llama 3.1 8B, Qwen2.5 7B, and Mistral 7B Instruct v0.3. The final test is not scored.**

## Question

Can selective guarded repair reduce regression risk while maintaining competitive repair success compared with unconditional RAG repair policies?

## Method

`select_repair` generates every legal action, runs each one, and records target improvement, regressions, retriever calls, and latency.

Actions: `merge_retrieved`, `rerank_candidates`, `restore_original_query`, `rerun_retriever`, `increase_k`, `abstain`.

The full rule applies a change only when every failed claim becomes supported and no previously supported claim regresses. If several repairs pass, it keeps the one with the fewest retriever calls. If none pass, it abstains. A failed claim labeled `unknown` abstains with no retriever call. Latency is recorded and is not part of the choice.

Ablations, also frozen: `no_guard` may apply a repair that regresses; `no_abstention` must apply a repair; `no_selector` is the old one-shot `choose_experiment`; disabling restore, merge, or rerank drops that action.

The internal judge is not ground truth.

## What stays frozen

The sealed holdout, the mixed Hotpot set, and judge-dev are preliminary evidence only. `assert_not_frozen` refuses those names. Do not tune this rule on them.

## Benchmark

Sealed and not scored. Source is SQuAD v1.1 dev, not RAGTruth and not the mixed Hotpot set. Files are under `rag/datasets/study/`.

10,142 questions. 2,158 failures and 7,984 controls, labeled by whether the gold answer string is in the retrieved text. Development has 5,045 ids and the final test has 5,097. The split is `SPLIT_SEED` 11. `method_scored` is false.

Environments, one per question: BM25, hashed-vector dense, hybrid, logged rerank, query rewrite, 40-word windows, and a 90-word context budget. Each question is searched over its gold paragraph plus 15 other paragraphs. BM25, hybrid, rewrite, and rerank miss rarely at k=1. Dense and the context budget produce most of the failures. Do not change those operating points after this seal.

The development split was scored once on this machine. The final test was not scored. The internal judge was lexical overlap. The reported label was the SQuAD answer string.

| Policy | Repair | Control regression | Failures abstained | Repairs under a 5% regression budget |
|---|---:|---:|---:|---:|
| Always increase k | 382/1073 (35.6%) | 95/3972 (2.4%) | 0% | 382 |
| Always rerun | 0/1073 | 0/3972 | 0% | 0 |
| One-shot diagnosis | 299/1073 (27.9%) | 0/3972 | 20.9% | 299 |
| Selective, no guard | 354/1073 (33.0%) | 0/3972 | 65.2% | 354 |
| Selective | 354/1073 (33.0%) | 0/3972 | 65.2% | 354 |
| Oracle, gold label | 382/1073 (35.6%) | 0/3972 | 64.4% | 382 |

On the pre-registered 5% budget, increase-k still ranks first because 2.4% is inside the budget. On the secondary curve it is ineligible at 0%, 1%, and 2%, where selective still counts 354 repairs. Selective is close to the oracle and regresses no controls. With this overlap judge, the guard and the no-guard ablation made the same choices.

The same development split was then scored with three internal judges. Increase-k, rerun, and the oracle stay 382, 0, and 382. Selective repairs were Llama 333 (1 control regression), Qwen 274 (0), and Mistral 247 (0). The guard matched no-guard on each of them. Full counts are in RESULTS.md. Gemma, DeepSeek, and Phi were not run. Do not edit the rule from these tables, and do not score the final test.

## Labels

Dataset answer-string containment where the dataset provides it. Two-person human labels where containment is ambiguous. Model judges are ablations and the policy's internal signal, not the headline score.

## Comparison

Always increase-k, always rerun, an LLM that picks a repair, an LLM that proposes a configuration, one-shot diagnosis, selective repair without the guard, full selective repair, and an oracle that sees the gold label. RECTIFY only if it calls this retriever hook.

Primary number: how many failures are repaired while control regressions stay at or below 5%. That cutoff stays 5%. The secondary curve reports the same repairs at 0%, 1%, 2%, and 5%. Also report repair success, regression rate, abstention rate, net repairs, retriever calls, and latency.

The six internal judges are locked in `rag/datasets/study/lock.json`: Llama 3.1 8B, Qwen2.5 7B, Mistral 7B Instruct v0.3, Gemma 2 9B, DeepSeek-R1-Distill-Llama 8B, and Phi-3.5-mini. Each development run writes its own `dev_score_<judge>.jsonl`. The final test is not part of those runs.

## Human labels

182 development cases contain the answer string but not as a whole word. Those ids, plus a 50-case audit drawn with seed 11, are in `rag/datasets/study/adjudication_dev.json`. Two people label them independently and do not see the policy choice. A third person is used only on a disagreement. The other development labels stay automatic.

## Go / no-go

Increase-k wins the 5% metric on every scored judge, and the guard matched no-guard. Stop the methodological paper. The CLI can be released later as a tool, with these development numbers and without a claim that selective repair wins. The final test stays sealed. The extension is not part of this study.
