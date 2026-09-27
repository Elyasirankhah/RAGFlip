"""Response-level RAGTruth detector. One model call per answer, batched with vLLM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.datasets.ragtruth import _tally, export_split, read_examples

SYSTEM_PROMPT = (
    "You decide whether a RAG response is fully supported by the reference.\n"
    "supported: every claim is stated or fairly paraphrased by the reference.\n"
    "hallucination: any claim is absent, contradicted, or only topically related.\n"
    'Return JSON only: {"label":"supported"|"hallucination"}'
)


def evidence_text(example: dict, limit: int = 12000) -> str:
    chunks = (example.get("trace") or {}).get("retrieved_chunks") or []
    text = "\n\n".join(str(chunk.get("text") or "") for chunk in chunks)
    return text[:limit]


def detector_messages(example: dict, with_label: bool = False) -> List[dict]:
    trace = example.get("trace") or {}
    user = (
        f"Question:\n{str(trace.get('question') or '')[:2000]}\n\n"
        f"Reference:\n{evidence_text(example)}\n\n"
        f"Response:\n{str(trace.get('answer') or '')[:8000]}\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if with_label:
        label = example.get("gold") if example.get("gold") in {"supported", "hallucination"} else "hallucination"
        messages.append({"role": "assistant", "content": json.dumps({"label": label})})
    return messages


def parse_detector_label(raw: str) -> str:
    text = (raw or "").strip()
    fenced = text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        fenced = text[start : end + 1]
    try:
        label = str(json.loads(fenced).get("label", "")).strip().lower()
    except json.JSONDecodeError:
        label = text.lower()
    if "hallucination" in label:
        return "hallucination"
    if "supported" in label:
        return "supported"
    return "hallucination"


def render_prompt(tokenizer, example: dict) -> str:
    return tokenizer.apply_chat_template(
        detector_messages(example),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def score_with_vllm(
    examples: List[dict],
    model: str,
    checkpoint_path: Path,
    summary_path: Path,
    batch_size: int = 64,
) -> Dict[str, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("vLLM is not installed. On the GPU node run: pip install vllm") from exc

    done_ids = set()
    rows: List[dict] = []
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            done_ids.add(row.get("response_id"))
    pending = [example for example in examples if _response_id(example) not in done_ids]
    if done_ids:
        print(f"Resuming: {len(rows)} already saved, {len(pending)} left", flush=True)

    tokenizer = None
    llm = None
    if pending:
        llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.9)
        tokenizer = llm.get_tokenizer()
    sampling = None
    if llm is not None:
        sampling = SamplingParams(temperature=0, max_tokens=40)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompts = []
            for example in batch:
                try:
                    prompts.append(render_prompt(tokenizer, example))
                except TypeError:
                    prompts.append(
                        tokenizer.apply_chat_template(
                            detector_messages(example),
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    )
            outputs = llm.generate(prompts, sampling)
            for example, output in zip(batch, outputs):
                pred = parse_detector_label(output.outputs[0].text)
                row = {
                    "response_id": _response_id(example),
                    "gold": example.get("gold"),
                    "pred": pred,
                    "task_type": (example.get("trace") or {}).get("metadata", {}).get("task_type") or "unknown",
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(f"[{len(rows)}/{len(examples)}] batch saved", flush=True)
            _write_summary(summary_path, {"judge": "vllm", "model": model, "complete": False, **_tally(rows)})

    report = {"judge": "vllm", "model": model, "complete": len(rows) == len(examples), **_tally(rows), "rows": rows}
    _write_summary(summary_path, {key: value for key, value in report.items() if key != "rows"})
    return report


def _response_id(example: dict) -> Any:
    return (example.get("trace") or {}).get("metadata", {}).get("response_id")


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score RAGTruth with a vLLM response detector")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Merged fine-tuned model directory or HF id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    export_path = args.out / f"{args.split}.jsonl"
    if not export_path.exists():
        export_split(args.root, export_path, args.split)
    examples = read_examples(export_path)
    report = score_with_vllm(
        examples,
        model=args.model,
        checkpoint_path=args.out / f"{args.split}_vllm_partial.jsonl",
        summary_path=args.out / f"{args.split}_vllm.json",
        batch_size=args.batch_size,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
