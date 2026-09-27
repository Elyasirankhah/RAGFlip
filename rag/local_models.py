"""Local open-weight entailment judge and embedder for GPU nodes."""
from __future__ import annotations

import os
from typing import List, Optional

from rag.judge import SYSTEM_PROMPT, JudgeResult, _parse_judge_output, build_user_prompt

# Official dense checkpoint. Hugging Face lists it near 28B because of the vision tower.
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"


class LocalEmbedder:
    """Sentence-transformer embeddings. Used to pick evidence before the judge."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RAG_DEBUGGER_LOCAL_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "Local embeddings need sentence-transformers. On the GPU node run: "
                'pip install -e ".[local]"'
            ) from exc
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        vector = self.model.encode(text or "", normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


def _load_causal_model(model_name: str, torch, causal_cls):
    """Qwen3.8-27B is a vision-language checkpoint; text judging still uses that class."""
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if any(tag in model_name for tag in ("Qwen3.8", "Qwen3.6", "Qwen3.5")):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    return causal_cls.from_pretrained(model_name, **kwargs)


class LocalSupportJudge:
    """Same entailment prompt as the OpenAI judge, run on a local instruct model."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RAG_DEBUGGER_LOCAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Local judge needs torch and transformers. On the GPU node run: "
                'pip install -e ".[local]"'
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = _load_causal_model(self.model_name, torch, AutoModelForCausalLM)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        if not claim.strip():
            return JudgeResult(label="unsupported", reason="Empty claim.")
        if not evidence.strip():
            return JudgeResult(label="unsupported", reason="No evidence provided.")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(claim, evidence, question)},
        ]
        return _parse_judge_output(self._generate(messages, max_new_tokens=120))

    def suggest_action(self, question: str, answer: str, retrieved: str) -> str:
        """Pick one repair name. This does not see the true label and does not run retrieval."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Choose one retrieval repair. Reply with exactly one of: "
                    "increase_k, rerun_retriever, no_retrieval_change."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nAnswer: {answer}\nRetrieved:\n{retrieved}\n"
                    "Reply with one name only."
                ),
            },
        ]
        raw = self._generate(messages, max_new_tokens=16).lower()
        for name in ("increase_k", "rerun_retriever", "no_retrieval_change"):
            if name in raw:
                return name
        return "no_retrieval_change"

    def _generate(self, messages: List[dict], max_new_tokens: int) -> str:
        template_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if "Qwen3" in self.model_name:
            template_kwargs["enable_thinking"] = False
        if "Qwen3.8" in self.model_name:
            template_kwargs["preserve_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
