"""Off-the-shelf NLI entailment judge. Separate from frozen repair numbers."""
from __future__ import annotations

import os
from typing import Optional

from rag.judge import JudgeResult

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLISupportJudge:
    """Premise = evidence, hypothesis = claim. Entailment maps to supported."""

    def __init__(self, model_name: Optional[str] = None, pipe=None):
        self.model_name = model_name or os.getenv("RAG_DEBUGGER_NLI_JUDGE_MODEL", DEFAULT_NLI_MODEL)
        if pipe is not None:
            self.pipe = pipe
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise SystemExit(
                "NLI judge needs transformers. On the GPU node run: pip install -e \".[local]\""
            ) from exc
        self.pipe = pipeline(
            "text-classification",
            model=self.model_name,
            truncation=True,
            max_length=512,
        )

    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        del question
        if not claim.strip():
            return JudgeResult(label="unsupported", reason="Empty claim.")
        if not evidence.strip():
            return JudgeResult(label="unsupported", reason="No evidence provided.")
        result = self.pipe({"text": evidence, "text_pair": claim}, top_k=None)
        if isinstance(result, list) and result and isinstance(result[0], list):
            scores = {row["label"].lower(): float(row["score"]) for row in result[0]}
        elif isinstance(result, list):
            scores = {row["label"].lower(): float(row["score"]) for row in result}
        else:
            scores = {str(result.get("label", "")).lower(): float(result.get("score", 0.0))}
        entail = scores.get("entailment", scores.get("entail", 0.0))
        contradict = scores.get("contradiction", scores.get("contradict", 0.0))
        neutral = scores.get("neutral", 0.0)
        if entail >= contradict and entail >= neutral and entail >= 0.5:
            return JudgeResult("supported", f"nli entailment={entail:.3f}")
        if contradict >= entail and contradict >= neutral:
            return JudgeResult("unsupported", f"nli contradiction={contradict:.3f}")
        return JudgeResult("unsupported", f"nli neutral={neutral:.3f}")
