"""LLM support/entailment judge for a claim against evidence."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import openai


VALID_LABELS = {"supported", "partial", "unsupported"}

SYSTEM_PROMPT = (
    "You check whether a CLAIM is entailed by EVIDENCE.\n"
    "- supported: the evidence states the claim (paraphrase is OK).\n"
    "- partial: the evidence covers only part of the claim.\n"
    "- unsupported: the evidence does not establish the claim, "
    "contradicts it, or is only topically related.\n"
    "Related topic is not enough. Return JSON only: "
    '{"label":"supported"|"partial"|"unsupported","reason":"short"}'
)


@dataclass
class JudgeResult:
    label: str
    reason: str


def build_user_prompt(claim: str, evidence: str, question: str = "") -> str:
    return (
        "Question (context only):\n"
        f"{question or '(none)'}\n\n"
        "Claim:\n"
        f"{claim}\n\n"
        "Evidence:\n"
        f"{evidence}\n"
    )


class SupportJudge:
    """Decides whether a claim is entailed by evidence text."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model or os.getenv("RAG_DEBUGGER_JUDGE_MODEL", "gpt-4o-mini")

    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        if not claim.strip():
            return JudgeResult(label="unsupported", reason="Empty claim.")
        if not evidence.strip():
            return JudgeResult(label="unsupported", reason="No evidence provided.")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(claim, evidence, question)},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_judge_output(raw)


def _parse_judge_output(raw: str) -> JudgeResult:
    text = raw.strip()
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if fenced:
        text = fenced.group(0)
    try:
        payload = json.loads(text)
        label = str(payload.get("label", "unsupported")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
    except json.JSONDecodeError:
        label = "unsupported"
        reason = "Judge returned non-JSON output."
    if label not in VALID_LABELS:
        label = "unsupported"
    return JudgeResult(label=label, reason=reason or "No reason given.")
