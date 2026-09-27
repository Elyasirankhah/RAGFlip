"""Laya / TypeSafe Jev System-1 decision backend.

Same wire for local Laya weights and a Jev-compatible HTTP server
(`POST /v1/systemone`). Used first as a claim verifier. Repair selection and
regression guards can call the same client later without changing the frozen
repair benchmarks.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from rag.judge import JudgeResult

DEFAULT_LAYA_MODEL = "english"
DEFAULT_LAYA_THRESHOLD = 0.5
SUPPORTED_NOUL = (
    "Is the claim fully entailed by the evidence? "
    "Paraphrase is OK. Related topic alone is not enough. "
    "Answer yes only if the evidence establishes the claim."
)


class SystemOneClient:
    """Thin client: local `laya.Router` or HTTP `/v1/systemone`."""

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        router: Any = None,
    ):
        self.backend = (backend or os.getenv("RAG_DEBUGGER_SYSTEMONE_BACKEND") or "local").lower()
        self.model = model or os.getenv("RAG_DEBUGGER_LAYA_MODEL", DEFAULT_LAYA_MODEL)
        self.base_url = (base_url or os.getenv("RAG_DEBUGGER_JEV_BASE_URL") or "").rstrip("/")
        self._router = router
        if self.backend == "http" and not self.base_url:
            raise SystemExit(
                "HTTP System One needs RAG_DEBUGGER_JEV_BASE_URL, e.g. http://127.0.0.1:8000"
            )
        if self.backend == "local" and self._router is None:
            try:
                from laya import Router
            except ImportError as exc:
                raise SystemExit(
                    "Local Laya needs the laya package. On the GPU node run: pip install laya"
                ) from exc
            self._router = Router()

    def decide(self, state: Any, questions: Dict[str, dict]) -> Dict[str, Any]:
        if self.backend == "http":
            return self._http(state, questions)
        kwargs = {}
        if self.model:
            kwargs["model"] = self.model
        return self._router.predict(state, questions, **kwargs)

    def _http(self, state: Any, questions: Dict[str, dict]) -> Dict[str, Any]:
        payload = {"state": state, "model": self.model or "laya", "questions": questions}
        request = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        api_key = os.getenv("RAG_DEBUGGER_JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"System One HTTP {exc.code}: {body}") from exc


class LayaSupportJudge:
    """Verifier: noul P(claim supported by evidence)."""

    def __init__(
        self,
        client: Optional[SystemOneClient] = None,
        threshold: Optional[float] = None,
    ):
        self.client = client or SystemOneClient()
        raw = os.getenv("RAG_DEBUGGER_LAYA_THRESHOLD")
        self.threshold = float(threshold if threshold is not None else (raw or DEFAULT_LAYA_THRESHOLD))

    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        if not claim.strip():
            return JudgeResult(label="unsupported", reason="Empty claim.")
        if not evidence.strip():
            return JudgeResult(label="unsupported", reason="No evidence provided.")
        state = {
            "question": question or "",
            "claim": claim,
            "evidence": evidence,
        }
        questions = {
            "supported": {
                "type": "noul",
                "instructions": SUPPORTED_NOUL,
            }
        }
        result = self.client.decide(state, questions)
        answers = result.get("answers") or {}
        answer = answers.get("supported") or {}
        prob = float(answer.get("noul", 0.0))
        if prob >= self.threshold:
            return JudgeResult("supported", f"laya noul={prob:.3f} threshold={self.threshold}")
        return JudgeResult("unsupported", f"laya noul={prob:.3f} threshold={self.threshold}")


def choose_repair(client: SystemOneClient, question: str, answer: str, retrieved: str) -> str:
    """Typed repair choice. Not used in the verifier experiment."""
    state = {"question": question, "answer": answer, "retrieved": retrieved}
    questions = {
        "repair": {
            "type": "choice",
            "instructions": "Pick the single smallest retrieval change to try next.",
            "criteria": {
                "increase_k": "The supporting passage is probably ranked just below the current k.",
                "rerun_retriever": "Retrieval looks flaky; call the retriever again at the same k.",
                "merge_retrieved": "Individual retrieved chunks are incomplete but together cover the claim.",
                "restore_original_query": "A logged rewrite likely dropped the supporting passage.",
                "rerank_candidates": "A logged second score prefers a candidate the retriever did not return.",
                "no_retrieval_change": "The answer is unsupported by any corpus evidence, or already fine.",
            },
        }
    }
    result = client.decide(state, questions)
    choice = ((result.get("answers") or {}).get("repair") or {}).get("choice")
    return str(choice or "no_retrieval_change")


def claim_regressed(
    client: SystemOneClient,
    claim: str,
    before_evidence: str,
    after_evidence: str,
    question: str = "",
) -> bool:
    """Regression guard: was a previously supported claim damaged?"""
    state = {
        "question": question or "",
        "claim": claim,
        "before_evidence": before_evidence,
        "after_evidence": after_evidence,
    }
    questions = {
        "damaged": {
            "type": "noul",
            "instructions": (
                "The claim was supported by the before evidence. "
                "After the repair, is the claim no longer supported by the after evidence?"
            ),
        }
    }
    result = client.decide(state, questions)
    prob = float(((result.get("answers") or {}).get("damaged") or {}).get("noul", 0.0))
    return prob >= DEFAULT_LAYA_THRESHOLD
