"""Development retriever for repair selection. It is not the sealed holdout.

Wider k is not always a superset. The lamp query drops the precise card once
the token budget is filled with the noisy catalog.
"""
from __future__ import annotations

DOCS = {
    "paris": "Paris is the capital of France.",
    "bananas": "Bananas are a yellow fruit.",
    "lamp": "The desk lamp uses 40 watts.",
    "lamp_catalog": "The desk lamp catalog lists 400 watts for the industrial lamp.",
    "cern_left": "CERN was founded",
    "cern_right": "in 1961 near Geneva.",
}


def retrieve(query: str, k: int):
    lowered = query.lower()
    if "desk lamp" in lowered and k > 1:
        ranked = ["lamp_catalog", "bananas"]
    elif lowered.strip() == "capital of france":
        ranked = ["paris"]
    elif "capital" in lowered and "france" in lowered:
        ranked = ["bananas", "bananas", "bananas", "bananas", "bananas", "paris"]
    else:
        ranked = ["bananas"]
    return [{"id": name, "text": DOCS[name]} for name in ranked[: max(int(k), 1)]]
