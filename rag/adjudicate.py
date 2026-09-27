"""Which development labels a person must check.

Automatic labels stay automatic. A case is ambiguous only when the answer
string is inside the retrieved text but not as a whole word. Two people label
those cases independently. They do not see the policy's choice. A third person
is used only when the two disagree. This rule is fixed before the final test.
"""
from __future__ import annotations

import re

from rag.study import SPLIT_SEED, dataset_supported

AUDIT_SEED = SPLIT_SEED
AUDIT_DEV_SAMPLE = 50


def is_ambiguous(answer: str, context: str) -> bool:
    if not dataset_supported(answer, context):
        return False
    pattern = r"\b" + re.escape(" ".join(answer.strip().lower().split())) + r"\b"
    return re.search(pattern, context.lower()) is None
