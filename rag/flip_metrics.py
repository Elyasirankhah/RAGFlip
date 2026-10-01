"""Metrics for a retriever swap.

A question is supported when a judged relevant passage is inside the top k.
These functions are the measurement study. They are not the repair study.
"""
from __future__ import annotations

import math

Z_95 = 1.959963984540054


def counts(bm25_hit: bool, dense_hit: bool) -> dict:
    return {
        "fixed": int((not bm25_hit) and dense_hit),
        "broken": int(bm25_hit and not dense_hit),
        "both_correct": int(bm25_hit and dense_hit),
        "bm25_supported": int(bm25_hit),
        "dense_supported": int(dense_hit),
    }


def summarize(rows: list[tuple[bool, bool]], questions: int | None = None) -> dict:
    fixed = broken = both = bm25_hit = dense_hit = 0
    for had, now in rows:
        fixed += int((not had) and now)
        broken += int(had and (not now))
        both += int(had and now)
        bm25_hit += int(had)
        dense_hit += int(now)
    n = questions if questions is not None else len(rows)
    missed = n - bm25_hit
    return {
        "questions": n,
        "bm25_supported": bm25_hit,
        "dense_supported": dense_hit,
        "both_correct": both,
        "fixed": fixed,
        "broken": broken,
        "fix_rate": _rate(fixed, missed),
        "negative_flip_rate": _rate(broken, bm25_hit),
        "net_gain": fixed - broken,
        "compatibility": _rate(both, bm25_hit),
    }


def _rate(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return numerator / denominator


def wilson_interval(successes: int, total: int, z: float = Z_95):
    """95% Wilson interval for a binomial rate."""
    if total <= 0:
        return None
    probability = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(probability * (1.0 - probability) / total + z2 / (4.0 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def net_gain_interval(fixed: int, broken: int, questions: int, z: float = Z_95):
    """Wald interval for the sum of per-question scores: +1 fixed, -1 broken, 0 otherwise."""
    if questions <= 0:
        return None
    net = fixed - broken
    variance_of_sum = (fixed + broken) - (net * net) / questions
    half = z * math.sqrt(max(0.0, variance_of_sum))
    return [net - half, net + half]


def mcnemar_p(only_first: int, only_second: int) -> float:
    """Two-sided McNemar test, chi-square with 1 degree of freedom and continuity correction."""
    discordant = only_first + only_second
    if discordant <= 0:
        return 1.0
    statistic = (abs(only_first - only_second) - 1.0) ** 2 / discordant
    if statistic < 0:
        return 1.0
    return min(1.0, math.erfc(math.sqrt(statistic / 2.0)))


PUBLISHED = (
    ("Natural Questions, BGE", 3452, {1: (517, 194, 902), 5: (1240, 149, 1306), 10: (1600, 97, 1274), 20: (1956, 89, 1145), 50: (2347, 55, 919)}),
    ("Natural Questions, E5", 3452, {1: (517, 148, 1107), 5: (1240, 105, 1511), 10: (1600, 63, 1422), 20: (1956, 47, 1231), 50: (2347, 37, 953)}),
    ("HotpotQA, BGE", 7405, {1: (4875, 498, 1673), 5: (6130, 205, 1015), 10: (6484, 143, 747), 20: (6759, 105, 533), 50: (7025, 61, 318)}),
    ("HotpotQA, E5", 7405, {1: (4875, 492, 1692), 5: (6130, 187, 1016), 10: (6484, 156, 745), 20: (6759, 106, 524), 50: (7025, 66, 311)}),
    ("FiQA, BGE", 648, {1: (139, 27, 176), 5: (246, 25, 197), 10: (306, 28, 184), 20: (365, 16, 166), 50: (414, 10, 152)}),
    ("FiQA, E5", 648, {1: (139, 32, 162), 5: (246, 38, 170), 10: (306, 38, 165), 20: (365, 35, 147), 50: (414, 24, 140)}),
)


def published_lines():
    lines = []
    for name, questions, rows in PUBLISHED:
        lines.append(name)
        for k, (bm25_hit, broken, fixed) in rows.items():
            missed = questions - bm25_hit
            flip = wilson_interval(broken, bm25_hit)
            fix = wilson_interval(fixed, missed)
            compatibility = wilson_interval(bm25_hit - broken, bm25_hit)
            net = net_gain_interval(fixed, broken, questions)
            lines.append(
                f"{k}\t{broken / bm25_hit:.4f}\t[{flip[0]:.4f}, {flip[1]:.4f}]\t"
                f"{fixed / missed:.4f}\t[{fix[0]:.4f}, {fix[1]:.4f}]\t"
                f"{(bm25_hit - broken) / bm25_hit:.4f}\t[{compatibility[0]:.4f}, {compatibility[1]:.4f}]\t"
                f"{fixed - broken}\t[{net[0]:.1f}, {net[1]:.1f}]"
            )
    return lines


def with_intervals(summary: dict) -> dict:
    questions = summary["questions"]
    missed = questions - summary["bm25_supported"]
    result = dict(summary)
    result["fix_rate_ci"] = wilson_interval(summary["fixed"], missed)
    result["negative_flip_rate_ci"] = wilson_interval(summary["broken"], summary["bm25_supported"])
    result["compatibility_ci"] = wilson_interval(summary["both_correct"], summary["bm25_supported"])
    result["net_gain_ci"] = net_gain_interval(summary["fixed"], summary["broken"], questions)
    return result


if __name__ == "__main__":
    print("\n".join(published_lines()))
