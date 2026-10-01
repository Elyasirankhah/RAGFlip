"""Locks the retriever-swap metric definitions to the Natural Questions k=10 count."""
from rag.flip_metrics import mcnemar_p, net_gain_interval, summarize, wilson_interval, with_intervals


def test_nq_bge_k10_definitions():
    summary = summarize([(True, False)] * 97 + [(True, True)] * 1503 + [(False, True)] * 1274 + [(False, False)] * 578, questions=3452)
    assert summary["bm25_supported"] == 1600
    assert summary["fixed"] == 1274
    assert summary["broken"] == 97
    assert summary["both_correct"] == 1503
    assert summary["net_gain"] == 1177
    assert summary["negative_flip_rate"] == 97 / 1600
    assert summary["fix_rate"] == 1274 / (3452 - 1600)
    assert summary["compatibility"] == 1503 / 1600


def test_wilson_and_net_intervals_cover_the_point():
    interval = wilson_interval(97, 1600)
    assert interval[0] < 97 / 1600 < interval[1]
    net_interval = net_gain_interval(1274, 97, 3452)
    assert net_interval[0] < 1177 < net_interval[1]


def test_with_intervals_names_every_rate():
    reported = with_intervals(summarize([(True, False), (False, True)], questions=2))
    assert set(reported) >= {
        "fix_rate",
        "negative_flip_rate",
        "net_gain",
        "compatibility",
        "fix_rate_ci",
        "negative_flip_rate_ci",
        "compatibility_ci",
        "net_gain_ci",
    }


def test_mcnemar_is_large_when_the_models_disagree_equally():
    assert mcnemar_p(40, 40) > 0.5
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(80, 5) < 0.001
