from __future__ import annotations

import math

from src.tools.eval_metrics import compute_metrics


def test_no_results_returns_all_zero() -> None:
    m = compute_metrics([], k=10, total_relevant=1)
    assert all(v == 0.0 for v in m.values())


def test_no_relevant_found_returns_all_zero() -> None:
    m = compute_metrics([False] * 10, k=10, total_relevant=1)
    assert all(v == 0.0 for v in m.values())


def test_single_relevant_at_rank_one_is_perfect_except_precision() -> None:
    # With exactly one relevant document in the corpus, Precision@k is capped
    # at 1/k regardless of rank — same ceiling documented in the source
    # spreadsheet's notes.
    flags = [True] + [False] * 9
    m = compute_metrics(flags, k=10, total_relevant=1)
    assert m["precision"] == 0.1
    assert m["recall"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ap"] == 1.0
    assert m["ndcg"] == 1.0


def test_single_relevant_at_rank_three_degrades_rank_sensitive_metrics() -> None:
    flags = [False, False, True] + [False] * 7
    m = compute_metrics(flags, k=10, total_relevant=1)
    assert m["precision"] == 0.1
    assert m["recall"] == 1.0
    assert m["mrr"] == 1 / 3
    assert m["ap"] == 1 / 3
    # DCG = 1/log2(4); IDCG (capped at total_relevant=1) = 1/log2(2)
    expected_ndcg = (1 / math.log2(4)) / (1 / math.log2(2))
    assert abs(m["ndcg"] - expected_ndcg) < 1e-9


def test_f1_is_harmonic_mean_of_precision_and_recall() -> None:
    flags = [True, True, False, False, False]
    m = compute_metrics(flags, k=5, total_relevant=2)
    assert m["precision"] == 2 / 5
    assert m["recall"] == 1.0
    expected_f1 = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
    assert abs(m["f1"] - expected_f1) < 1e-9


def test_unjudged_results_must_be_precoerced_to_not_relevant() -> None:
    # The caller is responsible for turning an unjudged (None) result into
    # False before calling compute_metrics — mirrors a blank yellow cell in
    # the spreadsheet counting as "not relevant yet".
    flags = [bool(v) for v in [None, True, None]]
    m = compute_metrics(flags, k=3, total_relevant=1)
    assert m["mrr"] == 0.5
