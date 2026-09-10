"""Retrieval-evaluation metrics for the search benchmark harness.

Same formulas as the original hf-queries.py spreadsheet (Precision@k,
Recall@k, F1@k, MRR, AP, nDCG@k), reimplemented in Python so the app can
compute them live instead of relying on Excel formulas. Kept dependency-free
and pure so it's trivial to unit test.
"""

from __future__ import annotations

import math
from typing import Dict, List

METRIC_NAMES = ("precision", "recall", "f1", "mrr", "ap", "ndcg")


def compute_metrics(
    relevant_flags: List[bool],
    *,
    k: int,
    total_relevant: int,
) -> Dict[str, float]:
    """Compute one query's metrics from its judged result list.

    relevant_flags: judgments in rank order (index 0 = rank 1). An unjudged
    result must already be coerced to False by the caller — same as a blank
    yellow cell in the spreadsheet counting as "not relevant yet".
    k: the cutoff results were captured at (Precision@k divides by this,
    not by len(relevant_flags), matching the spreadsheet).
    total_relevant: known count of relevant documents in the corpus for this
    query, used as the Recall/AP denominator and the nDCG ideal ranking.
    """
    if not relevant_flags or total_relevant <= 0:
        return {name: 0.0 for name in METRIC_NAMES}

    found = sum(1 for r in relevant_flags if r)
    precision = (found / k) if k else 0.0
    recall = found / total_relevant
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mrr = 0.0
    for i, rel in enumerate(relevant_flags, start=1):
        if rel:
            mrr = 1.0 / i
            break

    cum = 0
    ap_sum = 0.0
    dcg_sum = 0.0
    idcg_sum = 0.0
    for i, rel in enumerate(relevant_flags, start=1):
        cum += 1 if rel else 0
        precision_at_i = cum / i
        if rel:
            ap_sum += precision_at_i
        dcg_sum += (1.0 if rel else 0.0) / math.log2(i + 1)
        if i <= total_relevant:
            idcg_sum += 1.0 / math.log2(i + 1)

    ap = ap_sum / total_relevant
    ndcg = (dcg_sum / idcg_sum) if idcg_sum else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "ap": ap,
        "ndcg": ndcg,
    }


def mean_metric(rows: List[Dict[str, float]], key: str) -> float:
    """Mean of one metric across queries — the MEAN row on the Summary sheet.
    The mean of the 'ap' column specifically is what's conventionally called MAP.
    """
    if not rows:
        return 0.0
    return sum(r[key] for r in rows) / len(rows)
