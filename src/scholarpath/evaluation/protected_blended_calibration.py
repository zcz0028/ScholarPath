from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import log2
from statistics import fmean
from typing import Any, Mapping, Sequence

from scholarpath.evaluation.matching import match_papers


BETA_GRID: tuple[float, ...] = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def minmax(values: Sequence[float]) -> list[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-15:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def source_rank_prior(rank: Any) -> float:
    """Convert B4 best-source rank into a bounded retrieval prior.

    Missing/invalid ranks map to 0. The best possible rank maps to 1.
    """
    try:
        value = int(rank)
    except (TypeError, ValueError):
        return 0.0
    if value < 1:
        return 0.0
    return 1.0 / log2(value + 1.0)


def _raw(paper: Any) -> Mapping[str, Any]:
    if isinstance(paper, Mapping):
        raw = paper.get("raw")
    else:
        raw = getattr(paper, "raw", None)
    return raw if isinstance(raw, Mapping) else {}


def e3_score(paper: Any) -> float:
    return safe_float(_raw(paper).get("day8_final_score"))


def b4_best_rank(paper: Any) -> Any:
    return _raw(paper).get("b4_best_source_rank")


def blended_scores(
    papers: Sequence[Any],
    *,
    beta: float,
) -> list[float]:
    """Compute protected E3 + B4 source-rank prior scores.

    E3 remains the dominant signal. E3 scores are min-max normalized within
    each query. The B4 rank prior is already bounded to [0, 1].
    """
    beta = float(beta)
    if beta < 0.0 or beta > 1.0:
        raise ValueError("beta must be in [0, 1]")

    e3 = minmax([e3_score(paper) for paper in papers])
    prior = [source_rank_prior(b4_best_rank(paper)) for paper in papers]
    return [
        (1.0 - beta) * e3_value + beta * prior_value
        for e3_value, prior_value in zip(e3, prior)
    ]


def rerank_papers(
    papers: Sequence[Any],
    *,
    beta: float,
) -> list[Any]:
    """Return a stable reranking of exactly the same candidate objects."""
    papers = list(papers)
    if beta <= 1e-15:
        return papers

    scores = blended_scores(papers, beta=beta)
    order = sorted(
        range(len(papers)),
        key=lambda index: (
            -scores[index],
            index,
        ),
    )
    return [papers[index] for index in order]


def assign_query_folds(
    qids: Sequence[str],
    *,
    n_folds: int = 5,
) -> dict[str, int]:
    """Deterministic query-level folds.

    Sorting by a stable SHA-256 key avoids depending on input file ordering;
    round-robin assignment keeps fold sizes as balanced as possible.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    unique = sorted(
        {str(qid) for qid in qids},
        key=lambda qid: (
            sha256(qid.encode("utf-8")).hexdigest(),
            qid,
        ),
    )
    return {
        qid: index % n_folds
        for index, qid in enumerate(unique)
    }


def _papers(record: Mapping[str, Any]) -> list[Any]:
    return list(record.get("papers") or [])


def rerank_prediction_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    beta: float,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for qid, record in records.items():
        item = dict(record)
        item["papers"] = rerank_papers(
            _papers(record),
            beta=beta,
        )
        output[str(qid)] = item
    return output


def subset_records(
    records: Mapping[str, Mapping[str, Any]],
    qids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    wanted = {str(qid) for qid in qids}
    return {
        str(qid): record
        for qid, record in records.items()
        if str(qid) in wanted
    }


def binary_ndcg_at_10(
    *,
    gold_records: Mapping[str, Mapping[str, Any]],
    prediction_records: Mapping[str, Mapping[str, Any]],
) -> float:
    per_query: list[float] = []

    for qid, gold_record in gold_records.items():
        prediction_record = prediction_records.get(qid)
        predictions = (
            list(prediction_record.get("papers") or [])[:10]
            if prediction_record is not None
            else []
        )
        gold = list(gold_record.get("papers") or [])

        match = match_papers(
            predictions=predictions,
            gold=gold,
            mode="strict",
        )
        dcg = sum(
            1.0 / log2(pair.prediction_index + 2.0)
            for pair in match.pairs
        )

        ideal_count = min(len(gold), 10)
        idcg = sum(
            1.0 / log2(rank + 2.0)
            for rank in range(ideal_count)
        )
        per_query.append(dcg / idcg if idcg else 0.0)

    return fmean(per_query) if per_query else 0.0


def strict_top100_tp_and_zero_recall(
    *,
    gold_records: Mapping[str, Mapping[str, Any]],
    prediction_records: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int]:
    total_tp = 0
    zero_recall = 0

    for qid, gold_record in gold_records.items():
        prediction_record = prediction_records.get(qid)
        predictions = (
            list(prediction_record.get("papers") or [])[:100]
            if prediction_record is not None
            else []
        )
        gold = list(gold_record.get("papers") or [])
        match = match_papers(
            predictions=predictions,
            gold=gold,
            mode="strict",
        )
        total_tp += match.tp
        if match.tp == 0:
            zero_recall += 1

    return total_tp, zero_recall


def _identity_signature(paper: Any) -> tuple[str, ...]:
    identity_keys = getattr(paper, "identity_keys", None)
    if callable(identity_keys):
        keys = sorted(str(key) for key in identity_keys())
        if keys:
            return tuple(keys)

    normalized = getattr(paper, "normalized_title", None)
    if normalized:
        return ("title", str(normalized))

    title = getattr(paper, "title", None)
    if title:
        return ("raw_title", str(title))

    if isinstance(paper, Mapping):
        normalized = paper.get("normalized_title")
        if normalized:
            return ("title", str(normalized))
        title = paper.get("title")
        if title:
            return ("raw_title", str(title))

    return ("object", repr(paper))


def candidate_multiset(record: Mapping[str, Any]) -> Counter[tuple[str, ...]]:
    return Counter(
        _identity_signature(paper)
        for paper in _papers(record)
    )


def candidate_identity_unchanged(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> bool:
    if set(baseline) != set(candidate):
        return False
    return all(
        candidate_multiset(baseline[qid])
        == candidate_multiset(candidate[qid])
        for qid in baseline
    )


@dataclass(slots=True, frozen=True)
class FoldSelection:
    fold: int
    train_qids: tuple[str, ...]
    validation_qids: tuple[str, ...]
    selected_beta: float
    train_macro_f1_at_5: float
    train_ndcg_at_10: float


def select_beta(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select by Macro-F1@5, then NDCG@10, then smaller beta."""
    if not rows:
        raise ValueError("No beta candidates provided")
    return max(
        rows,
        key=lambda row: (
            float(row["macro_f1_at_5"]),
            float(row["ndcg_at_10"]),
            -float(row["beta"]),
        ),
    )
