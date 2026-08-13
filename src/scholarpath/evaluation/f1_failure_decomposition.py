from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence


DEFAULT_FIXED_K = 5
DEFAULT_TOP_K = 20
DEFAULT_EPSILON = 0.01


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_from_counts(tp: int, returned: int, gold_count: int) -> float:
    precision = safe_divide(tp, returned)
    recall = safe_divide(tp, gold_count)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def tp_at_k(hit_ranks: Sequence[int], k: int) -> int:
    return sum(1 for rank in hit_ranks if int(rank) <= int(k))


def query_f1_at_k(
    hit_ranks: Sequence[int],
    *,
    k: int,
    gold_count: int,
) -> float:
    if k <= 0:
        return 0.0
    return f1_from_counts(
        tp_at_k(hit_ranks, k),
        k,
        gold_count,
    )


def classify_failure_family(
    *,
    hit_ranks: Sequence[int],
    oracle_f1: float,
    fixed_f1: float,
    fixed_k: int = DEFAULT_FIXED_K,
    top_k: int = DEFAULT_TOP_K,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[str, str]:
    """Assign one mutually-exclusive Day13-3 failure family.

    The taxonomy is analysis-only. It uses strict matched ranks from the
    Day13-1 oracle artifact and therefore must never be used at production
    inference time.
    """
    ranks = sorted(int(rank) for rank in hit_ranks)
    if not ranks:
        return (
            "retrieval_ceiling",
            "No strict gold hit exists anywhere in the frozen E3 Top100.",
        )

    first_hit = ranks[0]
    oracle_gap = max(0.0, float(oracle_f1) - float(fixed_f1))

    if first_hit > top_k:
        return (
            "deep_ranked_hit",
            (
                f"The first strict hit appears after rank {top_k}; the paper "
                "is retrieved but too deep for normal result cutoffs."
            ),
        )

    if first_hit > fixed_k:
        return (
            "mid_ranked_hit",
            (
                f"The first strict hit appears after fixed K={fixed_k} but "
                f"within Top{top_k}; ranking depth dominates the fixed-cutoff loss."
            ),
        )

    if oracle_gap <= epsilon:
        return (
            "already_good",
            (
                f"Fixed K={fixed_k} is within {epsilon:.3f} absolute F1 of "
                "the query-level oracle."
            ),
        )

    return (
        "cutoff_mismatch",
        (
            f"At least one strict hit is already inside Top{fixed_k}, but "
            "the oracle gains materially by using a different result boundary."
        ),
    )


def headroom_source(family: str) -> str:
    mapping = {
        "retrieval_ceiling": "retrieval",
        "deep_ranked_hit": "ranking",
        "mid_ranked_hit": "ranking",
        "cutoff_mismatch": "selection",
        "already_good": "freeze",
    }
    return mapping.get(family, "unknown")


@dataclass(slots=True)
class FailureDecompositionRow:
    qid: str
    question: str
    gold_count: int
    candidate_count: int
    hit_ranks: list[int]
    first_hit_rank: int | None
    last_hit_rank: int | None
    tp_at_5: int
    tp_at_20: int
    tp_at_100: int
    fixed_k: int
    fixed_f1: float
    top20_f1: float
    oracle_best_k: int
    oracle_f1: float
    oracle_gap_vs_fixed: float
    family: str
    headroom_source: str
    family_reason: str
    adaptive_predicted_k: int | None = None
    adaptive_f1: float | None = None
    adaptive_gain_vs_fixed: float | None = None
    adaptive_regret_vs_oracle: float | None = None
    cutoff_error: int | None = None
    cutoff_error_abs: int | None = None
    cutoff_direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_failure_row(
    oracle_row: Mapping[str, Any],
    *,
    adaptive_row: Mapping[str, Any] | None = None,
    adaptive_query_f1: float | None = None,
    fixed_k: int = DEFAULT_FIXED_K,
    top_k: int = DEFAULT_TOP_K,
    epsilon: float = DEFAULT_EPSILON,
) -> FailureDecompositionRow:
    hit_ranks = sorted(
        int(rank)
        for rank in (oracle_row.get("hit_ranks") or [])
    )
    gold_count = int(oracle_row.get("gold_count") or 0)
    oracle_f1 = float(oracle_row.get("best_f1") or 0.0)
    oracle_best_k = int(oracle_row.get("best_k") or 1)

    fixed_f1 = query_f1_at_k(
        hit_ranks,
        k=fixed_k,
        gold_count=gold_count,
    )
    top20_f1 = query_f1_at_k(
        hit_ranks,
        k=top_k,
        gold_count=gold_count,
    )

    family, reason = classify_failure_family(
        hit_ranks=hit_ranks,
        oracle_f1=oracle_f1,
        fixed_f1=fixed_f1,
        fixed_k=fixed_k,
        top_k=top_k,
        epsilon=epsilon,
    )

    adaptive_predicted_k = None
    adaptive_f1 = None
    adaptive_gain = None
    adaptive_regret = None
    cutoff_error = None
    cutoff_error_abs = None
    cutoff_direction = None

    if adaptive_row is not None:
        adaptive_predicted_k = int(
            adaptive_row.get("predicted_k") or fixed_k
        )
        adaptive_f1 = (
            float(adaptive_query_f1)
            if adaptive_query_f1 is not None
            else query_f1_at_k(
                hit_ranks,
                k=adaptive_predicted_k,
                gold_count=gold_count,
            )
        )
        adaptive_gain = adaptive_f1 - fixed_f1
        adaptive_regret = oracle_f1 - adaptive_f1
        cutoff_error = adaptive_predicted_k - oracle_best_k
        cutoff_error_abs = abs(cutoff_error)
        if cutoff_error > 0:
            cutoff_direction = "over_return"
        elif cutoff_error < 0:
            cutoff_direction = "under_return"
        else:
            cutoff_direction = "exact"

    return FailureDecompositionRow(
        qid=str(oracle_row.get("qid") or ""),
        question=str(oracle_row.get("question") or ""),
        gold_count=gold_count,
        candidate_count=int(oracle_row.get("candidate_count") or 0),
        hit_ranks=hit_ranks,
        first_hit_rank=(
            int(oracle_row["first_hit_rank"])
            if oracle_row.get("first_hit_rank") is not None
            else None
        ),
        last_hit_rank=(
            int(oracle_row["last_hit_rank"])
            if oracle_row.get("last_hit_rank") is not None
            else None
        ),
        tp_at_5=tp_at_k(hit_ranks, 5),
        tp_at_20=tp_at_k(hit_ranks, 20),
        tp_at_100=tp_at_k(hit_ranks, 100),
        fixed_k=fixed_k,
        fixed_f1=fixed_f1,
        top20_f1=top20_f1,
        oracle_best_k=oracle_best_k,
        oracle_f1=oracle_f1,
        oracle_gap_vs_fixed=oracle_f1 - fixed_f1,
        family=family,
        headroom_source=headroom_source(family),
        family_reason=reason,
        adaptive_predicted_k=adaptive_predicted_k,
        adaptive_f1=adaptive_f1,
        adaptive_gain_vs_fixed=adaptive_gain,
        adaptive_regret_vs_oracle=adaptive_regret,
        cutoff_error=cutoff_error,
        cutoff_error_abs=cutoff_error_abs,
        cutoff_direction=cutoff_direction,
    )


def summarize_failure_rows(
    rows: Sequence[FailureDecompositionRow],
) -> dict[str, Any]:
    family_order = (
        "retrieval_ceiling",
        "deep_ranked_hit",
        "mid_ranked_hit",
        "cutoff_mismatch",
        "already_good",
    )
    total_queries = len(rows)

    fixed_macro_f1 = (
        fmean(row.fixed_f1 for row in rows)
        if rows
        else 0.0
    )
    top20_macro_f1 = (
        fmean(row.top20_f1 for row in rows)
        if rows
        else 0.0
    )
    oracle_macro_f1 = (
        fmean(row.oracle_f1 for row in rows)
        if rows
        else 0.0
    )

    adaptive_rows = [
        row
        for row in rows
        if row.adaptive_f1 is not None
    ]
    adaptive_macro_f1 = (
        fmean(float(row.adaptive_f1) for row in adaptive_rows)
        if adaptive_rows
        else None
    )

    family_summaries: dict[str, Any] = {}
    for family in family_order:
        members = [row for row in rows if row.family == family]
        gap_sum = sum(row.oracle_gap_vs_fixed for row in members)
        family_summaries[family] = {
            "query_count": len(members),
            "query_share": safe_divide(len(members), total_queries),
            "oracle_gap_sum_vs_fixed": gap_sum,
            "oracle_gap_macro_contribution": safe_divide(
                gap_sum,
                total_queries,
            ),
            "mean_oracle_gap_vs_fixed": (
                fmean(row.oracle_gap_vs_fixed for row in members)
                if members
                else 0.0
            ),
            "mean_gold_count": (
                fmean(row.gold_count for row in members)
                if members
                else 0.0
            ),
            "qids": [row.qid for row in members],
        }

    source_counts: dict[str, int] = {}
    source_gap: dict[str, float] = {}
    for row in rows:
        source_counts[row.headroom_source] = (
            source_counts.get(row.headroom_source, 0) + 1
        )
        source_gap[row.headroom_source] = (
            source_gap.get(row.headroom_source, 0.0)
            + row.oracle_gap_vs_fixed
        )

    adaptive_error = None
    if adaptive_rows:
        over = sum(
            row.cutoff_direction == "over_return"
            for row in adaptive_rows
        )
        under = sum(
            row.cutoff_direction == "under_return"
            for row in adaptive_rows
        )
        exact = sum(
            row.cutoff_direction == "exact"
            for row in adaptive_rows
        )
        adaptive_error = {
            "query_count": len(adaptive_rows),
            "macro_f1": adaptive_macro_f1,
            "gain_vs_fixed_macro_f1": (
                float(adaptive_macro_f1) - fixed_macro_f1
            ),
            "regret_vs_oracle_macro_f1": (
                oracle_macro_f1 - float(adaptive_macro_f1)
            ),
            "mean_absolute_cutoff_error": fmean(
                int(row.cutoff_error_abs or 0)
                for row in adaptive_rows
            ),
            "over_return_queries": over,
            "under_return_queries": under,
            "exact_cutoff_queries": exact,
        }

    return {
        "schema_version": "day13.f1-failure-decomposition.v1",
        "query_count": total_queries,
        "fixed_k": (
            rows[0].fixed_k if rows else DEFAULT_FIXED_K
        ),
        "top20_macro_f1_recomputed": top20_macro_f1,
        "fixed_macro_f1_recomputed": fixed_macro_f1,
        "oracle_macro_f1_recomputed": oracle_macro_f1,
        "oracle_gap_vs_fixed_macro_f1": (
            oracle_macro_f1 - fixed_macro_f1
        ),
        "family_summaries": family_summaries,
        "headroom_source_query_counts": source_counts,
        "headroom_source_gap_sums": source_gap,
        "adaptive_error_summary": adaptive_error,
        "analysis_boundary": {
            "gold_used": True,
            "gold_usage": "offline_failure_analysis_only",
            "production_inference": False,
            "ranking_modified": False,
            "api_calls": 0,
            "llm_calls": 0,
        },
    }


def choose_priority_queries(
    rows: Sequence[FailureDecompositionRow],
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Return analysis priorities without inventing production changes."""
    def key(row: FailureDecompositionRow) -> tuple[float, float, int, str]:
        if row.family == "retrieval_ceiling":
            # A retrieval ceiling has zero Oracle-K gap because cutoff cannot
            # recover a paper that is absent. Use missing gold volume only for
            # analysis priority, never as an inference signal.
            return (
                2.0,
                float(row.gold_count),
                0,
                row.qid,
            )
        return (
            1.0,
            float(row.oracle_gap_vs_fixed),
            int(row.tp_at_100),
            row.qid,
        )

    ranked = sorted(rows, key=key, reverse=True)[: max(0, int(limit))]
    output: list[dict[str, Any]] = []
    for row in ranked:
        output.append(
            {
                "qid": row.qid,
                "family": row.family,
                "headroom_source": row.headroom_source,
                "gold_count": row.gold_count,
                "first_hit_rank": row.first_hit_rank,
                "oracle_best_k": row.oracle_best_k,
                "fixed_f1": row.fixed_f1,
                "oracle_f1": row.oracle_f1,
                "oracle_gap_vs_fixed": row.oracle_gap_vs_fixed,
                "recommended_next_analysis": {
                    "retrieval_ceiling": (
                        "Audit retrieval coverage; cutoff/rerank cannot recover "
                        "a missing Top100 gold paper."
                    ),
                    "deep_ranked_hit": (
                        "Audit why known relevant papers are ranked below Top20; "
                        "do not widen output blindly."
                    ),
                    "mid_ranked_hit": (
                        "Audit E3 score calibration around ranks 6-20."
                    ),
                    "cutoff_mismatch": (
                        "Audit result-boundary confidence and score-shape "
                        "features without using gold at inference."
                    ),
                    "already_good": "Freeze; low optimization priority.",
                }[row.family],
            }
        )
    return output


def derive_go_no_go(summary: Mapping[str, Any]) -> dict[str, Any]:
    families = summary.get("family_summaries") or {}
    retrieval_n = int(
        (families.get("retrieval_ceiling") or {}).get(
            "query_count",
            0,
        )
    )
    ranking_n = int(
        (families.get("deep_ranked_hit") or {}).get(
            "query_count",
            0,
        )
    ) + int(
        (families.get("mid_ranked_hit") or {}).get(
            "query_count",
            0,
        )
    )
    selection_n = int(
        (families.get("cutoff_mismatch") or {}).get(
            "query_count",
            0,
        )
    )

    adaptive = summary.get("adaptive_error_summary")
    adaptive_beats_fixed = bool(
        adaptive
        and float(adaptive.get("gain_vs_fixed_macro_f1") or 0.0) > 0
    )

    # Day13-2 already establishes whether a gold-free selector generalizes.
    # If it fails to beat the fixed policy, do not recommend more selector
    # tuning merely because Oracle-K is high.
    if selection_n > 0 and not adaptive_beats_fixed:
        primary = "freeze_selector"
        decision = (
            "Do not continue adaptive-cutoff tuning. Preserve the simple "
            "fixed policy as the internal F1 baseline and inspect ranking/"
            "retrieval bottlenecks before any further algorithm work."
        )
    elif ranking_n > retrieval_n and ranking_n >= selection_n:
        primary = "ranking_diagnostic"
        decision = (
            "A final small ranking/calibration diagnostic is justified; "
            "do not alter the frozen E3 production ranking yet."
        )
    elif retrieval_n >= ranking_n and retrieval_n >= selection_n:
        primary = "retrieval_diagnostic"
        decision = (
            "Retrieval ceiling is the dominant unresolved family. Only a "
            "targeted, hypothesis-driven retrieval experiment would be "
            "justified; broad query expansion is already rejected."
        )
    else:
        primary = "competition_polish"
        decision = (
            "No additional algorithm experiment is justified by the current "
            "failure decomposition; move to competition polish."
        )

    return {
        "primary_decision": primary,
        "decision_text": decision,
        "retrieval_ceiling_queries": retrieval_n,
        "ranking_depth_queries": ranking_n,
        "selection_mismatch_queries": selection_n,
        "adaptive_beats_fixed": adaptive_beats_fixed,
    }
