from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Any, Mapping, Sequence


FEATURE_SPECS: tuple[tuple[str, bool], ...] = (
    ("day8_final_score", True),
    ("query_overlap", True),
    ("expanded_query_overlap", True),
    ("title_overlap", True),
    ("abstract_signal", True),
    ("canonical_constraint_coverage", True),
    ("raw_constraint_coverage", True),
    ("b3_semantic_score", True),
    ("b3_final_score", True),
    ("b4_pre_rerank_score", True),
    ("b4_source_count", True),
    ("b4_best_source_rank", False),
    ("evidence_match_rate", True),
    ("evidence_mean_confidence", True),
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_paper_features(paper: Mapping[str, Any]) -> dict[str, float]:
    raw = paper.get("raw")
    raw = raw if isinstance(raw, Mapping) else {}

    pure = raw.get("day8_pure_semantic_features")
    pure = pure if isinstance(pure, Mapping) else {}

    evidence = raw.get("day8_constraint_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    matched = [
        item
        for item in evidence
        if isinstance(item, Mapping) and bool(item.get("matched"))
    ]
    confidences = [
        safe_float(item.get("confidence"))
        for item in matched
    ]

    best_source_rank = raw.get("b4_best_source_rank")
    best_source_rank_value = (
        safe_float(best_source_rank, 999.0)
        if best_source_rank is not None
        else 999.0
    )

    return {
        "day8_final_score": safe_float(raw.get("day8_final_score")),
        "query_overlap": safe_float(pure.get("query_overlap")),
        "expanded_query_overlap": safe_float(
            pure.get("expanded_query_overlap")
        ),
        "title_overlap": safe_float(pure.get("title_overlap")),
        "abstract_signal": safe_float(pure.get("abstract_signal")),
        "canonical_constraint_coverage": safe_float(
            raw.get("day8_canonical_constraint_coverage")
        ),
        "raw_constraint_coverage": safe_float(
            raw.get("day8_raw_constraint_coverage")
        ),
        "b3_semantic_score": safe_float(raw.get("b3_semantic_score")),
        "b3_final_score": safe_float(raw.get("b3_final_score")),
        "b4_pre_rerank_score": safe_float(
            raw.get("b4_pre_rerank_score")
        ),
        "b4_source_count": safe_float(raw.get("b4_source_count")),
        "b4_best_source_rank": best_source_rank_value,
        "evidence_match_rate": (
            len(matched) / len(evidence)
            if evidence
            else 0.0
        ),
        "evidence_mean_confidence": (
            fmean(confidences)
            if confidences
            else 0.0
        ),
    }


def oriented_advantage(
    positive_value: float,
    negative_value: float,
    *,
    higher_is_better: bool,
) -> float:
    if higher_is_better:
        return positive_value - negative_value
    return negative_value - positive_value


def feature_sort_key(
    value: float,
    *,
    higher_is_better: bool,
) -> float:
    return value if higher_is_better else -value


@dataclass(slots=True)
class QueryRankingAttribution:
    qid: str
    question: str
    family: str
    gold_count: int
    first_hit_rank: int
    hit_ranks: list[int]
    blocker_count: int
    first_tp_features: dict[str, float]
    blocker_feature_means: dict[str, float]
    first_tp_advantages: dict[str, float]
    local_top5_hits_current: int
    local_top20_hits_current: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_query_attribution(
    *,
    qid: str,
    question: str,
    family: str,
    gold_count: int,
    hit_ranks: Sequence[int],
    papers: Sequence[Mapping[str, Any]],
) -> QueryRankingAttribution:
    ranks = sorted(int(rank) for rank in hit_ranks)
    if not ranks:
        raise ValueError("Ranking attribution requires at least one hit.")
    first_hit = ranks[0]
    if first_hit < 1 or first_hit > len(papers):
        raise ValueError(
            f"first_hit_rank={first_hit} outside candidate list "
            f"of size {len(papers)}"
        )

    first_features = extract_paper_features(papers[first_hit - 1])
    blockers = [
        extract_paper_features(papers[index])
        for index in range(first_hit - 1)
        if (index + 1) not in set(ranks)
    ]

    blocker_means: dict[str, float] = {}
    advantages: dict[str, float] = {}
    spec_by_name = dict(FEATURE_SPECS)
    for feature, _ in FEATURE_SPECS:
        values = [item[feature] for item in blockers]
        mean_value = fmean(values) if values else 0.0
        blocker_means[feature] = mean_value
        advantages[feature] = oriented_advantage(
            first_features[feature],
            mean_value,
            higher_is_better=spec_by_name[feature],
        )

    return QueryRankingAttribution(
        qid=qid,
        question=question,
        family=family,
        gold_count=int(gold_count),
        first_hit_rank=first_hit,
        hit_ranks=ranks,
        blocker_count=len(blockers),
        first_tp_features=first_features,
        blocker_feature_means=blocker_means,
        first_tp_advantages=advantages,
        local_top5_hits_current=sum(rank <= 5 for rank in ranks),
        local_top20_hits_current=sum(rank <= 20 for rank in ranks),
    )


def pairwise_feature_summary(
    *,
    ranking_rows: Sequence[Mapping[str, Any]],
    paper_records_by_qid: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    specs = dict(FEATURE_SPECS)
    accum: dict[str, dict[str, Any]] = {
        name: {
            "pair_count": 0,
            "wins": 0,
            "ties": 0,
            "first_tp_advantages": [],
        }
        for name, _ in FEATURE_SPECS
    }

    for row in ranking_rows:
        qid = str(row["qid"])
        record = paper_records_by_qid[qid]
        papers = record.get("papers") or []
        hit_ranks = sorted(int(x) for x in row.get("hit_ranks") or [])
        hit_set = set(hit_ranks)

        if not hit_ranks:
            continue

        first_rank = hit_ranks[0]
        first_features = extract_paper_features(papers[first_rank - 1])
        blockers = [
            extract_paper_features(papers[index])
            for index in range(first_rank - 1)
            if index + 1 not in hit_set
        ]

        for feature, higher_is_better in FEATURE_SPECS:
            blocker_values = [item[feature] for item in blockers]
            blocker_mean = (
                fmean(blocker_values)
                if blocker_values
                else 0.0
            )
            accum[feature]["first_tp_advantages"].append(
                oriented_advantage(
                    first_features[feature],
                    blocker_mean,
                    higher_is_better=higher_is_better,
                )
            )

        # All relevant papers versus false positives ranked before them.
        for hit_rank in hit_ranks:
            tp_features = extract_paper_features(
                papers[hit_rank - 1]
            )
            for index in range(hit_rank - 1):
                if index + 1 in hit_set:
                    continue
                fp_features = extract_paper_features(papers[index])
                for feature, higher_is_better in FEATURE_SPECS:
                    a = tp_features[feature]
                    b = fp_features[feature]
                    advantage = oriented_advantage(
                        a,
                        b,
                        higher_is_better=higher_is_better,
                    )
                    bucket = accum[feature]
                    bucket["pair_count"] += 1
                    if advantage > 1e-12:
                        bucket["wins"] += 1
                    elif abs(advantage) <= 1e-12:
                        bucket["ties"] += 1

    output: dict[str, Any] = {}
    for feature, _ in FEATURE_SPECS:
        item = accum[feature]
        pair_count = int(item["pair_count"])
        win_rate = (
            (
                int(item["wins"])
                + 0.5 * int(item["ties"])
            )
            / pair_count
            if pair_count
            else 0.0
        )
        advantages = list(item["first_tp_advantages"])
        output[feature] = {
            "higher_is_better": specs[feature],
            "pair_count": pair_count,
            "pairwise_tp_win_rate_vs_blocking_fp": win_rate,
            "first_tp_mean_oriented_advantage": (
                fmean(advantages) if advantages else 0.0
            ),
            "first_tp_median_oriented_advantage": (
                median(advantages) if advantages else 0.0
            ),
            "queries_with_positive_first_tp_advantage": sum(
                value > 0.0
                for value in advantages
            ),
            "query_count": len(advantages),
        }
    return output


def feature_only_counterfactual(
    *,
    ranking_rows: Sequence[Mapping[str, Any]],
    paper_records_by_qid: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Analysis-only feature-only rerank inside ranking-depth queries.

    Relevance labels come from Day13-1 strict hit ranks. The result is not a
    production metric or a recommended reranker. It only asks whether an
    existing signal can move known relevant papers upward.
    """
    output: dict[str, Any] = {}

    current_top5 = 0
    current_top20 = 0
    for row in ranking_rows:
        hits = [int(x) for x in row.get("hit_ranks") or []]
        current_top5 += sum(rank <= 5 for rank in hits)
        current_top20 += sum(rank <= 20 for rank in hits)

    for feature, higher_is_better in FEATURE_SPECS:
        top5_hits = 0
        top20_hits = 0
        improved_top20_queries = 0
        worsened_top20_queries = 0

        for row in ranking_rows:
            qid = str(row["qid"])
            papers = list(
                paper_records_by_qid[qid].get("papers") or []
            )
            hit_set = {
                int(rank)
                for rank in (row.get("hit_ranks") or [])
            }
            labels = [
                1 if index + 1 in hit_set else 0
                for index in range(len(papers))
            ]
            order = sorted(
                range(len(papers)),
                key=lambda index: feature_sort_key(
                    extract_paper_features(papers[index])[feature],
                    higher_is_better=higher_is_better,
                ),
                reverse=True,
            )
            reranked_labels = [labels[index] for index in order]

            current_query_top20 = sum(labels[:20])
            new_query_top20 = sum(reranked_labels[:20])

            top5_hits += sum(reranked_labels[:5])
            top20_hits += new_query_top20
            if new_query_top20 > current_query_top20:
                improved_top20_queries += 1
            elif new_query_top20 < current_query_top20:
                worsened_top20_queries += 1

        output[feature] = {
            "analysis_only": True,
            "ranking_query_count": len(ranking_rows),
            "current_label_hits_at_5": current_top5,
            "current_label_hits_at_20": current_top20,
            "feature_only_label_hits_at_5": top5_hits,
            "feature_only_label_hits_at_20": top20_hits,
            "delta_label_hits_at_5": top5_hits - current_top5,
            "delta_label_hits_at_20": top20_hits - current_top20,
            "queries_improved_at_20": improved_top20_queries,
            "queries_worsened_at_20": worsened_top20_queries,
        }

    return output


def derive_ranking_decision(
    *,
    pairwise_summary: Mapping[str, Mapping[str, Any]],
    counterfactual: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature, values in pairwise_summary.items():
        cf = counterfactual.get(feature) or {}
        win_rate = float(
            values.get("pairwise_tp_win_rate_vs_blocking_fp") or 0.0
        )
        delta20 = int(cf.get("delta_label_hits_at_20") or 0)
        if (
            feature != "day8_final_score"
            and win_rate >= 0.58
            and delta20 > 0
        ):
            candidates.append(
                {
                    "feature": feature,
                    "pairwise_win_rate": win_rate,
                    "delta_label_hits_at_20": delta20,
                    "delta_label_hits_at_5": int(
                        cf.get("delta_label_hits_at_5") or 0
                    ),
                }
            )

    candidates.sort(
        key=lambda item: (
            item["delta_label_hits_at_20"],
            item["pairwise_win_rate"],
            item["delta_label_hits_at_5"],
        ),
        reverse=True,
    )

    if candidates:
        return {
            "primary_decision": (
                "small_blended_calibration_ablation_warranted"
            ),
            "candidate_signals": candidates,
            "decision_text": (
                "At least one existing non-E3 signal separates known "
                "relevant papers from E3 blocking false positives often "
                "enough to justify one small, protected blended-calibration "
                "ablation. Do not replace E3 with a feature-only reranker."
            ),
            "guardrail": (
                "The next experiment must be offline, query-level CV where "
                "parameters are learned, must keep Top100 candidate identity "
                "unchanged, and must be rejected if held-out F1/NDCG or "
                "zero-recall protections regress."
            ),
        }

    return {
        "primary_decision": "freeze_ranking",
        "candidate_signals": [],
        "decision_text": (
            "No existing non-E3 feature shows enough blocker separation and "
            "local Top20 capture gain to justify another ranking experiment."
        ),
        "guardrail": (
            "Freeze E3 and move to retrieval/system competition polish."
        ),
    }
