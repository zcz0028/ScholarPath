from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from scholarpath.query.constraints import ConstraintDecomposition, QueryConstraint
from scholarpath.rerank.constraint_evidence import (
    CanonicalConstraint,
    ConstraintEvidence,
    build_canonical_constraints,
    build_constraint_evidence,
    normalize_constraint_text,
    normalized_tokens,
)
from scholarpath.rerank.semantic_rerank import (
    expanded_query_tokens,
    paper_text_parts,
    soft_overlap_score,
    token_set,
)

VARIANTS = ("E0", "E1", "E2", "E3")
DEFAULT_ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


@dataclass(slots=True, frozen=True)
class EvidenceAwareConfig:
    variant: str = "E2"
    alpha: float = 0.7

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")

    @property
    def beta(self) -> float:
        return 1.0 - self.alpha


@dataclass(slots=True, frozen=True)
class PureSemanticFeatures:
    query_overlap: float
    expanded_query_overlap: float
    title_overlap: float
    abstract_signal: float
    base_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class CoverageResult:
    matched_weight: float
    total_weight: float
    coverage: float
    matched_constraint_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["matched_constraint_ids"] = list(self.matched_constraint_ids)
        return data


@dataclass(slots=True, frozen=True)
class PaperDay8Features:
    b3_final_score: float
    normalized_b3_score: float
    pure_semantic_base: float
    raw_constraint_coverage: float
    canonical_constraint_coverage: float
    alpha: float
    beta: float
    final_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_pure_semantic_base(
    *,
    paper: dict[str, Any],
    question: str,
    decomposition: ConstraintDecomposition,
) -> PureSemanticFeatures:
    """Compute E3's isolated text-semantic base without legacy constraint signals.

    Deliberately excludes B2 score, original rank, old phrase/core constraint
    coverage, title boost, core-constraint bonus, identifier bonus and penalties.
    Existing semantic_rerank.py is imported only for stable text/token helpers;
    its business scoring function is not modified or reused here.
    """
    parts = paper_text_parts(paper)
    query_tokens = token_set(question)
    expanded_tokens = expanded_query_tokens(question, decomposition.constraints)
    doc_tokens = token_set(parts["all"])
    title_tokens = token_set(parts["title"])
    abstract_tokens = token_set(parts["abstract"])

    query_overlap = soft_overlap_score(query_tokens, doc_tokens)
    expanded_overlap = soft_overlap_score(expanded_tokens, doc_tokens)
    title_overlap = soft_overlap_score(query_tokens | expanded_tokens, title_tokens)
    abstract_signal = soft_overlap_score(query_tokens | expanded_tokens, abstract_tokens)

    base = (
        0.38 * query_overlap
        + 0.30 * expanded_overlap
        + 0.24 * title_overlap
        + 0.08 * abstract_signal
    )
    return PureSemanticFeatures(
        query_overlap=query_overlap,
        expanded_query_overlap=expanded_overlap,
        title_overlap=title_overlap,
        abstract_signal=abstract_signal,
        base_score=_clamp01(base),
    )


def raw_constraints_for_evidence(
    constraints: Sequence[QueryConstraint],
) -> list[CanonicalConstraint]:
    """Adapt raw planner constraints to the Day8 matcher without deduplication.

    This exists only for E1 ablation. Duplicate planner constraints remain
    duplicate evidence units, making E1 vs E2 isolate canonicalization/dedup.
    """
    output: list[CanonicalConstraint] = []
    for index, item in enumerate(constraints, start=1):
        text = normalize_constraint_text(item.text)
        terms = normalized_tokens(text)
        if not text or not terms:
            continue
        output.append(
            CanonicalConstraint(
                id=f"raw{index}_{item.id}",
                canonical_text=text,
                constraint_type=item.constraint_type,
                terms=terms,
                aliases=(text,),
                weight=max(0.0, float(item.weight)),
                source_constraint_ids=(item.id,),
            )
        )
    return output


def weighted_constraint_coverage(
    constraints: Sequence[CanonicalConstraint],
    evidence: Sequence[ConstraintEvidence],
) -> CoverageResult:
    """Compute one paper's weighted coverage from that paper's evidence only."""
    if not constraints:
        return CoverageResult(0.0, 0.0, 0.0, ())

    weights = {item.id: max(0.0, float(item.weight)) for item in constraints}
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        return CoverageResult(0.0, 0.0, 0.0, ())

    matched_ids = {
        item.constraint_id
        for item in evidence
        if item.matched and item.constraint_id in weights
    }
    matched_weight = sum(weights[cid] for cid in matched_ids)
    return CoverageResult(
        matched_weight=matched_weight,
        total_weight=total_weight,
        coverage=_clamp01(matched_weight / total_weight),
        matched_constraint_ids=tuple(sorted(matched_ids)),
    )


def minmax_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.5 for _ in values]
    span = high - low
    return [_clamp01((value - low) / span) for value in values]


def existing_b3_final_score(paper: Mapping[str, Any]) -> float:
    raw = paper.get("raw")
    if isinstance(raw, Mapping):
        for value in (
            raw.get("b3_final_score"),
            (raw.get("b3_features") or {}).get("final_score")
            if isinstance(raw.get("b3_features"), Mapping)
            else None,
        ):
            parsed = _safe_float(value)
            if parsed is not None:
                return parsed
    parsed = _safe_float(paper.get("score"))
    return parsed if parsed is not None else 0.0


def annotate_and_evidence_rerank_record(
    prediction_record: dict[str, Any],
    decomposition: ConstraintDecomposition,
    config: EvidenceAwareConfig,
) -> dict[str, Any]:
    question = str(prediction_record.get("question") or decomposition.question or "")
    papers = prediction_record.get("papers") or []
    if not isinstance(papers, list):
        papers = []

    canonical = build_canonical_constraints(
        question=question,
        constraints=decomposition.constraints,
    )
    raw_constraints = raw_constraints_for_evidence(decomposition.constraints)

    valid_papers = [paper for paper in papers if isinstance(paper, dict)]
    b3_scores = [existing_b3_final_score(paper) for paper in valid_papers]
    normalized_b3 = minmax_normalize(b3_scores)

    annotated: list[dict[str, Any]] = []
    for paper, b3_score, b3_norm in zip(valid_papers, b3_scores, normalized_b3):
        canonical_evidence = build_constraint_evidence(canonical, paper)
        raw_evidence = build_constraint_evidence(raw_constraints, paper)
        canonical_cov = weighted_constraint_coverage(canonical, canonical_evidence)
        raw_cov = weighted_constraint_coverage(raw_constraints, raw_evidence)
        pure = compute_pure_semantic_base(
            paper=paper,
            question=question,
            decomposition=decomposition,
        )

        if config.variant == "E0":
            final_score = b3_score
        elif config.variant == "E1":
            final_score = config.alpha * b3_norm + config.beta * raw_cov.coverage
        elif config.variant == "E2":
            final_score = config.alpha * b3_norm + config.beta * canonical_cov.coverage
        else:
            final_score = config.alpha * pure.base_score + config.beta * canonical_cov.coverage

        row = dict(paper)
        raw = dict(row.get("raw") or {})
        features = PaperDay8Features(
            b3_final_score=b3_score,
            normalized_b3_score=b3_norm,
            pure_semantic_base=pure.base_score,
            raw_constraint_coverage=raw_cov.coverage,
            canonical_constraint_coverage=canonical_cov.coverage,
            alpha=config.alpha,
            beta=config.beta,
            final_score=final_score,
        )
        raw.update(
            {
                "day8_variant": config.variant,
                "day8_features": features.to_dict(),
                "day8_pure_semantic_features": pure.to_dict(),
                "day8_raw_constraint_coverage": raw_cov.coverage,
                "day8_canonical_constraint_coverage": canonical_cov.coverage,
                "day8_alpha": config.alpha,
                "day8_beta": config.beta,
                "day8_final_score": final_score,
                "day8_constraint_evidence": [item.to_dict() for item in canonical_evidence],
            }
        )
        row["raw"] = raw
        annotated.append(row)

    # E0 is a true untouched-order baseline. E1/E2/E3 rerank. Python sort is
    # stable, so exact score ties preserve the existing B3 order.
    if config.variant != "E0":
        annotated.sort(
            key=lambda paper: _safe_float((paper.get("raw") or {}).get("day8_final_score")) or 0.0,
            reverse=True,
        )

    result = dict(prediction_record)
    result["papers"] = annotated
    result["day8_variant"] = config.variant
    result["day8_alpha"] = config.alpha
    result["day8_beta"] = config.beta
    result["day8_canonical_constraints"] = [item.to_dict() for item in canonical]
    return result


def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    return sum(
        (2.0 ** float(rel) - 1.0) / math.log2(rank + 1.0)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    actual = dcg_at_k(relevances, k=k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k=k)
    if ideal <= 0.0:
        return 0.0
    return actual / ideal


def deterministic_folds(qids: Sequence[str], folds: int = 5) -> list[list[str]]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    buckets: list[list[str]] = [[] for _ in range(folds)]
    for index, qid in enumerate(sorted(set(map(str, qids)))):
        buckets[index % folds].append(qid)
    return buckets


def select_alpha_from_cv(
    per_query_ndcg: Mapping[float, Mapping[str, float]],
    qids: Sequence[str],
    *,
    folds: int = 5,
    tie_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Evaluate fixed alpha candidates on held-out query folds.

    Each alpha is scored only on validation partitions; no paper-level split is
    used. Selection prefers higher mean held-out NDCG, and when scores are within
    tie_tolerance it prefers the larger alpha (smaller evidence beta).
    """
    fold_qids = deterministic_folds(qids, folds=folds)
    rows: list[dict[str, Any]] = []
    for alpha in sorted(per_query_ndcg, reverse=True):
        fold_scores: list[float] = []
        for validation in fold_qids:
            values = [per_query_ndcg[alpha].get(qid, 0.0) for qid in validation]
            fold_scores.append(sum(values) / len(values) if values else 0.0)
        mean = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
        variance = (
            sum((value - mean) ** 2 for value in fold_scores) / len(fold_scores)
            if fold_scores else 0.0
        )
        rows.append(
            {
                "alpha": alpha,
                "beta": 1.0 - alpha,
                "fold_ndcg_at_10": fold_scores,
                "mean_ndcg_at_10": mean,
                "std_ndcg_at_10": math.sqrt(variance),
            }
        )

    if not rows:
        return {"selected_alpha": 1.0, "selected_beta": 0.0, "candidates": []}

    best_mean = max(row["mean_ndcg_at_10"] for row in rows)
    near_best = [
        row for row in rows
        if best_mean - row["mean_ndcg_at_10"] <= tie_tolerance
    ]
    selected = max(near_best, key=lambda row: row["alpha"])
    return {
        "selected_alpha": selected["alpha"],
        "selected_beta": selected["beta"],
        "tie_tolerance": tie_tolerance,
        "candidates": rows,
    }


def _safe_float(value: object | None) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
