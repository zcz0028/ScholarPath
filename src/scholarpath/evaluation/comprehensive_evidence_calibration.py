from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.constraint_evidence import (
    CanonicalConstraint,
    build_canonical_constraints,
    build_constraint_evidence,
)
from scholarpath.evaluation.protected_blended_calibration import minmax, safe_float

BETA_GRID: tuple[float, ...] = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10)

_FIELD_QUALITY = {"title": 1.00, "abstract": 0.92, "concept": 0.78}
_TYPE_SPECIFICITY = {
    "model_or_entity": 0.82,
    "topic": 1.00,
    "method": 1.00,
    "task": 1.00,
    "property": 1.00,
    "constraint": 1.00,
}


def _paper_mapping(paper: Any) -> dict[str, Any]:
    if isinstance(paper, Mapping):
        return dict(paper)
    to_dict = getattr(paper, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict(include_raw=True))
        except TypeError:
            return dict(to_dict())
    raise TypeError(f"Unsupported paper value: {type(paper)!r}")


def _raw(paper: Any) -> Mapping[str, Any]:
    if isinstance(paper, Mapping):
        raw = paper.get("raw")
    else:
        raw = getattr(paper, "raw", None)
    return raw if isinstance(raw, Mapping) else {}


def e3_score(paper: Any) -> float:
    return safe_float(_raw(paper).get("day8_final_score"))


@dataclass(slots=True, frozen=True)
class ComprehensiveEvidenceBreakdown:
    score: float
    weighted_coverage: float
    matched_confidence: float
    field_quality: float
    specificity_factor: float
    generic_entity_only_penalty: float
    matched_constraint_count: int
    constraint_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "weighted_coverage": self.weighted_coverage,
            "matched_confidence": self.matched_confidence,
            "field_quality": self.field_quality,
            "specificity_factor": self.specificity_factor,
            "generic_entity_only_penalty": self.generic_entity_only_penalty,
            "matched_constraint_count": self.matched_constraint_count,
            "constraint_count": self.constraint_count,
        }


def comprehensive_evidence_score(
    constraints: Sequence[CanonicalConstraint],
    paper: Any,
) -> ComprehensiveEvidenceBreakdown:
    if not constraints:
        return ComprehensiveEvidenceBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    evidence = build_constraint_evidence(constraints, _paper_mapping(paper))
    by_id = {item.constraint_id: item for item in evidence}
    total_weight = sum(max(0.0, float(c.weight)) for c in constraints)
    matched = [item for item in evidence if item.matched]

    matched_weight = sum(
        max(0.0, float(c.weight))
        for c in constraints
        if by_id.get(c.id) is not None and by_id[c.id].matched
    )
    weighted_coverage = matched_weight / total_weight if total_weight else 0.0

    if matched_weight:
        matched_confidence = sum(
            max(0.0, float(c.weight)) * by_id[c.id].confidence * by_id[c.id].token_coverage
            for c in constraints
            if by_id.get(c.id) is not None and by_id[c.id].matched
        ) / matched_weight
        field_quality = sum(
            max(0.0, float(c.weight)) * _FIELD_QUALITY.get(by_id[c.id].evidence_field or "", 0.70)
            for c in constraints
            if by_id.get(c.id) is not None and by_id[c.id].matched
        ) / matched_weight
        specificity_factor = sum(
            max(0.0, float(c.weight)) * _TYPE_SPECIFICITY.get(c.constraint_type, 0.95)
            for c in constraints
            if by_id.get(c.id) is not None and by_id[c.id].matched
        ) / matched_weight
    else:
        matched_confidence = field_quality = specificity_factor = 0.0

    generic_only = (
        len(matched) == 1 and matched[0].constraint_type == "model_or_entity"
    )
    generic_penalty = 0.15 if generic_only else 0.0
    raw_score = (
        weighted_coverage
        * matched_confidence
        * field_quality
        * specificity_factor
        - generic_penalty
    )
    score = max(0.0, min(1.0, raw_score))
    return ComprehensiveEvidenceBreakdown(
        score=score,
        weighted_coverage=weighted_coverage,
        matched_confidence=matched_confidence,
        field_quality=field_quality,
        specificity_factor=specificity_factor,
        generic_entity_only_penalty=generic_penalty,
        matched_constraint_count=len(matched),
        constraint_count=len(constraints),
    )


def canonical_constraints_for_question(question: str) -> list[CanonicalConstraint]:
    decomposition = ConstraintDecomposer().decompose(question)
    return build_canonical_constraints(
        question=question,
        constraints=decomposition.constraints,
    )


def blended_scores(
    papers: Sequence[Any],
    *,
    constraints: Sequence[CanonicalConstraint],
    beta: float,
) -> list[float]:
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    e3 = minmax([e3_score(paper) for paper in papers])
    evidence = [comprehensive_evidence_score(constraints, paper).score for paper in papers]
    return [(1.0 - beta) * a + beta * b for a, b in zip(e3, evidence)]


def rerank_papers(
    papers: Sequence[Any],
    *,
    constraints: Sequence[CanonicalConstraint],
    beta: float,
) -> list[Any]:
    papers = list(papers)
    if beta <= 1e-15:
        return papers
    scores = blended_scores(papers, constraints=constraints, beta=beta)
    order = sorted(range(len(papers)), key=lambda i: (-scores[i], i))
    return [papers[i] for i in order]


def rerank_prediction_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    beta: float,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for qid, record in records.items():
        item = dict(record)
        question = str(record.get("question") or "")
        constraints = canonical_constraints_for_question(question)
        item["papers"] = rerank_papers(
            list(record.get("papers") or []), constraints=constraints, beta=beta
        )
        output[str(qid)] = item
    return output
