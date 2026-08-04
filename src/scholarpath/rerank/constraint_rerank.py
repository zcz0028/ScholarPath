from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from scholarpath.query.constraints import ConstraintDecomposition, QueryConstraint


@dataclass(slots=True)
class ConstraintRerankConfig:
    mode: str = "hybrid"
    rank_weight: float = 0.70
    constraint_weight: float = 0.30
    occurrence_weight: float = 0.05
    strong_id_bonus: float = 0.01
    core_constraint_bonus: float = 0.05
    missing_core_penalty: float = 0.04


MODE_PRESETS = {
    "conservative": {
        "rank_weight": 0.85,
        "constraint_weight": 0.15,
        "occurrence_weight": 0.03,
        "core_constraint_bonus": 0.03,
        "missing_core_penalty": 0.02,
    },
    "hybrid": {
        "rank_weight": 0.70,
        "constraint_weight": 0.30,
        "occurrence_weight": 0.05,
        "core_constraint_bonus": 0.05,
        "missing_core_penalty": 0.04,
    },
    "coverage_first": {
        "rank_weight": 0.50,
        "constraint_weight": 0.50,
        "occurrence_weight": 0.05,
        "core_constraint_bonus": 0.08,
        "missing_core_penalty": 0.06,
    },
}


CORE_TYPES = {
    "performance_relation",
    "data_condition",
    "method_or_property",
}


def make_config(mode: str) -> ConstraintRerankConfig:
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown rerank mode: {mode}")
    values = {"mode": mode, **MODE_PRESETS[mode]}
    return ConstraintRerankConfig(**values)


def normalize_text(value: object | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold()
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_identifier(value: object | None) -> str:
    text = str(value or "").strip()
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("https://arxiv.org/abs/", "")
    text = text.replace("arXiv:", "").replace("arxiv:", "")
    text = re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)
    return text.casefold().strip()


def paper_key(paper: dict[str, Any]) -> str:
    for field, prefix in (
        ("doi", "doi"),
        ("arxiv_id", "arxiv"),
        ("openalex_id", "openalex"),
        ("semantic_scholar_id", "s2"),
        ("canonical_id", "canonical"),
    ):
        value = normalize_identifier(paper.get(field))
        if value:
            return f"{prefix}:{value}"

    title = normalize_text(paper.get("title"))
    if title:
        return f"title:{title}"

    return f"object:{id(paper)}"


def raw_meta(paper: dict[str, Any]) -> dict[str, Any]:
    raw = paper.get("raw")
    return raw if isinstance(raw, dict) else {}


def paper_text(paper: dict[str, Any]) -> str:
    raw = raw_meta(paper)
    concepts = raw.get("concepts")
    concept_text = ""
    if isinstance(concepts, list):
        concept_text = " ".join(
            str(item.get("display_name", ""))
            for item in concepts
            if isinstance(item, dict)
        )

    authors = paper.get("authors")
    author_text = ""
    if isinstance(authors, list):
        author_text = " ".join(str(item) for item in authors)

    return normalize_text(
        " ".join(
            [
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
                str(paper.get("venue") or ""),
                concept_text,
                author_text,
            ]
        )
    )


def term_matches(term: str, text: str) -> bool:
    normalized = normalize_text(term)
    if not normalized:
        return False
    escaped = re.escape(normalized)
    if normalized.endswith("s"):
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    else:
        pattern = rf"(?<![a-z0-9]){escaped}s?(?![a-z0-9])"
    return re.search(pattern, text) is not None


def compute_constraint_coverage(
    paper: dict[str, Any],
    constraints: list[QueryConstraint],
) -> dict[str, Any]:
    text = paper_text(paper)
    covered: list[str] = []
    covered_types: list[str] = []
    details: list[dict[str, Any]] = []

    if not text:
        return {
            "covered_constraint_ids": [],
            "covered_constraint_types": [],
            "coverage_count": 0,
            "coverage_weight": 0.0,
            "core_coverage_count": 0,
            "details": [],
        }

    for constraint in constraints:
        terms = [term for term in constraint.terms if normalize_text(term)]
        if not terms:
            continue

        matches = [term for term in terms if term_matches(term, text)]
        if not matches:
            continue

        # Short constraints require all terms; longer constraints allow one missing term.
        required = len(terms) if len(terms) <= 2 else max(2, len(terms) - 1)
        if len(matches) >= required:
            covered.append(constraint.id)
            covered_types.append(constraint.constraint_type)
            details.append(
                {
                    "constraint_id": constraint.id,
                    "constraint_text": constraint.text,
                    "constraint_type": constraint.constraint_type,
                    "matched_terms": matches,
                    "required_terms": required,
                    "weight": constraint.weight,
                }
            )

    coverage_weight = sum(float(item.get("weight", 1.0)) for item in details)
    core_count = sum(1 for item in details if item.get("constraint_type") in CORE_TYPES)
    return {
        "covered_constraint_ids": covered,
        "covered_constraint_types": covered_types,
        "coverage_count": len(covered),
        "coverage_weight": coverage_weight,
        "core_coverage_count": core_count,
        "details": details,
    }


def has_strong_identifier(paper: dict[str, Any]) -> bool:
    return any(
        normalize_identifier(paper.get(field))
        for field in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id")
    )


def original_rank_score(rank: int) -> float:
    return 1.0 / (max(1, int(rank)) + 4)


def constraint_coverage_score(coverage: dict[str, Any], total_constraints: int) -> float:
    if total_constraints <= 0:
        return 0.0
    count_score = float(coverage.get("coverage_count", 0)) / total_constraints
    weight_score = float(coverage.get("coverage_weight", 0.0)) / max(1.0, total_constraints)
    core_score = min(1.0, float(coverage.get("core_coverage_count", 0)) / 2.0)
    return 0.50 * count_score + 0.35 * weight_score + 0.15 * core_score


def b1_support_score(paper: dict[str, Any]) -> float:
    raw = raw_meta(paper)
    occurrence = int(raw.get("b1_occurrence_count") or 1)
    anchor_rank = int(raw.get("b1_best_anchor_rank") or 999999)
    variant_types = set(raw.get("b1_variant_types") or [])

    score = 0.0
    if occurrence >= 2:
        score += min(0.05, 0.02 * (occurrence - 1))
    if anchor_rank <= 20:
        score += 0.03
    elif anchor_rank <= 50:
        score += 0.015
    if "raw" in variant_types:
        score += 0.015
    if "cleaned" in variant_types:
        score += 0.010
    return score


def constraint_rerank_score(
    *,
    paper: dict[str, Any],
    original_rank: int,
    coverage: dict[str, Any],
    total_constraints: int,
    config: ConstraintRerankConfig,
) -> float:
    rank_part = original_rank_score(original_rank)
    coverage_part = constraint_coverage_score(coverage, total_constraints)
    support_part = b1_support_score(paper)

    score = (
        config.rank_weight * rank_part
        + config.constraint_weight * coverage_part
        + config.occurrence_weight * support_part
    )

    if int(coverage.get("core_coverage_count", 0)) > 0:
        score += config.core_constraint_bonus

    if total_constraints > 0 and int(coverage.get("core_coverage_count", 0)) == 0:
        score -= config.missing_core_penalty

    if has_strong_identifier(paper):
        score += config.strong_id_bonus

    return score


def annotate_paper(
    paper: dict[str, Any],
    *,
    original_rank: int,
    decomposition: ConstraintDecomposition,
    config: ConstraintRerankConfig,
) -> dict[str, Any]:
    constraints = decomposition.constraints
    coverage = compute_constraint_coverage(paper, constraints)
    score = constraint_rerank_score(
        paper=paper,
        original_rank=original_rank,
        coverage=coverage,
        total_constraints=len(constraints),
        config=config,
    )

    annotated = dict(paper)
    raw = dict(raw_meta(annotated))
    raw["b2_1_original_rank"] = original_rank
    raw["b2_1_rerank_mode"] = config.mode
    raw["b2_1_constraint_score"] = score
    raw["b2_1_covered_constraints"] = coverage["covered_constraint_ids"]
    raw["b2_1_covered_constraint_types"] = coverage["covered_constraint_types"]
    raw["b2_1_coverage_count"] = coverage["coverage_count"]
    raw["b2_1_coverage_weight"] = coverage["coverage_weight"]
    raw["b2_1_core_coverage_count"] = coverage["core_coverage_count"]
    raw["b2_1_coverage_details"] = coverage["details"]
    annotated["raw"] = raw
    return annotated


def annotate_and_rerank_record(
    prediction_record: dict[str, Any],
    decomposition: ConstraintDecomposition,
    config: ConstraintRerankConfig,
) -> dict[str, Any]:
    papers = prediction_record.get("papers") or []
    if not isinstance(papers, list):
        papers = []

    annotated: list[dict[str, Any]] = []
    seen: set[str] = set()

    for rank, paper in enumerate(papers, start=1):
        if not isinstance(paper, dict):
            continue
        key = paper_key(paper)
        if key in seen:
            continue
        seen.add(key)
        annotated.append(
            annotate_paper(
                paper,
                original_rank=rank,
                decomposition=decomposition,
                config=config,
            )
        )

    annotated.sort(
        key=lambda item: (
            -float(raw_meta(item).get("b2_1_constraint_score", 0.0)),
            -int(raw_meta(item).get("b2_1_core_coverage_count", 0)),
            -int(raw_meta(item).get("b2_1_coverage_count", 0)),
            int(raw_meta(item).get("b2_1_original_rank", 999999)),
            normalize_text(item.get("title")),
        )
    )

    return {
        "qid": prediction_record.get("qid"),
        "question": prediction_record.get("question"),
        "papers": annotated,
        "rerank": {
            "strategy": "b2_1_constraint_coverage_light_rerank",
            "config": asdict(config),
            "constraint_count": len(decomposition.constraints),
        },
    }
