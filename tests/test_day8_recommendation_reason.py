from __future__ import annotations

from scholarpath.rerank.constraint_evidence import CanonicalConstraint, ConstraintEvidence
from scholarpath.rerank.recommendation_reason import (
    attach_recommendation_reason,
    build_recommendation_reason,
)


def c(cid: str, text: str, ctype: str) -> CanonicalConstraint:
    return CanonicalConstraint(
        id=cid,
        canonical_text=text,
        constraint_type=ctype,
        terms=tuple(text.split()),
        aliases=(text,),
        weight=1.0,
        source_constraint_ids=(cid,),
    )


def e(
    cid: str,
    text: str,
    ctype: str,
    matched: bool,
    field: str | None = None,
    match_type: str = "none",
    confidence: float = 0.0,
) -> ConstraintEvidence:
    return ConstraintEvidence(
        constraint_id=cid,
        constraint_text=text,
        constraint_type=ctype,
        matched=matched,
        match_type=match_type,
        evidence_field=field,
        evidence_text=text if matched else None,
        confidence=confidence,
        token_coverage=1.0 if matched else 0.0,
    )


def test_empty_constraints_are_safe() -> None:
    r = build_recommendation_reason(constraints=[], evidence=[])
    assert r.reason_text == ""
    assert r.reason_tags == ()
    assert r.matched_count == 0
    assert r.constraint_count == 0


def test_all_unmatched_does_not_claim_relevance() -> None:
    constraints = [c("cc1", "multimodal", "task_or_modality")]
    r = build_recommendation_reason(
        constraints=constraints,
        evidence=[e("cc1", "multimodal", "task_or_modality", False)],
    )
    assert r.reason_tags == ()
    assert r.reason_text == "No verifiable constraint-matching evidence was found."
    assert r.matched_constraints == ()
    assert r.unmatched_constraints == ("multimodal",)


def test_single_title_match_builds_machine_tags_and_text() -> None:
    constraints = [c("cc1", "large language model", "model_or_entity")]
    r = build_recommendation_reason(
        constraints=constraints,
        evidence=[e("cc1", "large language model", "model_or_entity", True, "title", "title_phrase_match", 1.0)],
    )
    assert r.reason_tags == ("model_or_entity_match", "title_evidence")
    assert r.matched_constraints == ("large language model",)
    assert "large language model" in r.reason_text
    assert "paper title" in r.reason_text


def test_multiple_constraints_and_sources_have_stable_order() -> None:
    constraints = [
        c("cc1", "large language model", "model_or_entity"),
        c("cc2", "multimodal", "task_or_modality"),
        c("cc3", "document understanding", "topic"),
    ]
    evidence = [
        e("cc3", "document understanding", "topic", True, "concept", "concept_phrase_match", .75),
        e("cc2", "multimodal", "task_or_modality", True, "abstract", "abstract_phrase_match", .9),
        e("cc1", "large language model", "model_or_entity", True, "title", "title_phrase_match", 1.0),
    ]
    r = build_recommendation_reason(constraints=constraints, evidence=evidence)
    assert r.reason_tags == (
        "model_or_entity_match",
        "task_or_modality_match",
        "topic_match",
        "title_evidence",
        "abstract_evidence",
        "concept_evidence",
    )
    assert r.matched_count == 3
    assert r.constraint_count == 3


def test_matched_and_unmatched_are_kept_separate() -> None:
    constraints = [
        c("cc1", "large language model", "model_or_entity"),
        c("cc2", "multimodal", "task_or_modality"),
        c("cc3", "scientific document understanding", "topic"),
    ]
    evidence = [
        e("cc1", "large language model", "model_or_entity", True, "title", "title_phrase_match", 1.0),
        e("cc2", "multimodal", "task_or_modality", True, "title", "title_phrase_match", 1.0),
        e("cc3", "scientific document understanding", "topic", False),
    ]
    r = build_recommendation_reason(constraints=constraints, evidence=evidence)
    assert r.matched_count == 2
    assert r.constraint_count == 3
    assert r.unmatched_constraints == ("scientific document understanding",)
    assert "scientific document understanding" not in r.reason_text


def test_unknown_evidence_id_is_ignored() -> None:
    constraints = [c("cc1", "multimodal", "task_or_modality")]
    evidence = [
        e("cc999", "invented constraint", "topic", True, "title", "title_phrase_match", 1.0),
    ]
    r = build_recommendation_reason(constraints=constraints, evidence=evidence)
    assert r.matched_count == 0
    assert "invented constraint" not in r.reason_text


def test_duplicate_evidence_does_not_duplicate_reason() -> None:
    constraints = [c("cc1", "multimodal", "task_or_modality")]
    evidence = [
        e("cc1", "multimodal", "task_or_modality", False),
        e("cc1", "multimodal", "task_or_modality", True, "abstract", "abstract_phrase_match", .9),
        e("cc1", "multimodal", "task_or_modality", True, "title", "title_phrase_match", 1.0),
    ]
    r = build_recommendation_reason(constraints=constraints, evidence=evidence)
    assert r.matched_count == 1
    assert r.reason_tags == ("task_or_modality_match", "title_evidence")


def test_reason_does_not_present_confidence_as_probability() -> None:
    constraints = [c("cc1", "multimodal", "task_or_modality")]
    r = build_recommendation_reason(
        constraints=constraints,
        evidence=[e("cc1", "multimodal", "task_or_modality", True, "abstract", "abstract_token_match", .82)],
    )
    assert "82" not in r.reason_text
    assert "%" not in r.reason_text
    assert "confidence" not in r.reason_text.lower()


def test_method_and_topic_tags() -> None:
    constraints = [
        c("cc1", "graph neural network", "method_or_property"),
        c("cc2", "molecular property prediction", "topic"),
    ]
    evidence = [
        e("cc1", "graph neural network", "method_or_property", True, "title", "title_phrase_match", 1.0),
        e("cc2", "molecular property prediction", "topic", True, "title", "title_phrase_match", 1.0),
    ]
    r = build_recommendation_reason(constraints=constraints, evidence=evidence)
    assert r.reason_tags == ("method_or_property_match", "topic_match", "title_evidence")
    assert r.matched_count == 2


def test_attach_preserves_paper_and_existing_raw() -> None:
    paper = {"title": "Paper A", "raw": {"existing": 7}}
    constraints = [c("cc1", "multimodal", "task_or_modality")]
    evidence = [e("cc1", "multimodal", "task_or_modality", True, "title", "title_phrase_match", 1.0)]
    out = attach_recommendation_reason(paper, constraints=constraints, evidence=evidence)
    assert out["title"] == "Paper A"
    assert out["raw"]["existing"] == 7
    assert out["raw"]["day8_matched_count"] == 1
    assert paper["raw"] == {"existing": 7}
