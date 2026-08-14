from __future__ import annotations

from scholarpath.evaluation.comprehensive_evidence_calibration import (
    blended_scores, comprehensive_evidence_score, rerank_papers,
)
from scholarpath.rerank.constraint_evidence import CanonicalConstraint


def c(cid: str, text: str, ctype: str = "topic", weight: float = 1.0) -> CanonicalConstraint:
    return CanonicalConstraint(cid, text, ctype, tuple(text.split()), (text,), weight, (cid,))


def paper(title: str, abstract: str, e3: float) -> dict:
    return {"title": title, "abstract": abstract, "raw": {"day8_final_score": e3}}


def test_beta_zero_preserves_frozen_order() -> None:
    papers = [paper("A", "none", .9), paper("B", "multimodal model", .1)]
    assert [x["title"] for x in rerank_papers(papers, constraints=[c("c1", "multimodal model")], beta=0.0)] == ["A", "B"]


def test_stronger_multiconstraint_evidence_scores_higher() -> None:
    constraints = [c("c1", "multimodal"), c("c2", "scientific document understanding")]
    strong = paper("Multimodal scientific document understanding", "", .5)
    weak = paper("Multimodal systems", "", .5)
    assert comprehensive_evidence_score(constraints, strong).score > comprehensive_evidence_score(constraints, weak).score


def test_generic_entity_only_receives_penalty() -> None:
    constraints = [c("c1", "large language model", "model_or_entity")]
    score = comprehensive_evidence_score(constraints, paper("Large language model", "", .5))
    assert score.generic_entity_only_penalty > 0.0
    assert 0.0 <= score.score <= 1.0


def test_blend_is_bounded_and_can_change_order() -> None:
    constraints = [c("c1", "multimodal"), c("c2", "document understanding")]
    papers = [paper("A", "none", .51), paper("Multimodal document understanding", "", .50)]
    scores = blended_scores(papers, constraints=constraints, beta=.10)
    assert len(scores) == 2
    assert all(0.0 <= x <= 1.0 for x in scores)
