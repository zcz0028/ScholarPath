from __future__ import annotations

import math

from scholarpath.query.constraints import ConstraintDecomposition, QueryConstraint
from scholarpath.rerank.constraint_evidence import CanonicalConstraint, ConstraintEvidence
from scholarpath.rerank.evidence_aware_rerank import (
    EvidenceAwareConfig,
    annotate_and_evidence_rerank_record,
    compute_pure_semantic_base,
    dcg_at_k,
    deterministic_folds,
    minmax_normalize,
    ndcg_at_k,
    raw_constraints_for_evidence,
    select_alpha_from_cv,
    weighted_constraint_coverage,
)


def qc(cid: str, text: str, weight: float = 1.0) -> QueryConstraint:
    return QueryConstraint(cid, text, "topic", text.split(), weight)


def cc(cid: str, text: str, weight: float = 1.0) -> CanonicalConstraint:
    return CanonicalConstraint(
        id=cid,
        canonical_text=text,
        constraint_type="topic",
        terms=tuple(text.split()),
        aliases=(text,),
        weight=weight,
        source_constraint_ids=(cid,),
    )


def ev(cid: str, matched: bool) -> ConstraintEvidence:
    return ConstraintEvidence(
        constraint_id=cid,
        constraint_text=cid,
        constraint_type="topic",
        matched=matched,
        match_type="title_phrase_match" if matched else "none",
        evidence_field="title" if matched else None,
        evidence_text=cid if matched else None,
        confidence=1.0 if matched else 0.0,
        token_coverage=1.0 if matched else 0.0,
    )


def decomposition(question: str, constraints: list[QueryConstraint]) -> ConstraintDecomposition:
    return ConstraintDecomposition(question, question, constraints, [])


def test_coverage_empty_constraints_is_zero() -> None:
    result = weighted_constraint_coverage([], [])
    assert result.coverage == 0.0
    assert result.total_weight == 0.0


def test_coverage_zero_weight_denominator_is_safe() -> None:
    result = weighted_constraint_coverage([cc("c1", "alpha", 0.0)], [ev("c1", True)])
    assert result.coverage == 0.0


def test_coverage_counts_only_matched_true() -> None:
    constraints = [cc("c1", "alpha", 1.0), cc("c2", "beta", 3.0)]
    result = weighted_constraint_coverage(constraints, [ev("c1", True), ev("c2", False)])
    assert result.matched_weight == 1.0
    assert result.total_weight == 4.0
    assert result.coverage == 0.25


def test_unknown_evidence_id_does_not_affect_coverage() -> None:
    result = weighted_constraint_coverage([cc("c1", "alpha", 1.0)], [ev("other", True)])
    assert result.coverage == 0.0


def test_duplicate_evidence_does_not_double_count_weight() -> None:
    result = weighted_constraint_coverage([cc("c1", "alpha", 2.0)], [ev("c1", True), ev("c1", True)])
    assert result.coverage == 1.0
    assert result.matched_weight == 2.0


def test_raw_constraints_preserve_duplicates_for_e1() -> None:
    raw = raw_constraints_for_evidence([
        qc("c1", "large language model", 1.1),
        qc("c2", "large language models", 1.1),
    ])
    assert len(raw) == 2
    assert raw[0].id != raw[1].id


def test_minmax_normalization() -> None:
    assert minmax_normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_minmax_equal_scores_returns_neutral_half() -> None:
    assert minmax_normalize([3.0, 3.0]) == [0.5, 0.5]


def test_minmax_empty_is_empty() -> None:
    assert minmax_normalize([]) == []


def test_pure_semantic_base_is_bounded() -> None:
    d = decomposition("graph neural network molecular property prediction", [])
    features = compute_pure_semantic_base(
        paper={"title": "Graph neural networks for molecular property prediction", "abstract": ""},
        question=d.question,
        decomposition=d,
    )
    assert 0.0 <= features.base_score <= 1.0


def test_pure_semantic_base_does_not_read_b2_or_b3_scores() -> None:
    d = decomposition("graph neural network", [])
    paper_a = {"title": "Graph neural network", "raw": {"b2_1_constraint_score": 0.0, "b3_final_score": 0.1}}
    paper_b = {"title": "Graph neural network", "raw": {"b2_1_constraint_score": 99.0, "b3_final_score": 99.0}}
    a = compute_pure_semantic_base(paper=paper_a, question=d.question, decomposition=d)
    b = compute_pure_semantic_base(paper=paper_b, question=d.question, decomposition=d)
    assert a.base_score == b.base_score


def test_e0_preserves_b3_order() -> None:
    d = decomposition("graph neural network", [qc("c1", "graph neural network")])
    record = {
        "question": d.question,
        "papers": [
            {"title": "low", "raw": {"b3_final_score": 0.2}},
            {"title": "high", "raw": {"b3_final_score": 0.8}},
        ],
    }
    result = annotate_and_evidence_rerank_record(record, d, EvidenceAwareConfig("E0", 1.0))
    assert [p["title"] for p in result["papers"]] == ["low", "high"]


def test_e2_can_promote_higher_canonical_coverage() -> None:
    d = decomposition(
        "multimodal large language model scientific document understanding",
        [
            QueryConstraint(
                "c1",
                "large language model",
                "model_or_entity",
                ["large", "language", "model"],
                1.1,
            ),
            QueryConstraint(
                "c2",
                "multimodal",
                "task_or_modality",
                ["multimodal"],
                1.15,
            ),
            QueryConstraint(
                "c3",
                "scientific document understanding",
                "topic",
                ["scientific", "document", "understanding"],
                1.0,
            ),
        ],
    )

    record = {
        "question": d.question,
        "papers": [
            {
                "title": "A Survey of Large Language Models",
                "abstract": "",
                "raw": {
                    "b3_final_score": 0.60,
                },
            },
            {
                "title": "Multimodal Large Language Models",
                "abstract": "",
                "raw": {
                    "b3_final_score": 0.60,
                },
            },
        ],
    }

    result = annotate_and_evidence_rerank_record(
        record,
        d,
        EvidenceAwareConfig("E2", 0.7),
    )

    assert len(result["papers"]) == 2

    # 两篇论文拥有完全相同的 B3 baseline。
    # 因此 E2 的排序差异只能来自 canonical constraint coverage。
    assert (
        result["papers"][0]["title"]
        == "Multimodal Large Language Models"
    )

    higher = result["papers"][0]["raw"]
    lower = result["papers"][1]["raw"]

    assert (
        higher["day8_canonical_constraint_coverage"]
        > lower["day8_canonical_constraint_coverage"]
    )

    assert (
        higher["day8_final_score"]
        > lower["day8_final_score"]
    )

def test_day8_debug_fields_are_written() -> None:
    d = decomposition("graph neural network", [qc("c1", "graph neural network")])
    result = annotate_and_evidence_rerank_record(
        {"question": d.question, "papers": [{"title": "Graph neural network", "raw": {"b3_final_score": 0.5}}]},
        d,
        EvidenceAwareConfig("E3", 0.7),
    )
    raw = result["papers"][0]["raw"]
    assert raw["day8_variant"] == "E3"
    assert raw["day8_alpha"] == 0.7
    assert raw["day8_beta"] == pytest_approx(0.3)
    assert isinstance(raw["day8_constraint_evidence"], list)


def pytest_approx(value: float, tol: float = 1e-12):
    class Approx:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, (int, float)) and abs(float(other) - value) <= tol
    return Approx()


def test_dcg_known_value() -> None:
    expected = 1.0 + 1.0 / math.log2(4.0)
    assert abs(dcg_at_k([1, 0, 1], 3) - expected) < 1e-12


def test_ndcg_perfect_is_one() -> None:
    assert ndcg_at_k([1, 1, 0, 0], 10) == 1.0


def test_ndcg_zero_ideal_is_zero() -> None:
    assert ndcg_at_k([0, 0, 0], 10) == 0.0


def test_ndcg_rewards_earlier_relevant_paper() -> None:
    assert ndcg_at_k([1, 0, 0], 10) > ndcg_at_k([0, 0, 1], 10)


def test_deterministic_folds_cover_each_qid_once() -> None:
    qids = [f"q{i}" for i in range(10)]
    folds = deterministic_folds(qids, 5)
    flattened = [qid for fold in folds for qid in fold]
    assert sorted(flattened) == sorted(qids)
    assert len(flattened) == len(set(flattened))


def test_cv_tie_prefers_larger_alpha_smaller_beta() -> None:
    qids = [f"q{i}" for i in range(10)]
    scores = {
        0.9: {qid: 0.700 for qid in qids},
        0.8: {qid: 0.704 for qid in qids},
    }
    selected = select_alpha_from_cv(scores, qids, folds=5, tie_tolerance=0.005)
    assert selected["selected_alpha"] == 0.9


def test_cv_selects_clear_ndcg_winner() -> None:
    qids = [f"q{i}" for i in range(10)]
    scores = {
        0.9: {qid: 0.70 for qid in qids},
        0.7: {qid: 0.80 for qid in qids},
    }
    selected = select_alpha_from_cv(scores, qids, folds=5, tie_tolerance=0.005)
    assert selected["selected_alpha"] == 0.7
