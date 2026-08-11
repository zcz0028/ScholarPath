from __future__ import annotations

from scholarpath.rerank.constraint_evidence import (
    CanonicalConstraint,
    EvidenceMatcherConfig,
    build_constraint_evidence,
    match_constraint_evidence,
)


def cc(
    text: str,
    *,
    cid: str = "cc1",
    ctype: str = "topic",
    aliases: tuple[str, ...] = (),
) -> CanonicalConstraint:
    return CanonicalConstraint(
        id=cid,
        canonical_text=text,
        constraint_type=ctype,
        terms=tuple(text.split()),
        aliases=aliases,
        weight=1.0,
        source_constraint_ids=("c1",),
    )


def test_normalized_title_phrase_match() -> None:
    ev = build_constraint_evidence(
        [cc("graph neural network")],
        {"title": "A Review of Graph Neural Networks"},
    )[0]
    assert ev.matched is True
    assert ev.match_type == "title_phrase_match"
    assert ev.evidence_field == "title"
    assert ev.confidence == 1.0


def test_hyphen_case_plural_title_match() -> None:
    ev = build_constraint_evidence(
        [cc("graph neural network")],
        {"title": "GRAPH-NEURAL NETWORKS for Chemistry"},
    )[0]
    assert ev.matched is True
    assert ev.match_type == "title_phrase_match"


def test_canonical_phrase_precedes_alias() -> None:
    constraint = cc(
        "large language model",
        aliases=("llm", "large language model"),
    )
    ev = build_constraint_evidence(
        [constraint],
        {"title": "Large Language Models (LLMs) for Science"},
    )[0]
    assert ev.match_type == "title_phrase_match"
    assert ev.evidence_text == "large language model"


def test_alias_title_match() -> None:
    constraint = cc("large language model", aliases=("llms",))
    ev = build_constraint_evidence(
        [constraint],
        {"title": "LLMs for Scientific Document Analysis"},
    )[0]
    assert ev.matched is True
    assert ev.match_type == "title_alias_match"
    assert ev.evidence_text == "llms"


def test_abstract_phrase_match() -> None:
    ev = build_constraint_evidence(
        [cc("molecular property prediction")],
        {
            "title": "A Chemistry Benchmark",
            "abstract": "We study molecular property prediction at scale.",
        },
    )[0]
    assert ev.match_type == "abstract_phrase_match"
    assert ev.evidence_field == "abstract"


def test_empty_abstract_title_still_matches() -> None:
    ev = build_constraint_evidence(
        [cc("graph neural network")],
        {"title": "Graph Neural Networks in Chemistry", "abstract": ""},
    )[0]
    assert ev.matched is True
    assert ev.match_type == "title_phrase_match"


def test_empty_abstract_unrelated_paper_returns_false_without_error() -> None:
    result = build_constraint_evidence(
        [cc("graph neural network"), cc("molecular property prediction", cid="cc2")],
        {"title": "Unrelated Paper", "abstract": "", "raw": {"concepts": []}},
    )
    assert len(result) == 2
    assert all(item.matched is False for item in result)
    assert all(item.match_type == "none" for item in result)


def test_openalex_inverted_abstract_is_reconstructed() -> None:
    paper = {
        "title": "Chemistry Study",
        "abstract": None,
        "raw": {
            "abstract_inverted_index": {
                "We": [0],
                "use": [1],
                "graph": [2],
                "neural": [3],
                "networks": [4],
            }
        },
    }
    ev = build_constraint_evidence([cc("graph neural network")], paper)[0]
    assert ev.matched is True
    assert ev.match_type == "abstract_phrase_match"


def test_concepts_use_display_name() -> None:
    paper = {
        "title": "Unrelated title",
        "raw": {"concepts": [{"display_name": "Graph neural network"}]},
    }
    ev = build_constraint_evidence([cc("graph neural network")], paper)[0]
    assert ev.matched is True
    assert ev.match_type == "concept_phrase_match"
    assert ev.evidence_field == "concept"


def test_concepts_are_not_joined_across_items() -> None:
    paper = {
        "title": "Unrelated title",
        "raw": {
            "concepts": [
                {"display_name": "Scientific document"},
                {"display_name": "Understanding"},
            ]
        },
    }
    ev = build_constraint_evidence([cc("scientific document understanding")], paper)[0]
    assert ev.matched is False


def test_token_reordered_full_coverage_matches() -> None:
    ev = build_constraint_evidence(
        [cc("scientific document understanding")],
        {"title": "Understanding Scientific Documents with Vision Models"},
    )[0]
    assert ev.matched is True
    assert ev.match_type == "title_token_match"
    assert ev.token_coverage == 1.0


def test_two_token_constraint_requires_two_of_two() -> None:
    ev = build_constraint_evidence(
        [cc("factor mining")],
        {"title": "Factor Models for Finance"},
    )[0]
    assert ev.matched is False


def test_three_token_constraint_two_of_three_does_not_match() -> None:
    ev = build_constraint_evidence(
        [cc("scientific document understanding")],
        {"title": "Scientific Document Retrieval"},
    )[0]
    assert ev.matched is False


def test_five_token_constraint_four_of_five_matches() -> None:
    ev = build_constraint_evidence(
        [cc("multimodal scientific document question answering")],
        {"title": "Multimodal Scientific Document Answering Systems"},
    )[0]
    assert ev.matched is True
    assert ev.match_type == "title_token_match"
    assert ev.token_coverage == 0.8


def test_generic_single_token_is_rejected() -> None:
    ev = build_constraint_evidence(
        [cc("network")],
        {"title": "Network Analysis for Biology"},
    )[0]
    assert ev.matched is False
    assert ev.match_type == "none"


def test_generic_single_token_does_not_create_token_fallback() -> None:
    # Directly exercise a field where no canonical phrase exists after token
    # normalization is impossible for one token; therefore use a boundary-safe
    # negative case to guarantee no fallback broadening occurs.
    ev = build_constraint_evidence(
        [cc("analysis")],
        {"title": "Analytical Methods for Biology"},
    )[0]
    assert ev.matched is False


def test_distinctive_single_token_can_match() -> None:
    ev = build_constraint_evidence(
        [cc("multimodal")],
        {"title": "A Multimodal Benchmark"},
    )[0]
    assert ev.matched is True
    assert ev.match_type == "title_phrase_match"


def test_unmatched_constraint_is_retained() -> None:
    ev = build_constraint_evidence(
        [cc("scientific document understanding")],
        {"title": "A Survey of Large Language Models"},
    )[0]
    assert ev.matched is False
    assert ev.match_type == "none"
    assert ev.evidence_field is None
    assert ev.evidence_text is None
    assert ev.confidence == 0.0


def test_title_beats_abstract() -> None:
    paper = {
        "title": "Graph Neural Networks for Molecules",
        "abstract": "Graph neural networks are widely studied.",
    }
    ev = build_constraint_evidence([cc("graph neural network")], paper)[0]
    assert ev.match_type == "title_phrase_match"


def test_abstract_beats_concept() -> None:
    paper = {
        "title": "Chemistry Study",
        "abstract": "We use graph neural networks.",
        "raw": {"concepts": [{"display_name": "Graph neural network"}]},
    }
    ev = build_constraint_evidence([cc("graph neural network")], paper)[0]
    assert ev.match_type == "abstract_phrase_match"


def test_empty_constraints_returns_empty_list() -> None:
    assert build_constraint_evidence([], {"title": "Anything"}) == []


def test_empty_paper_mapping_is_safe() -> None:
    ev = build_constraint_evidence([cc("graph neural network")], {})[0]
    assert ev.matched is False


def test_invalid_concept_items_are_ignored() -> None:
    paper = {"raw": {"concepts": [None, "bad", {}, {"display_name": None}]}}
    ev = build_constraint_evidence([cc("graph neural network")], paper)[0]
    assert ev.matched is False


def test_long_threshold_is_configurable_and_integer_safe() -> None:
    cfg = EvidenceMatcherConfig(long_constraint_token_coverage=0.8)
    ev = match_constraint_evidence(
        cc("one two three four five"),
        title="one two three four",
        config=cfg,
    )
    assert ev.matched is True
    assert ev.token_coverage == 0.8
