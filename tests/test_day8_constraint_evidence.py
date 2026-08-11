from __future__ import annotations

from scholarpath.query.constraints import ConstraintDecomposer, QueryConstraint
from scholarpath.rerank.constraint_evidence import (
    ConstraintDedupConfig,
    build_canonical_constraints,
    canonical_alias_text,
    deduplicate_constraints,
    normalize_constraint_text,
    normalize_evidence_text,
    recover_residual_constraint_spans,
    token_jaccard,
)


def make_constraint(
    cid: str,
    text: str,
    ctype: str,
    *,
    weight: float = 1.0,
) -> QueryConstraint:
    return QueryConstraint(
        id=cid,
        text=text,
        constraint_type=ctype,
        terms=text.split(),
        weight=weight,
    )


def test_normalize_evidence_text_handles_case_hyphen_and_plural() -> None:
    assert normalize_evidence_text("Graph-Neural Networks") == "graph neural network"
    assert normalize_evidence_text("MULTI\u2011MODAL   Models") == "multi modal model"


def test_boundary_connector_cleanup_is_phrase_only() -> None:
    assert normalize_constraint_text("graph neural networks for") == "graph neural network"
    assert normalize_constraint_text("for molecular property prediction") == "molecular property prediction"
    assert normalize_constraint_text("bag of words") == "bag of words"
    assert normalize_constraint_text("in context learning") == "in context learning"


def test_alias_normalization_merges_llm_variants() -> None:
    assert canonical_alias_text("LLMs") == "large language model"
    assert canonical_alias_text("Large Language Models") == "large language model"


def test_alias_normalization_merges_gnn_variants() -> None:
    assert canonical_alias_text("GNN") == "graph neural network"
    assert canonical_alias_text("Graph Neural Networks") == "graph neural network"


def test_boundary_cleanup_allows_cross_type_exact_merge() -> None:
    constraints = [
        make_constraint("c1", "graph neural network", "method_or_property", weight=1.25),
        make_constraint("c2", "graph neural networks for", "topic", weight=1.0),
    ]
    result = deduplicate_constraints(constraints)
    assert len(result) == 1
    assert result[0].canonical_text == "graph neural network"
    assert result[0].weight == 1.25
    assert set(result[0].source_constraint_ids) == {"c1", "c2"}


def test_deduplicate_exact_normalized_constraints() -> None:
    constraints = [
        make_constraint("c1", "Graph-Neural Networks", "method_or_property", weight=1.25),
        make_constraint("c2", "graph neural network", "method_or_property", weight=1.10),
    ]
    result = deduplicate_constraints(constraints)
    assert len(result) == 1
    assert result[0].canonical_text == "graph neural network"
    assert set(result[0].source_constraint_ids) == {"c1", "c2"}


def test_deduplicate_aliases_and_keep_max_weight_not_sum() -> None:
    constraints = [
        make_constraint("c1", "large language models", "model_or_entity", weight=1.10),
        make_constraint("c2", "LLM", "model_or_entity", weight=0.90),
        make_constraint("c3", "LLMs", "model_or_entity", weight=1.00),
    ]
    result = deduplicate_constraints(constraints)
    assert len(result) == 1
    assert result[0].canonical_text == "large language model"
    assert result[0].weight == 1.10
    assert result[0].weight != 3.0
    assert set(result[0].source_constraint_ids) == {"c1", "c2", "c3"}


def test_partial_overlap_is_not_wrongly_merged() -> None:
    constraints = [
        make_constraint("c1", "multimodal", "task_or_modality", weight=1.15),
        make_constraint("c2", "multimodal large language model", "task_or_modality", weight=1.15),
    ]
    result = deduplicate_constraints(constraints)
    assert len(result) == 2


def test_token_similarity_only_merges_same_type() -> None:
    constraints = [
        make_constraint("c1", "scientific document understanding", "topic"),
        make_constraint("c2", "scientific document understandings", "task_or_modality"),
    ]
    result = deduplicate_constraints(
        constraints,
        config=ConstraintDedupConfig(same_type_jaccard_threshold=0.80),
    )
    assert len(result) == 2


def test_high_jaccard_same_type_can_merge() -> None:
    constraints = [
        make_constraint("c1", "financial factor mining method", "topic", weight=1.0),
        make_constraint("c2", "financial factor mining methods", "topic", weight=1.2),
    ]
    result = deduplicate_constraints(constraints)
    assert len(result) == 1
    assert result[0].weight == 1.2


def test_residual_recovery_finds_scientific_document_understanding() -> None:
    query = (
        "recent papers on multimodal large language models "
        "for scientific document understanding"
    )
    decomposition = ConstraintDecomposer().decompose(query)
    canonical = build_canonical_constraints(
        question=query,
        constraints=decomposition.constraints,
    )
    assert [item.canonical_text for item in canonical] == [
        "large language model",
        "multimodal",
        "scientific document understanding",
    ]
    assert canonical[-1].constraint_type == "topic"
    assert canonical[-1].weight == 1.0
    assert canonical[-1].source_constraint_ids == ("residual_query_span_1",)


def test_gnn_query_has_only_two_final_canonical_constraints() -> None:
    query = "graph neural networks for molecular property prediction"
    decomposition = ConstraintDecomposer().decompose(query)
    canonical = build_canonical_constraints(
        question=query,
        constraints=decomposition.constraints,
    )
    assert [item.canonical_text for item in canonical] == [
        "graph neural network",
        "molecular property prediction",
    ]
    assert canonical[0].weight == 1.25


def test_residual_recovery_can_be_disabled() -> None:
    query = (
        "recent papers on multimodal large language models "
        "for scientific document understanding"
    )
    decomposition = ConstraintDecomposer().decompose(query)
    canonical = build_canonical_constraints(
        question=query,
        constraints=decomposition.constraints,
        config=ConstraintDedupConfig(recover_residual_constraints=False),
    )
    assert [item.canonical_text for item in canonical] == [
        "large language model",
        "multimodal",
    ]


def test_residual_span_does_not_duplicate_existing_concept() -> None:
    constraints = [make_constraint("c1", "large language model", "model_or_entity")]
    base = deduplicate_constraints(constraints)
    spans = recover_residual_constraint_spans(
        question="papers about large language models",
        canonical_constraints=base,
    )
    assert spans == []


def test_token_jaccard_is_deterministic() -> None:
    score = token_jaccard(
        {"multimodal", "large", "language", "model"},
        {"large", "language", "model"},
    )
    assert score == 0.75


def test_output_is_deterministic_and_json_ready() -> None:
    constraints = [
        make_constraint("c1", "LLM", "model_or_entity", weight=1.1),
        make_constraint("c2", "large language models", "model_or_entity", weight=1.0),
    ]
    result = deduplicate_constraints(constraints)
    payload = result[0].to_dict()
    assert payload["id"].startswith("cc1_")
    assert payload["canonical_text"] == "large language model"
    assert isinstance(payload["aliases"], list)
    assert isinstance(payload["source_constraint_ids"], list)
