from scholarpath.query.constraints import ConstraintDecomposer


def test_decomposer_extracts_scaling_law_multimodal_constraints() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "Is there any work that analyzes the scaling law of the multi-module models, such as video-text, image-text models?"
    )
    texts = " ".join(item.text for item in decomposition.constraints).casefold()
    assert "scaling law" in texts
    assert "video text" in texts
    assert "image text" in texts
    assert decomposition.subqueries
    assert any(item.subquery_type == "pairwise_constraint" for item in decomposition.subqueries)


def test_decomposer_has_keyword_fallback() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "papers about robust neural ranking for scientific literature search"
    )
    assert decomposition.constraints
    assert decomposition.subqueries
    assert any(item.subquery_type == "keyword_fallback" for item in decomposition.subqueries)


def test_decomposer_limits_subqueries() -> None:
    decomposition = ConstraintDecomposer(max_subqueries=3).decompose(
        "scaling law multimodal model video text image text retrieval augmented generation"
    )
    assert len(decomposition.subqueries) <= 3


def test_decomposer_extracts_dataset_size_comparison_constraints() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets."
    )
    texts = " ".join(item.text for item in decomposition.constraints).casefold()
    types = {item.constraint_type for item in decomposition.constraints}

    assert "large language model" in texts
    assert "smaller dataset" in texts
    assert "bigger dataset" in texts
    assert "better models" in texts
    assert "pre training" in texts or "pretraining" in texts
    assert "data_condition" in types
    assert "performance_relation" in types
    assert "give me" not in texts
