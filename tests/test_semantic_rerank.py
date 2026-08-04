from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.semantic_rerank import (
    annotate_and_semantic_rerank_record,
    compute_semantic_features,
    make_semantic_config,
    tokenize,
)


def test_tokenize_removes_prompt_noise() -> None:
    tokens = tokenize("Give me papers about large language model pre-training")
    assert "give" not in tokens
    assert "papers" not in tokens
    assert "language" in tokens
    assert "model" in tokens


def test_semantic_features_score_relevant_title_higher_than_generic() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "papers about scaling law of video text models"
    )
    config = make_semantic_config("hybrid")

    relevant = {
        "title": "Scaling Laws for Video Text Models",
        "abstract": "We study scaling behavior in video text multimodal models.",
        "arxiv_id": "2401.00001",
    }
    generic = {
        "title": "A Survey of Neural Networks",
        "abstract": "This survey discusses neural methods.",
        "arxiv_id": "2401.00002",
    }

    relevant_features = compute_semantic_features(
        paper=relevant,
        question="papers about scaling law of video text models",
        decomposition=decomposition,
        original_rank=2,
        config=config,
    )
    generic_features = compute_semantic_features(
        paper=generic,
        question="papers about scaling law of video text models",
        decomposition=decomposition,
        original_rank=1,
        config=config,
    )

    assert relevant_features.semantic_score > generic_features.semantic_score


def test_semantic_rerank_can_promote_relevant_candidate() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "papers about scaling law of video text models"
    )
    record = {
        "qid": "q1",
        "question": "papers about scaling law of video text models",
        "papers": [
            {
                "title": "A Survey of Neural Networks",
                "abstract": "This survey discusses neural methods.",
                "arxiv_id": "2401.00002",
            },
            {
                "title": "Scaling Laws for Video Text Models",
                "abstract": "We study scaling behavior in video text multimodal models.",
                "arxiv_id": "2401.00001",
            },
        ],
    }

    reranked = annotate_and_semantic_rerank_record(
        record,
        decomposition,
        make_semantic_config("semantic_first"),
    )

    assert reranked["papers"][0]["title"] == "Scaling Laws for Video Text Models"
    assert "b3_features" in reranked["papers"][0]["raw"]
    assert "b3_final_score" in reranked["papers"][0]["raw"]
