from __future__ import annotations

from scholarpath.rerank.constraint_evidence import _normalized_phrase_in_text


def test_phrase_match_normalizes_case_hyphen_and_plural() -> None:
    assert _normalized_phrase_in_text(
        "graph neural network",
        "A Review of Graph-Neural Networks for Chemistry",
    )


def test_phrase_match_requires_token_sequence_boundary() -> None:
    assert _normalized_phrase_in_text("model", "a model for reasoning")
    assert not _normalized_phrase_in_text("model", "modeling behavior")
    assert not _normalized_phrase_in_text("net", "network analysis")


def test_phrase_match_requires_contiguous_normalized_phrase() -> None:
    assert _normalized_phrase_in_text(
        "large language model",
        "Recent advances in large-language models",
    )
    assert not _normalized_phrase_in_text(
        "large language model",
        "large multimodal language model",
    )


def test_phrase_match_handles_empty_values() -> None:
    assert not _normalized_phrase_in_text("", "some title")
    assert not _normalized_phrase_in_text("graph neural network", "")
    assert not _normalized_phrase_in_text(None, None)


def test_phrase_match_handles_long_abstract_without_regex_pathology() -> None:
    long_abstract = ("background evidence " * 20000) + (
        "multimodal large language models for scientific documents"
    )
    assert _normalized_phrase_in_text(
        "large language model",
        long_abstract,
    )
