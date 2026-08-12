from __future__ import annotations

from unittest.mock import patch
from scholarpath.paper.schema import PaperRecord
from scholarpath.retrieval.citation_path_builder import (
    build_citation_path,
)


def test_empty_openalex_returns_none():
    """
    没有 OpenAlex ID 时，不触发 citation expansion
    """

    result = build_citation_path(
        paper={
            "title": "test paper",
        },
        qid="q1",
        seed_rank=1,
    )

    assert result is None


def test_citation_api_failure_is_safe():
    """
    citation expansion 失败不能影响搜索
    """

    with patch(
        "scholarpath.retrieval.citation_path_builder.OpenAlexCitationProvider"
    ) as mock_provider:

        instance = mock_provider.return_value

        instance.get_references.side_effect = RuntimeError(
            "OpenAlex unavailable"
        )

        result = build_citation_path(
            paper={
                "title": "seed paper",
                "openalex_id": "W123456",
            },
            qid="q1",
            seed_rank=1,
        )

    assert result is None


def test_reference_citation_path_created():
    """
    正常生成 citation path
    """

    fake_reference = PaperRecord(
        title="Referenced Paper",
        openalex_id="W999",
        source="test",
    )

    with patch(
        "scholarpath.retrieval.citation_path_builder.OpenAlexCitationProvider"
    ) as mock_provider:

        instance = mock_provider.return_value

        instance.get_references.return_value = [
            fake_reference
        ]

        result = build_citation_path(
            paper={
                "title": "Seed Paper",
                "openalex_id": "W123",
            },
            qid="q1",
            seed_rank=1,
        )

    assert result is not None

    assert result["seed_openalex_id"] == "W123"

    assert result["expanded_openalex_id"] == "W999"

    assert result["expanded_title"] == "Referenced Paper"

    assert result["edge_type"] == "references"

    assert result["hop"] == 1