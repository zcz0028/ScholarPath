from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from scholarpath.retrieval.base import SearchRequest
from scholarpath.retrieval.openalex import (
    OpenAlexConfig,
    OpenAlexError,
    OpenAlexRetriever,
    openalex_work_to_paper,
    sanitize_openalex_search_query,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.content = json.dumps(payload).encode("utf-8")

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        captured = {"url": url, **kwargs}
        if "params" in captured and isinstance(captured["params"], dict):
            captured["params"] = dict(captured["params"])
        self.calls.append(captured)
        return self.responses.pop(0)


def sample_payload() -> dict[str, Any]:
    return {
        "meta": {"count": 1},
        "results": [
            {
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1000/Example",
                "display_name": "A Retrieval Paper",
                "publication_year": 2024,
                "publication_date": "2024-01-15",
                "authorships": [
                    {"author": {"display_name": "Alice Smith"}}
                ],
                "primary_location": {
                    "landing_page_url": "https://arxiv.org/abs/2401.01234v2",
                    "source": {"display_name": "Synthetic Venue"},
                },
                "best_oa_location": None,
                "ids": {"openalex": "https://openalex.org/W123"},
                "cited_by_count": 5,
                "relevance_score": 12.5,
                "type": "article",
            }
        ],
    }


def test_work_mapping() -> None:
    paper = openalex_work_to_paper(sample_payload()["results"][0])
    assert paper.title == "A Retrieval Paper"
    assert paper.doi == "10.1000/example"
    assert paper.arxiv_id == "2401.01234"
    assert paper.openalex_id == "W123"
    assert paper.authors == ["Alice Smith"]
    assert paper.venue == "Synthetic Venue"


def test_search_builds_expected_request_and_cache(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(sample_payload())])
    retriever = OpenAlexRetriever(
        OpenAlexConfig(
            api_key="secret",
            cache_dir=str(tmp_path / "cache"),
            max_retries=0,
        ),
        session=session,
    )
    request = SearchRequest(
        query="raw complex query",
        per_page=100,
        to_publication_date="2024-09-24",
    )

    first = retriever.search(request)
    assert len(first.papers) == 1
    assert first.stats.actual_api_calls == 1
    assert first.stats.cache_hits == 0
    assert session.calls[0]["params"]["search"] == "raw complex query"
    assert (
        session.calls[0]["params"]["filter"]
        == "to_publication_date:2024-09-24"
    )

    second = retriever.search(request)
    assert len(second.papers) == 1
    assert second.stats.actual_api_calls == 0
    assert second.stats.cache_hits == 1
    assert len(session.calls) == 1


def test_missing_api_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    retriever = OpenAlexRetriever(
        OpenAlexConfig(api_key=None, cache_dir=str(tmp_path / "cache")),
        session=FakeSession([]),
    )
    with pytest.raises(OpenAlexError, match="OPENALEX_API_KEY"):
        retriever.search(SearchRequest(query="test"))


def test_retry_on_429(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(
                {"error": "rate limited"},
                status_code=429,
                headers={"Retry-After": "0"},
            ),
            FakeResponse(sample_payload()),
        ]
    )
    retriever = OpenAlexRetriever(
        OpenAlexConfig(
            api_key="secret",
            cache_dir=str(tmp_path / "cache"),
            max_retries=1,
        ),
        session=session,
        sleep_fn=lambda _: None,
    )
    result = retriever.search(SearchRequest(query="test"))
    assert result.stats.actual_api_calls == 2
    assert result.stats.retries == 1
    assert len(result.papers) == 1


def test_sanitize_openalex_search_query() -> None:
    raw = (
        "Is there any work that analyzes the scaling law of the "
        "multi-module models, such as video-text, image-text models?"
    )
    assert sanitize_openalex_search_query(raw) == (
        "Is there any work that analyzes the scaling law of the "
        "multi module models such as video text image text models"
    )


def test_400_bad_request_retries_once_with_sanitized_query(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse({"error": "bad request"}, status_code=400),
            FakeResponse(sample_payload(), status_code=200),
        ]
    )
    retriever = OpenAlexRetriever(
        OpenAlexConfig(
            api_key="secret",
            cache_dir=str(tmp_path / "cache"),
            max_retries=0,
        ),
        session=session,
        sleep_fn=lambda _: None,
    )
    request = SearchRequest(
        query=(
            "Is there any work that analyzes the scaling law of the "
            "multi-module models, such as video-text, image-text models?"
        )
    )

    result = retriever.search(request)

    assert len(result.papers) == 1
    assert result.stats.actual_api_calls == 2
    assert result.stats.retries == 1
    assert session.calls[0]["params"]["search"] == request.query
    assert session.calls[1]["params"]["search"] == (
        "Is there any work that analyzes the scaling law of the "
        "multi module models such as video text image text models"
    )
