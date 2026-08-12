from __future__ import annotations

from typing import Any

from scholarpath.paper.schema import PaperRecord
from scholarpath.retrieval.openalex import openalex_work_to_paper
from scholarpath.retrieval.citation_expansion import (
    CitationBudget,
    CitationExpansionConfig,
    OpenAlexCitationProvider,
)


def _paper_from_mapping(data: dict[str, Any]) -> PaperRecord:
    return PaperRecord.from_mapping(
        data,
        default_source="api",
    )


def build_citation_path(
    *,
    paper: dict[str, Any],
    qid: str,
    seed_rank: int,
) -> dict[str, Any] | None:
    """
    Build one user-facing citation edge.

    Failure should never affect search.
    """

    openalex_id = paper.get("openalex_id")

    if not openalex_id:
        return None

    try:
        seed = _paper_from_mapping(paper)

        provider = OpenAlexCitationProvider(
            config=CitationExpansionConfig(
                max_api_calls=12,
                max_references_per_seed=3,
                max_cited_by_per_seed=3,
            ),
            budget=CitationBudget(
                max_api_calls=3
            ),
        )

        references = provider.get_references(seed)

        if references:
            expanded = references[0]

            return {
                "seed_openalex_id": seed.openalex_id,
                "seed_title": seed.title,
                "expanded_openalex_id": expanded.openalex_id,
                "expanded_title": expanded.title,
                "edge_type": "references",
                "hop": 1,
            }


        cited = provider.get_cited_by(seed)

        if cited:
            expanded = cited[0]

            return {
                "seed_openalex_id": seed.openalex_id,
                "seed_title": seed.title,
                "expanded_openalex_id": expanded.openalex_id,
                "expanded_title": expanded.title,
                "edge_type": "cited_by",
                "hop": 1,
            }

    except Exception:
        return None

    return None