from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

from scholarpath.paper.schema import PaperRecord
from scholarpath.query.constraints import QueryConstraint
from scholarpath.retrieval.aggregation import candidate_key


@dataclass(slots=True)
class ConstraintHit:
    paper: PaperRecord
    score: float
    best_rank: int
    occurrence_count: int = 1
    covered_constraints: set[str] = field(default_factory=set)
    subquery_hits: list[dict[str, Any]] = field(default_factory=list)

    def add_hit(
        self,
        *,
        paper: PaperRecord,
        rank: int,
        query_text: str,
        subquery_type: str,
        constraint_ids: list[str],
        score: float,
    ) -> None:
        self.score += score
        self.best_rank = min(self.best_rank, max(1, int(rank)))
        self.occurrence_count += 1
        self.covered_constraints.update(constraint_ids)
        self.subquery_hits.append(
            {
                "query": query_text,
                "subquery_type": subquery_type,
                "rank": max(1, int(rank)),
                "constraint_ids": constraint_ids,
            }
        )
        self._merge_metadata(paper)

    def _merge_metadata(self, paper: PaperRecord) -> None:
        if not self.paper.doi and paper.doi:
            self.paper.doi = paper.doi
        if not self.paper.arxiv_id and paper.arxiv_id:
            self.paper.arxiv_id = paper.arxiv_id
        if not self.paper.openalex_id and paper.openalex_id:
            self.paper.openalex_id = paper.openalex_id
        if not self.paper.semantic_scholar_id and paper.semantic_scholar_id:
            self.paper.semantic_scholar_id = paper.semantic_scholar_id
        if not self.paper.abstract and paper.abstract:
            self.paper.abstract = paper.abstract
        if len(paper.authors) > len(self.paper.authors):
            self.paper.authors = list(paper.authors)
        if self.paper.citation_count is None and paper.citation_count is not None:
            self.paper.citation_count = paper.citation_count


def normalize_text(text: object | None) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def paper_text(paper: PaperRecord) -> str:
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    concepts = raw.get("concepts")
    concept_text = ""
    if isinstance(concepts, list):
        concept_text = " ".join(
            str(item.get("display_name", ""))
            for item in concepts
            if isinstance(item, dict)
        )
    return normalize_text(
        " ".join(
            [
                paper.title or "",
                paper.abstract or "",
                paper.venue or "",
                concept_text,
            ]
        )
    )


def textual_constraint_coverage(
    paper: PaperRecord,
    constraints: list[QueryConstraint],
) -> set[str]:
    text = paper_text(paper)
    covered: set[str] = set()
    if not text:
        return covered

    for constraint in constraints:
        terms = [normalize_text(term) for term in constraint.terms if normalize_text(term)]
        if not terms:
            continue

        # Require all terms for short constraints, otherwise most terms.
        matches = 0
        for term in terms:
            escaped = re.escape(term)
            if term.endswith("s"):
                pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
            else:
                # Simple plural tolerance: "law" should match "law" and "laws".
                pattern = rf"(?<![a-z0-9]){escaped}s?(?![a-z0-9])"
            if re.search(pattern, text):
                matches += 1
        required = len(terms) if len(terms) <= 2 else max(2, len(terms) - 1)
        if matches >= required:
            covered.add(constraint.id)
    return covered


def subquery_rank_score(rank: int, subquery_type: str) -> float:
    weight = {
        "full_cleaned": 1.00,
        "constraint_core": 1.10,
        "pairwise_constraint": 0.95,
        "single_constraint": 0.85,
        "keyword_fallback": 0.70,
    }.get(subquery_type, 0.60)
    return weight / (max(1, int(rank)) + 5)


def aggregate_constraint_hits(
    hits: Iterable[tuple[PaperRecord, int, str, str, list[str]]],
    *,
    constraints: list[QueryConstraint],
) -> list[PaperRecord]:
    candidates: dict[str, ConstraintHit] = {}

    for paper, rank, subquery_type, query_text, constraint_ids in hits:
        key = candidate_key(paper)
        textual_covered = textual_constraint_coverage(paper, constraints)
        all_constraints = set(constraint_ids) | textual_covered
        score = subquery_rank_score(rank, subquery_type)
        score += 0.04 * len(all_constraints)
        if paper.identity_keys():
            score += 0.015

        if key not in candidates:
            candidates[key] = ConstraintHit(
                paper=paper,
                score=score,
                best_rank=max(1, int(rank)),
                occurrence_count=1,
                covered_constraints=set(all_constraints),
                subquery_hits=[
                    {
                        "query": query_text,
                        "subquery_type": subquery_type,
                        "rank": max(1, int(rank)),
                        "constraint_ids": list(constraint_ids),
                    }
                ],
            )
        else:
            candidates[key].add_hit(
                paper=paper,
                rank=rank,
                query_text=query_text,
                subquery_type=subquery_type,
                constraint_ids=list(all_constraints),
                score=score,
            )

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -len(item.covered_constraints),
            -item.occurrence_count,
            -item.score,
            item.best_rank,
            item.paper.title.casefold(),
        ),
    )

    output: list[PaperRecord] = []
    for item in ordered:
        item.paper.raw = dict(item.paper.raw or {})
        item.paper.raw["b2_aggregate_score"] = item.score
        item.paper.raw["b2_occurrence_count"] = item.occurrence_count
        item.paper.raw["b2_best_rank"] = item.best_rank
        item.paper.raw["b2_covered_constraints"] = sorted(item.covered_constraints)
        item.paper.raw["b2_coverage_count"] = len(item.covered_constraints)
        item.paper.raw["b2_subquery_hits"] = item.subquery_hits
        output.append(item.paper)
    return output
