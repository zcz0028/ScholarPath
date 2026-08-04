from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from scholarpath.paper.schema import PaperRecord


VARIANT_TYPE_WEIGHTS = {
    "raw": 1.00,
    "cleaned": 0.95,
    "keyword_core": 0.90,
    "expanded": 0.85,
    "phrase_focus": 0.80,
}

RAW_ANCHOR_WEIGHTS = {
    "raw": 1.20,
    "cleaned": 1.00,
    "keyword_core": 0.35,
    "expanded": 0.15,
    "phrase_focus": 0.15,
}

RAW_ANCHOR_RANK_OFFSETS = {
    "raw": 3,
    "cleaned": 4,
    "keyword_core": 8,
    "expanded": 12,
    "phrase_focus": 12,
}


@dataclass(slots=True)
class CandidateHit:
    paper: PaperRecord
    score: float
    best_rank: int
    occurrence_count: int = 1
    variant_types: set[str] = field(default_factory=set)
    query_variants: list[dict[str, Any]] = field(default_factory=list)

    def add_hit(
        self,
        *,
        paper: PaperRecord,
        rank: int,
        variant_type: str,
        query_text: str,
        score: float,
    ) -> None:
        self.occurrence_count += 1
        self.score += score
        self.best_rank = min(self.best_rank, rank)
        self.variant_types.add(variant_type)
        self.query_variants.append(
            {
                "query": query_text,
                "variant_type": variant_type,
                "rank": rank,
            }
        )

        # Keep richer metadata when the new record has stronger identifiers.
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


def candidate_key(paper: PaperRecord) -> str:
    identity_keys = sorted(paper.identity_keys())
    if identity_keys:
        return identity_keys[0]
    title = paper.normalized_title
    if title:
        return f"title:{title}"
    return paper.canonical_id


def rank_score(rank: int, variant_type: str) -> float:
    weight = VARIANT_TYPE_WEIGHTS.get(variant_type, 0.75)
    safe_rank = max(1, int(rank))
    return weight / (safe_rank + 4)


def raw_anchor_score(item: CandidateHit) -> float:
    """Score candidates while keeping raw/cleaned as anchor channels.

    Compared with the original weighted score, keyword/expanded channels are
    intentionally down-weighted. This allows them to supplement recall without
    easily pushing high-ranked raw/cleaned candidates out of Top-K.
    """
    score = 0.0
    for hit in item.query_variants:
        variant_type = str(hit.get("variant_type") or "")
        rank = max(1, int(hit.get("rank") or 1))
        weight = RAW_ANCHOR_WEIGHTS.get(variant_type, 0.10)
        offset = RAW_ANCHOR_RANK_OFFSETS.get(variant_type, 12)
        score += weight / (rank + offset)

    if item.occurrence_count >= 2:
        score += 0.04 * (item.occurrence_count - 1)

    if item.paper.identity_keys():
        score += 0.015

    return score


def best_anchor_rank(item: CandidateHit) -> int:
    anchor_ranks = [
        int(hit.get("rank") or 999999)
        for hit in item.query_variants
        if hit.get("variant_type") in {"raw", "cleaned"}
    ]
    return min(anchor_ranks) if anchor_ranks else 999999


def best_raw_rank(item: CandidateHit) -> int:
    raw_ranks = [
        int(hit.get("rank") or 999999)
        for hit in item.query_variants
        if hit.get("variant_type") == "raw"
    ]
    return min(raw_ranks) if raw_ranks else 999999


def aggregate_candidate_hits(
    hits: Iterable[tuple[PaperRecord, int, str, str]],
    *,
    mode: str = "weighted",
) -> list[PaperRecord]:
    if mode not in {"weighted", "raw_anchor"}:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    candidates: dict[str, CandidateHit] = {}

    for paper, rank, variant_type, query_text in hits:
        key = candidate_key(paper)
        score = rank_score(rank, variant_type)
        if key not in candidates:
            candidates[key] = CandidateHit(
                paper=paper,
                score=score,
                best_rank=max(1, int(rank)),
                occurrence_count=1,
                variant_types={variant_type},
                query_variants=[
                    {
                        "query": query_text,
                        "variant_type": variant_type,
                        "rank": max(1, int(rank)),
                    }
                ],
            )
        else:
            candidates[key].add_hit(
                paper=paper,
                rank=max(1, int(rank)),
                variant_type=variant_type,
                query_text=query_text,
                score=score,
            )

    if mode == "weighted":
        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -item.score,
                -item.occurrence_count,
                item.best_rank,
                item.paper.title.casefold(),
            ),
        )
    else:
        for item in candidates.values():
            item.paper.raw = dict(item.paper.raw or {})
            item.paper.raw["b1_raw_anchor_score"] = raw_anchor_score(item)
            item.paper.raw["b1_best_anchor_rank"] = best_anchor_rank(item)
            item.paper.raw["b1_best_raw_rank"] = best_raw_rank(item)

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item.paper.raw.get("b1_raw_anchor_score", 0.0)),
                best_anchor_rank(item),
                best_raw_rank(item),
                -item.occurrence_count,
                item.best_rank,
                item.paper.title.casefold(),
            ),
        )

    output: list[PaperRecord] = []
    for item in ordered:
        item.paper.raw = dict(item.paper.raw or {})
        item.paper.raw["b1_aggregate_score"] = item.score
        item.paper.raw["b1_occurrence_count"] = item.occurrence_count
        item.paper.raw["b1_best_rank"] = item.best_rank
        item.paper.raw["b1_variant_types"] = sorted(item.variant_types)
        item.paper.raw["b1_query_variants"] = item.query_variants
        item.paper.raw["b1_aggregation_mode"] = mode
        output.append(item.paper)
    return output
