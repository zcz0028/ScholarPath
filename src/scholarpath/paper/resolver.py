from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .normalizers import first_author_key, normalize_title
from .schema import PaperRecord


@dataclass(slots=True)
class ResolutionConfig:
    max_year_gap: int = 2
    min_title_length: int = 12
    fuzzy_review_threshold: float = 0.93
    max_review_pairs: int = 1000


@dataclass(slots=True)
class PaperEntity:
    record: PaperRecord
    members: list[PaperRecord] = field(default_factory=list)
    match_reasons: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    years: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.members:
            self.members = [self.record]
        if not self.sources:
            self.sources = {m.source for m in self.members}
        if not self.years:
            self.years = {m.year for m in self.members if m.year is not None}

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.record.canonical_id,
            "paper": self.record.to_dict(),
            "sources": sorted(self.sources),
            "years": sorted(self.years),
            "member_count": len(self.members),
            "match_reasons": list(dict.fromkeys(self.match_reasons)),
            "members": [m.to_dict() for m in self.members],
        }


@dataclass(slots=True)
class ReviewCandidate:
    left_entity: int
    right_entity: int
    left_title: str
    right_title: str
    similarity: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_entity": self.left_entity,
            "right_entity": self.right_entity,
            "left_title": self.left_title,
            "right_title": self.right_title,
            "similarity": round(self.similarity, 6),
            "reason": self.reason,
        }


@dataclass(slots=True)
class ResolutionResult:
    entities: list[PaperEntity]
    review_candidates: list[ReviewCandidate]
    input_count: int
    exact_id_merges: int
    title_merges: int
    ambiguous_records: int

    def report(self) -> dict[str, Any]:
        return {
            "input_records": self.input_count,
            "resolved_entities": len(self.entities),
            "removed_duplicates": self.input_count - len(self.entities),
            "duplicate_entities": sum(len(e.members) > 1 for e in self.entities),
            "exact_id_merges": self.exact_id_merges,
            "title_author_year_merges": self.title_merges,
            "ambiguous_records": self.ambiguous_records,
            "review_candidate_pairs": len(self.review_candidates),
        }


def _identifier_conflicts(left: PaperRecord, right: PaperRecord) -> list[str]:
    conflicts: list[str] = []
    for name in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id"):
        lv = getattr(left, name)
        rv = getattr(right, name)
        if lv and rv and lv != rv:
            conflicts.append(name)
    return conflicts


def _title_merge_reason(left: PaperRecord, right: PaperRecord, config: ResolutionConfig) -> str | None:
    lt = normalize_title(left.title)
    rt = normalize_title(right.title)
    if not lt or lt != rt or len(lt) < config.min_title_length:
        return None
    if _identifier_conflicts(left, right):
        return None

    la = first_author_key(left.authors)
    ra = first_author_key(right.authors)
    same_author = bool(la and ra and la == ra)
    same_year = left.year is not None and right.year is not None and left.year == right.year
    nearby_year = (
        left.year is not None
        and right.year is not None
        and abs(left.year - right.year) <= config.max_year_gap
    )

    if same_author and nearby_year:
        return "exact_title+same_first_author+compatible_year"
    if same_author and (left.year is None or right.year is None):
        return "exact_title+same_first_author"
    if same_year and (not la or not ra):
        return "exact_title+same_year"
    return None


def _pick_longer(current: str | None, incoming: str | None) -> str | None:
    if not current:
        return incoming
    if not incoming:
        return current
    return incoming if len(incoming.strip()) > len(current.strip()) else current


def _merge_values(target: PaperRecord, incoming: PaperRecord) -> None:
    target.title = _pick_longer(target.title, incoming.title) or target.title
    if len(incoming.authors) > len(target.authors):
        target.authors = list(incoming.authors)
    years = [x for x in (target.year, incoming.year) if x is not None]
    target.year = min(years) if years else None
    target.publication_date = _pick_longer(target.publication_date, incoming.publication_date)
    target.venue = _pick_longer(target.venue, incoming.venue)
    target.doi = target.doi or incoming.doi
    target.arxiv_id = target.arxiv_id or incoming.arxiv_id
    target.openalex_id = target.openalex_id or incoming.openalex_id
    target.semantic_scholar_id = target.semantic_scholar_id or incoming.semantic_scholar_id
    target.url = target.url or incoming.url
    target.abstract = _pick_longer(target.abstract, incoming.abstract)
    counts = [x for x in (target.citation_count, incoming.citation_count) if x is not None]
    target.citation_count = max(counts) if counts else None


class PaperResolver:
    def __init__(self, config: ResolutionConfig | None = None) -> None:
        self.config = config or ResolutionConfig()

    def resolve(self, records: Iterable[PaperRecord]) -> ResolutionResult:
        items = list(records)
        entities: list[PaperEntity] = []
        exact_id_merges = 0
        title_merges = 0
        ambiguous = 0

        for record in items:
            id_matches = [i for i, e in enumerate(entities) if record.identity_keys() & e.record.identity_keys()]
            if len(id_matches) == 1:
                self._merge_into(entities[id_matches[0]], record, "shared_strong_identifier")
                exact_id_merges += 1
                continue
            if len(id_matches) > 1:
                target = entities[id_matches[0]]
                for idx in reversed(id_matches[1:]):
                    other = entities.pop(idx)
                    for member in other.members:
                        self._merge_into(target, member, "bridged_by_strong_identifiers")
                self._merge_into(target, record, "shared_strong_identifier")
                exact_id_merges += 1
                continue

            title_matches: list[tuple[int, str]] = []
            for i, entity in enumerate(entities):
                reason = _title_merge_reason(entity.record, record, self.config)
                if reason:
                    title_matches.append((i, reason))
            if len(title_matches) == 1:
                i, reason = title_matches[0]
                self._merge_into(entities[i], record, reason)
                title_merges += 1
                continue
            if len(title_matches) > 1:
                ambiguous += 1
            entities.append(PaperEntity(record=record))

        review = self._review_candidates(entities)
        return ResolutionResult(
            entities=entities,
            review_candidates=review,
            input_count=len(items),
            exact_id_merges=exact_id_merges,
            title_merges=title_merges,
            ambiguous_records=ambiguous,
        )

    @staticmethod
    def _merge_into(entity: PaperEntity, record: PaperRecord, reason: str) -> None:
        _merge_values(entity.record, record)
        entity.members.append(record)
        entity.sources.add(record.source)
        if record.year is not None:
            entity.years.add(record.year)
        entity.match_reasons.append(reason)

    def _review_candidates(self, entities: list[PaperEntity]) -> list[ReviewCandidate]:
        out: list[ReviewCandidate] = []
        for i, left_entity in enumerate(entities):
            lt = left_entity.record.normalized_title
            if not lt:
                continue
            for j in range(i + 1, len(entities)):
                if len(out) >= self.config.max_review_pairs:
                    return out
                right_entity = entities[j]
                rt = right_entity.record.normalized_title
                if not rt or lt == rt:
                    continue
                score = SequenceMatcher(None, lt, rt).ratio()
                if score < self.config.fuzzy_review_threshold:
                    continue
                conflicts = _identifier_conflicts(left_entity.record, right_entity.record)
                out.append(
                    ReviewCandidate(
                        left_entity=i,
                        right_entity=j,
                        left_title=left_entity.record.title,
                        right_title=right_entity.record.title,
                        similarity=score,
                        reason=(
                            "high_title_similarity_with_identifier_conflict"
                            if conflicts
                            else "high_title_similarity_requires_review"
                        ),
                    )
                )
        return out


def write_resolution_outputs(result: ResolutionResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "resolved": root / "resolved_papers.jsonl",
        "groups": root / "duplicate_groups.jsonl",
        "review": root / "review_candidates.jsonl",
        "report": root / "resolver_report.json",
    }
    with paths["resolved"].open("w", encoding="utf-8") as f:
        for entity in result.entities:
            f.write(json.dumps(entity.record.to_dict(), ensure_ascii=False) + "\n")
    with paths["groups"].open("w", encoding="utf-8") as f:
        for entity in result.entities:
            if len(entity.members) > 1:
                f.write(json.dumps(entity.to_dict(), ensure_ascii=False) + "\n")
    with paths["review"].open("w", encoding="utf-8") as f:
        for candidate in result.review_candidates:
            f.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
    paths["report"].write_text(json.dumps(result.report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths
