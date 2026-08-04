from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scholarpath.rerank.constraint_rerank import paper_key


@dataclass(slots=True)
class CandidateSource:
    name: str
    directory: str
    file_name: str = "predictions_top100.jsonl"
    weight: float = 1.0


@dataclass(slots=True)
class FusionConfig:
    max_fused_candidates: int = 200
    source_count_bonus: float = 0.08
    strong_id_bonus: float = 0.02


@dataclass(slots=True)
class FusedCandidate:
    paper: dict[str, Any]
    sources: set[str] = field(default_factory=set)
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    pre_rerank_score: float = 0.0

    def add_source(self, source: CandidateSource, rank: int, paper: dict[str, Any]) -> None:
        rank = max(1, int(rank))
        self.sources.add(source.name)
        previous_rank = self.source_ranks.get(source.name)
        if previous_rank is None or rank < previous_rank:
            self.source_ranks[source.name] = rank
        source_score = source.weight / math.log2(rank + 2)
        self.source_scores[source.name] = max(
            self.source_scores.get(source.name, 0.0),
            source_score,
        )
        self.paper = merge_paper_metadata(self.paper, paper)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}")
            records.append(value)
    return records


def load_source_predictions(source: CandidateSource) -> dict[str, dict[str, Any]]:
    path = Path(source.directory) / source.file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing source prediction file: {path}")
    records = read_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("qid"))
        if not qid:
            continue
        output[qid] = record
    return output


def raw_meta(paper: dict[str, Any]) -> dict[str, Any]:
    raw = paper.get("raw")
    return raw if isinstance(raw, dict) else {}


def has_strong_identifier(paper: dict[str, Any]) -> bool:
    return any(
        str(paper.get(field) or "").strip()
        for field in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id")
    )


def merge_lists(left: Any, right: Any) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for seq in (left, right):
        if not isinstance(seq, list):
            continue
        for item in seq:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
            if key in seen:
                continue
            values.append(item)
            seen.add(key)
    return values


def better_text(current: Any, new: Any) -> Any:
    current_text = str(current or "")
    new_text = str(new or "")
    if len(new_text) > len(current_text):
        return new
    return current


def merge_paper_metadata(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for field_name in (
        "title",
        "doi",
        "arxiv_id",
        "openalex_id",
        "semantic_scholar_id",
        "url",
        "pdf_url",
        "venue",
        "published_date",
    ):
        if not merged.get(field_name) and incoming.get(field_name):
            merged[field_name] = incoming.get(field_name)

    merged["abstract"] = better_text(merged.get("abstract"), incoming.get("abstract"))

    if isinstance(incoming.get("authors"), list):
        merged["authors"] = merge_lists(merged.get("authors"), incoming.get("authors"))

    base_raw = raw_meta(merged)
    incoming_raw = raw_meta(incoming)
    merged_raw = dict(base_raw)

    # Preserve useful raw metadata from any source.
    for key, value in incoming_raw.items():
        if key not in merged_raw or merged_raw.get(key) in (None, "", [], {}):
            merged_raw[key] = value

    # Merge selected list-like diagnostic fields.
    for key in (
        "b1_variant_types",
        "b2_covered_constraints",
        "b2_subquery_hits",
        "b2_1_covered_constraints",
        "b2_1_coverage_details",
    ):
        if key in base_raw or key in incoming_raw:
            merged_raw[key] = merge_lists(base_raw.get(key), incoming_raw.get(key))

    merged["raw"] = merged_raw
    return merged


def source_reason_tags(candidate: FusedCandidate) -> list[str]:
    tags: list[str] = []
    sources = candidate.sources
    if "b0" in sources:
        tags.append("stable_raw_baseline_source")
    if "b1_2" in sources or "b1" in sources:
        tags.append("query_rewrite_recall_source")
    if "b2" in sources:
        tags.append("constraint_retrieval_source")
    if len(sources) >= 2:
        tags.append("multi_source_supported")
    if any(rank <= 20 for rank in candidate.source_ranks.values()):
        tags.append("high_rank_in_at_least_one_source")
    return tags


def compute_pre_rerank_score(candidate: FusedCandidate, config: FusionConfig) -> float:
    source_score = sum(candidate.source_scores.values())
    source_bonus = config.source_count_bonus * max(0, len(candidate.sources) - 1)
    strong_bonus = config.strong_id_bonus if has_strong_identifier(candidate.paper) else 0.0
    best_rank_bonus = 0.0
    best_rank = min(candidate.source_ranks.values()) if candidate.source_ranks else 999
    if best_rank <= 10:
        best_rank_bonus = 0.04
    elif best_rank <= 20:
        best_rank_bonus = 0.025
    elif best_rank <= 50:
        best_rank_bonus = 0.01
    return source_score + source_bonus + strong_bonus + best_rank_bonus


def finalize_candidate(candidate: FusedCandidate, config: FusionConfig) -> dict[str, Any]:
    score = compute_pre_rerank_score(candidate, config)
    candidate.pre_rerank_score = score
    paper = dict(candidate.paper)
    raw = dict(raw_meta(paper))
    raw["b4_sources"] = sorted(candidate.sources)
    raw["b4_source_count"] = len(candidate.sources)
    raw["b4_source_ranks"] = dict(sorted(candidate.source_ranks.items()))
    raw["b4_source_scores"] = dict(sorted(candidate.source_scores.items()))
    raw["b4_best_source_rank"] = min(candidate.source_ranks.values()) if candidate.source_ranks else None
    raw["b4_pre_rerank_score"] = score
    raw["b4_fusion_reason_tags"] = source_reason_tags(candidate)
    paper["raw"] = raw
    return paper


def fuse_candidate_records(
    *,
    qid: str,
    question: str,
    source_records: list[tuple[CandidateSource, dict[str, Any] | None]],
    config: FusionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: dict[str, FusedCandidate] = {}
    source_counts: dict[str, int] = {}

    for source, record in source_records:
        papers = []
        if isinstance(record, dict):
            raw_papers = record.get("papers")
            if isinstance(raw_papers, list):
                papers = raw_papers
        source_counts[source.name] = len(papers)

        for rank, paper in enumerate(papers, start=1):
            if not isinstance(paper, dict):
                continue
            key = paper_key(paper)
            if key not in candidates:
                candidates[key] = FusedCandidate(paper=dict(paper))
            candidates[key].add_source(source, rank, paper)

    fused_papers = [finalize_candidate(item, config) for item in candidates.values()]
    fused_papers.sort(
        key=lambda paper: (
            -float(raw_meta(paper).get("b4_pre_rerank_score", 0.0)),
            -int(raw_meta(paper).get("b4_source_count", 0)),
            int(raw_meta(paper).get("b4_best_source_rank") or 999999),
            str(paper.get("title") or "").casefold(),
        )
    )
    fused_papers = fused_papers[: config.max_fused_candidates]

    stats = {
        "qid": qid,
        "question": question,
        "source_counts": source_counts,
        "unique_fused_candidates": len(candidates),
        "kept_fused_candidates": len(fused_papers),
        "multi_source_candidates": sum(
            1 for paper in fused_papers
            if int(raw_meta(paper).get("b4_source_count", 0)) >= 2
        ),
        "b0_only_candidates": sum(
            1 for paper in fused_papers
            if raw_meta(paper).get("b4_sources") == ["b0"]
        ),
        "b1_2_only_candidates": sum(
            1 for paper in fused_papers
            if raw_meta(paper).get("b4_sources") in (["b1_2"], ["b1"])
        ),
        "b2_only_candidates": sum(
            1 for paper in fused_papers
            if raw_meta(paper).get("b4_sources") == ["b2"]
        ),
    }

    record = {
        "qid": qid,
        "question": question,
        "papers": fused_papers,
        "fusion": {
            "strategy": "b4_multi_source_candidate_fusion",
            "source_names": [source.name for source, _ in source_records],
            "config": {
                "max_fused_candidates": config.max_fused_candidates,
                "source_count_bonus": config.source_count_bonus,
                "strong_id_bonus": config.strong_id_bonus,
            },
        },
    }
    return record, stats
