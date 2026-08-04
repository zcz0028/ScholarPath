from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class FusionConfig:
    base_keep_top: int = 45
    supplement_slots: int = 5
    max_output: int = 100
    min_occurrence: int = 2
    max_anchor_rank: int = 50
    min_raw_anchor_score: float = 0.025


@dataclass(slots=True)
class FusionStats:
    base_kept: int = 0
    supplement_inserted: int = 0
    base_filled: int = 0
    supplement_filled: int = 0
    duplicates_skipped: int = 0
    low_confidence_skipped: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_identifier(value: object | None) -> str:
    text = str(value or "").strip()
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("https://arxiv.org/abs/", "")
    text = text.replace("arXiv:", "").replace("arxiv:", "")
    text = re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)
    return text.casefold().strip()


def normalize_title(value: object | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def paper_key(paper: dict[str, Any]) -> str:
    for field, prefix in (
        ("doi", "doi"),
        ("arxiv_id", "arxiv"),
        ("openalex_id", "openalex"),
        ("semantic_scholar_id", "s2"),
        ("canonical_id", "canonical"),
    ):
        value = normalize_identifier(paper.get(field))
        if value:
            return f"{prefix}:{value}"
    title = normalize_title(paper.get("title"))
    if title:
        return f"title:{title}"
    return f"object:{id(paper)}"


def raw_meta(paper: dict[str, Any]) -> dict[str, Any]:
    raw = paper.get("raw")
    return raw if isinstance(raw, dict) else {}


def has_strong_identifier(paper: dict[str, Any]) -> bool:
    return any(
        normalize_identifier(paper.get(field))
        for field in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id")
    )


def confidence_score(paper: dict[str, Any]) -> float:
    meta = raw_meta(paper)
    score = float(meta.get("b1_raw_anchor_score") or 0.0)
    occurrence = int(meta.get("b1_occurrence_count") or 1)
    best_anchor_rank = int(meta.get("b1_best_anchor_rank") or 999999)
    variant_types = set(meta.get("b1_variant_types") or [])
    if occurrence >= 2:
        score += 0.05 * (occurrence - 1)
    if best_anchor_rank <= 20:
        score += 0.04
    elif best_anchor_rank <= 50:
        score += 0.02
    if "raw" in variant_types:
        score += 0.03
    if "cleaned" in variant_types:
        score += 0.02
    if has_strong_identifier(paper):
        score += 0.015
    return score


def is_high_confidence_supplement(paper: dict[str, Any], config: FusionConfig) -> bool:
    meta = raw_meta(paper)
    occurrence = int(meta.get("b1_occurrence_count") or 1)
    best_anchor_rank = int(meta.get("b1_best_anchor_rank") or 999999)
    score = float(meta.get("b1_raw_anchor_score") or 0.0)
    variant_types = set(meta.get("b1_variant_types") or [])
    if occurrence >= config.min_occurrence:
        return True
    if best_anchor_rank <= config.max_anchor_rank and ("raw" in variant_types or "cleaned" in variant_types):
        return True
    if score >= config.min_raw_anchor_score and has_strong_identifier(paper):
        return True
    return False


def add_paper(output: list[dict[str, Any]], seen: set[str], paper: dict[str, Any]) -> bool:
    key = paper_key(paper)
    if key in seen:
        return False
    seen.add(key)
    output.append(paper)
    return True


def fuse_papers(
    base_papers: list[dict[str, Any]],
    supplement_papers: list[dict[str, Any]],
    config: FusionConfig,
) -> tuple[list[dict[str, Any]], FusionStats]:
    stats = FusionStats()
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_keep_top = max(0, min(config.base_keep_top, config.max_output))
    supplement_slots = max(0, config.supplement_slots)

    for paper in base_papers[:base_keep_top]:
        if add_paper(output, seen, paper):
            stats.base_kept += 1
        else:
            stats.duplicates_skipped += 1

    supplement_candidates: list[dict[str, Any]] = []
    for paper in supplement_papers:
        key = paper_key(paper)
        if key in seen:
            stats.duplicates_skipped += 1
            continue
        if not is_high_confidence_supplement(paper, config):
            stats.low_confidence_skipped += 1
            continue
        supplement_candidates.append(paper)

    supplement_candidates.sort(
        key=lambda item: (
            -confidence_score(item),
            int(raw_meta(item).get("b1_best_anchor_rank") or 999999),
            int(raw_meta(item).get("b1_best_rank") or 999999),
            normalize_title(item.get("title")),
        )
    )

    for paper in supplement_candidates[:supplement_slots]:
        if len(output) >= config.max_output:
            break
        if add_paper(output, seen, paper):
            stats.supplement_inserted += 1
        else:
            stats.duplicates_skipped += 1

    for paper in base_papers[base_keep_top:]:
        if len(output) >= config.max_output:
            break
        if add_paper(output, seen, paper):
            stats.base_filled += 1
        else:
            stats.duplicates_skipped += 1

    for paper in supplement_papers:
        if len(output) >= config.max_output:
            break
        if add_paper(output, seen, paper):
            stats.supplement_filled += 1
        else:
            stats.duplicates_skipped += 1

    return output[: config.max_output], stats


def fuse_prediction_records(
    base_record: dict[str, Any],
    supplement_record: dict[str, Any],
    config: FusionConfig,
) -> tuple[dict[str, Any], FusionStats]:
    base_papers = base_record.get("papers") or []
    supplement_papers = supplement_record.get("papers") or []
    if not isinstance(base_papers, list):
        base_papers = []
    if not isinstance(supplement_papers, list):
        supplement_papers = []
    fused_papers, stats = fuse_papers(base_papers, supplement_papers, config)
    fused_record = {
        "qid": base_record.get("qid") or supplement_record.get("qid"),
        "question": base_record.get("question") or supplement_record.get("question"),
        "papers": fused_papers,
        "fusion": {
            "strategy": "b0_fallback_b1_high_confidence_supplement",
            "config": asdict(config),
            "stats": stats.to_dict(),
        },
    }
    return fused_record, stats
