from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from scholarpath.paper.normalizers import first_author_key, normalize_title
from scholarpath.paper.schema import PaperRecord

SUPPORTED_MATCH_MODES = {"strict", "strict_v2", "pasa_title"}


def pasa_title_key(title: str | None) -> str:
    """Reproduce PaSa's title key: retain alphabetic characters only."""
    if not title:
        return ""
    return "".join(char for char in title if char.isalpha()).lower()


def _identifier_conflicts(left: PaperRecord, right: PaperRecord) -> bool:
    for field_name in (
        "doi",
        "arxiv_id",
        "openalex_id",
        "semantic_scholar_id",
    ):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _identifier_conflicts_v2(left: PaperRecord, right: PaperRecord) -> bool:
    left_ids = left.inferred_identifiers()
    right_ids = right.inferred_identifiers()
    for field_name in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id"):
        left_value = left_ids.get(field_name)
        right_value = right_ids.get(field_name)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _shared_strong_identifier(left: PaperRecord, right: PaperRecord) -> bool:
    return bool(left.identity_keys() & right.identity_keys())


def _shared_strong_identifier_v2(left: PaperRecord, right: PaperRecord) -> bool:
    return bool(left.identity_keys_v2() & right.identity_keys_v2())


def _compatible_bibliographic_metadata(
    left: PaperRecord,
    right: PaperRecord,
    max_year_gap: int = 2,
) -> tuple[bool, str]:
    left_author = first_author_key(left.authors)
    right_author = first_author_key(right.authors)
    authors_available = bool(left_author and right_author)
    same_author = authors_available and left_author == right_author

    years_available = left.year is not None and right.year is not None
    compatible_year = years_available and abs(left.year - right.year) <= max_year_gap

    if authors_available and not same_author:
        return False, "first_author_conflict"
    if years_available and not compatible_year:
        return False, "year_conflict"
    if same_author and compatible_year:
        return True, "same_first_author+compatible_year"
    if same_author:
        return True, "same_first_author"
    if compatible_year:
        return True, "compatible_year"
    return True, "metadata_sparse"


@dataclass(slots=True, frozen=True)
class MatchPair:
    prediction_index: int
    gold_index: int
    match_type: str


@dataclass(slots=True)
class MatchResult:
    pairs: list[MatchPair]
    unmatched_prediction_indices: list[int]
    unmatched_gold_indices: list[int]

    @property
    def tp(self) -> int:
        return len(self.pairs)


@dataclass(slots=True, frozen=True)
class MatchAuditCandidate:
    prediction_index: int
    gold_index: int
    title_similarity: float
    token_jaccard: float
    same_first_author: bool
    year_gap: int | None
    identifier_conflict: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_index": self.prediction_index,
            "prediction_rank": self.prediction_index + 1,
            "gold_index": self.gold_index,
            "title_similarity": round(self.title_similarity, 6),
            "token_jaccard": round(self.token_jaccard, 6),
            "same_first_author": self.same_first_author,
            "year_gap": self.year_gap,
            "identifier_conflict": self.identifier_conflict,
            "reason": self.reason,
        }


def deduplicate_predictions(
    predictions: Iterable[PaperRecord],
) -> tuple[list[PaperRecord], list[dict[str, object]]]:
    """Legacy deduplication retained for first-week baseline reproducibility."""
    kept: list[PaperRecord] = []
    removed: list[dict[str, object]] = []

    for original_index, paper in enumerate(predictions):
        duplicate_of: int | None = None
        reason: str | None = None

        for kept_index, existing in enumerate(kept):
            if _shared_strong_identifier(existing, paper):
                duplicate_of = kept_index
                reason = "shared_strong_identifier"
                break

            if (
                paper.normalized_title
                and paper.normalized_title == existing.normalized_title
                and not _identifier_conflicts(existing, paper)
            ):
                existing_ids = existing.identity_keys()
                paper_ids = paper.identity_keys()
                if not existing_ids or not paper_ids or existing_ids == paper_ids:
                    duplicate_of = kept_index
                    reason = "exact_normalized_title"
                    break

        if duplicate_of is None:
            kept.append(paper)
        else:
            _merge_duplicate_metadata(kept[duplicate_of], paper)
            removed.append(
                {
                    "original_index": original_index,
                    "duplicate_of_kept_index": duplicate_of,
                    "reason": reason,
                    "paper": paper.to_dict(),
                }
            )

    return kept, removed



def _merge_duplicate_metadata(target: PaperRecord, incoming: PaperRecord) -> None:
    """Preserve identifiers when a later duplicate is removed."""
    target.doi = target.doi or incoming.doi
    target.arxiv_id = target.arxiv_id or incoming.arxiv_id
    target.openalex_id = target.openalex_id or incoming.openalex_id
    target.semantic_scholar_id = target.semantic_scholar_id or incoming.semantic_scholar_id
    target.url = target.url or incoming.url
    if not target.authors and incoming.authors:
        target.authors = list(incoming.authors)
    if target.year is None:
        target.year = incoming.year


def deduplicate_predictions_v2(
    predictions: Iterable[PaperRecord],
) -> tuple[list[PaperRecord], list[dict[str, object]]]:
    """Identity-aware deduplication for DOI/arXiv/URL and version alignment."""
    kept: list[PaperRecord] = []
    removed: list[dict[str, object]] = []

    for original_index, paper in enumerate(predictions):
        duplicate_of: int | None = None
        reason: str | None = None

        for kept_index, existing in enumerate(kept):
            if _shared_strong_identifier_v2(existing, paper):
                duplicate_of = kept_index
                reason = "shared_inferred_identifier"
                break

            if (
                paper.normalized_title
                and paper.normalized_title == existing.normalized_title
                and not _identifier_conflicts_v2(existing, paper)
            ):
                existing_ids = existing.identity_keys_v2()
                paper_ids = paper.identity_keys_v2()
                compatible, evidence = _compatible_bibliographic_metadata(existing, paper)
                if (not existing_ids or not paper_ids) and compatible:
                    duplicate_of = kept_index
                    reason = f"exact_title+{evidence}"
                    break
                if existing_ids == paper_ids and compatible:
                    duplicate_of = kept_index
                    reason = f"exact_title+same_identity+{evidence}"
                    break
                if existing_ids != paper_ids and compatible and evidence != "metadata_sparse":
                    duplicate_of = kept_index
                    reason = f"preprint_formal_version+{evidence}"
                    break

        if duplicate_of is None:
            kept.append(deepcopy(paper))
        else:
            _merge_duplicate_metadata(kept[duplicate_of], paper)
            removed.append(
                {
                    "original_index": original_index,
                    "duplicate_of_kept_index": duplicate_of,
                    "reason": reason,
                    "paper": paper.to_dict(),
                }
            )

    return kept, removed


def match_papers(
    predictions: list[PaperRecord],
    gold: list[PaperRecord],
    mode: str = "strict",
) -> MatchResult:
    if mode not in SUPPORTED_MATCH_MODES:
        raise ValueError(f"Unsupported matching mode: {mode}")

    unmatched_predictions = set(range(len(predictions)))
    unmatched_gold = set(range(len(gold)))
    pairs: list[MatchPair] = []

    if mode in {"strict", "strict_v2"}:
        use_v2 = mode == "strict_v2"
        shared_identifier = _shared_strong_identifier_v2 if use_v2 else _shared_strong_identifier
        conflict_check = _identifier_conflicts_v2 if use_v2 else _identifier_conflicts

        for prediction_index, prediction in enumerate(predictions):
            if prediction_index not in unmatched_predictions:
                continue
            for gold_index in sorted(unmatched_gold):
                if shared_identifier(prediction, gold[gold_index]):
                    pairs.append(
                        MatchPair(
                            prediction_index=prediction_index,
                            gold_index=gold_index,
                            match_type=(
                                "strong_identifier_v2" if use_v2 else "strong_identifier"
                            ),
                        )
                    )
                    unmatched_predictions.remove(prediction_index)
                    unmatched_gold.remove(gold_index)
                    break

        for prediction_index in sorted(unmatched_predictions.copy()):
            prediction = predictions[prediction_index]
            if not prediction.normalized_title:
                continue
            for gold_index in sorted(unmatched_gold):
                gold_paper = gold[gold_index]
                if (
                    prediction.normalized_title == gold_paper.normalized_title
                    and not conflict_check(prediction, gold_paper)
                ):
                    match_type = "normalized_title"
                    if use_v2:
                        compatible, evidence = _compatible_bibliographic_metadata(
                            prediction, gold_paper
                        )
                        if not compatible:
                            continue
                        match_type = f"normalized_title_v2+{evidence}"
                    pairs.append(
                        MatchPair(
                            prediction_index=prediction_index,
                            gold_index=gold_index,
                            match_type=match_type,
                        )
                    )
                    unmatched_predictions.remove(prediction_index)
                    unmatched_gold.remove(gold_index)
                    break
    else:
        gold_by_key: dict[str, list[int]] = {}
        for gold_index, gold_paper in enumerate(gold):
            key = pasa_title_key(gold_paper.title)
            if key:
                gold_by_key.setdefault(key, []).append(gold_index)

        for prediction_index, prediction in enumerate(predictions):
            key = pasa_title_key(prediction.title)
            if not key:
                continue
            candidate_gold = gold_by_key.get(key, [])
            for gold_index in candidate_gold:
                if gold_index in unmatched_gold:
                    pairs.append(
                        MatchPair(
                            prediction_index=prediction_index,
                            gold_index=gold_index,
                            match_type="pasa_title",
                        )
                    )
                    unmatched_predictions.remove(prediction_index)
                    unmatched_gold.remove(gold_index)
                    break

    pairs.sort(key=lambda pair: pair.prediction_index)
    return MatchResult(
        pairs=pairs,
        unmatched_prediction_indices=sorted(unmatched_predictions),
        unmatched_gold_indices=sorted(unmatched_gold),
    )


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def audit_possible_matches(
    predictions: list[PaperRecord],
    gold: list[PaperRecord],
    *,
    title_similarity_threshold: float = 0.88,
    token_jaccard_threshold: float = 0.75,
    max_candidates: int = 100,
) -> list[MatchAuditCandidate]:
    """Find plausible version/title matches without counting them as TP."""
    strict_result = match_papers(predictions, gold, mode="strict_v2")
    candidates: list[MatchAuditCandidate] = []
    for prediction_index in strict_result.unmatched_prediction_indices:
        prediction = predictions[prediction_index]
        prediction_title = prediction.normalized_title
        if not prediction_title:
            continue
        for gold_index in strict_result.unmatched_gold_indices:
            gold_paper = gold[gold_index]
            gold_title = gold_paper.normalized_title
            if not gold_title:
                continue
            similarity = SequenceMatcher(None, prediction_title, gold_title).ratio()
            jaccard = _token_jaccard(prediction.title, gold_paper.title)
            if similarity < title_similarity_threshold and jaccard < token_jaccard_threshold:
                continue
            left_author = first_author_key(prediction.authors)
            right_author = first_author_key(gold_paper.authors)
            same_author = bool(left_author and right_author and left_author == right_author)
            year_gap = (
                abs(prediction.year - gold_paper.year)
                if prediction.year is not None and gold_paper.year is not None
                else None
            )
            conflict = _identifier_conflicts_v2(prediction, gold_paper)
            reason = "near_title_requires_review"
            if conflict:
                reason = "near_title_with_identifier_conflict"
            elif same_author and (year_gap is None or year_gap <= 2):
                reason = "probable_preprint_formal_version"
            candidates.append(
                MatchAuditCandidate(
                    prediction_index=prediction_index,
                    gold_index=gold_index,
                    title_similarity=similarity,
                    token_jaccard=jaccard,
                    same_first_author=same_author,
                    year_gap=year_gap,
                    identifier_conflict=conflict,
                    reason=reason,
                )
            )
    candidates.sort(
        key=lambda item: (
            item.identifier_conflict,
            -max(item.title_similarity, item.token_jaccard),
            item.prediction_index,
        )
    )
    return candidates[:max_candidates]
