from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from scholarpath.paper.schema import PaperRecord


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


def _shared_strong_identifier(left: PaperRecord, right: PaperRecord) -> bool:
    return bool(left.identity_keys() & right.identity_keys())


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


def deduplicate_predictions(
    predictions: Iterable[PaperRecord],
) -> tuple[list[PaperRecord], list[dict[str, object]]]:
    """Remove exact duplicates while retaining ranking order.

    Rules:
    - shared strong identifier => duplicate;
    - exact normalized title => duplicate only when at least one side has no
      strong identifier, or both sides do not conflict on a populated ID;
    - same title with conflicting populated IDs remains separate.
    """
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
    if mode not in {"strict", "pasa_title"}:
        raise ValueError(f"Unsupported matching mode: {mode}")

    unmatched_predictions = set(range(len(predictions)))
    unmatched_gold = set(range(len(gold)))
    pairs: list[MatchPair] = []

    if mode == "strict":
        # Tier 1: exact normalized strong identifiers.
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index not in unmatched_predictions:
                continue
            for gold_index in sorted(unmatched_gold):
                if _shared_strong_identifier(prediction, gold[gold_index]):
                    pairs.append(
                        MatchPair(
                            prediction_index=prediction_index,
                            gold_index=gold_index,
                            match_type="strong_identifier",
                        )
                    )
                    unmatched_predictions.remove(prediction_index)
                    unmatched_gold.remove(gold_index)
                    break

        # Tier 2: exact normalized title, unless populated IDs conflict.
        for prediction_index in sorted(unmatched_predictions.copy()):
            prediction = predictions[prediction_index]
            if not prediction.normalized_title:
                continue
            for gold_index in sorted(unmatched_gold):
                gold_paper = gold[gold_index]
                if (
                    prediction.normalized_title == gold_paper.normalized_title
                    and not _identifier_conflicts(prediction, gold_paper)
                ):
                    pairs.append(
                        MatchPair(
                            prediction_index=prediction_index,
                            gold_index=gold_index,
                            match_type="normalized_title",
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
