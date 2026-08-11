from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from scholarpath.rerank.constraint_evidence import (
    CanonicalConstraint,
    ConstraintEvidence,
)


# Stable machine-readable tags. Translation belongs to the UI/API presentation
# layer; the ranking/reasoning layer does not hard-code Chinese UI labels.
_CONSTRAINT_TYPE_TAGS: dict[str, str] = {
    "model_or_entity": "model_or_entity_match",
    "task_or_modality": "task_or_modality_match",
    "method_or_property": "method_or_property_match",
    "topic": "topic_match",
}

_FIELD_TAGS: dict[str, str] = {
    "title": "title_evidence",
    "abstract": "abstract_evidence",
    "concept": "concept_evidence",
}

_FIELD_LABELS: dict[str, str] = {
    "title": "paper title",
    "abstract": "paper abstract",
    "concept": "OpenAlex concepts",
}

_FIELD_ORDER = ("title", "abstract", "concept")


@dataclass(slots=True, frozen=True)
class RecommendationReason:
    reason_tags: tuple[str, ...]
    reason_text: str
    matched_constraints: tuple[str, ...]
    unmatched_constraints: tuple[str, ...]
    matched_count: int
    constraint_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reason_tags"] = list(self.reason_tags)
        data["matched_constraints"] = list(self.matched_constraints)
        data["unmatched_constraints"] = list(self.unmatched_constraints)
        return data


def build_recommendation_reason(
    *,
    constraints: Sequence[CanonicalConstraint],
    evidence: Sequence[ConstraintEvidence],
) -> RecommendationReason:
    """Build a deterministic recommendation reason from structured evidence.

    Important boundaries:
    - does not inspect title/abstract/concepts;
    - does not perform matching;
    - does not call an LLM or network;
    - canonical constraints are the source of truth;
    - unknown/duplicate evidence rows cannot create duplicate reasons.
    """
    canonical = tuple(constraints or ())
    if not canonical:
        return RecommendationReason(
            reason_tags=(),
            reason_text="",
            matched_constraints=(),
            unmatched_constraints=(),
            matched_count=0,
            constraint_count=0,
        )

    evidence_by_id = _best_known_evidence(canonical, evidence or ())

    matched: list[CanonicalConstraint] = []
    unmatched: list[CanonicalConstraint] = []
    matched_evidence: list[ConstraintEvidence] = []

    for constraint in canonical:
        item = evidence_by_id.get(constraint.id)
        if item is not None and item.matched:
            matched.append(constraint)
            matched_evidence.append(item)
        else:
            unmatched.append(constraint)

    tags = _build_tags(matched, matched_evidence)
    reason_text = _build_reason_text(
        matched_constraints=matched,
        matched_evidence=matched_evidence,
        constraint_count=len(canonical),
    )

    return RecommendationReason(
        reason_tags=tags,
        reason_text=reason_text,
        matched_constraints=tuple(c.canonical_text for c in matched),
        unmatched_constraints=tuple(c.canonical_text for c in unmatched),
        matched_count=len(matched),
        constraint_count=len(canonical),
    )


def attach_recommendation_reason(
    paper: Mapping[str, Any],
    *,
    constraints: Sequence[CanonicalConstraint],
    evidence: Sequence[ConstraintEvidence],
) -> dict[str, Any]:
    """Return a copy of *paper* annotated with Day8-1D raw fields."""
    result = dict(paper)
    raw = dict(result.get("raw") or {})
    reason = build_recommendation_reason(
        constraints=constraints,
        evidence=evidence,
    )

    raw["day8_reason_tags"] = list(reason.reason_tags)
    raw["day8_reason_text"] = reason.reason_text
    raw["day8_matched_constraints"] = list(reason.matched_constraints)
    raw["day8_unmatched_constraints"] = list(reason.unmatched_constraints)
    raw["day8_matched_count"] = reason.matched_count
    raw["day8_constraint_count"] = reason.constraint_count
    result["raw"] = raw
    return result


def _best_known_evidence(
    constraints: Sequence[CanonicalConstraint],
    evidence: Sequence[ConstraintEvidence],
) -> dict[str, ConstraintEvidence]:
    known_ids = {constraint.id for constraint in constraints}
    selected: dict[str, ConstraintEvidence] = {}

    for item in evidence:
        if item.constraint_id not in known_ids:
            continue

        current = selected.get(item.constraint_id)
        if current is None:
            selected[item.constraint_id] = item
            continue

        # A matched row is always more useful than an unmatched duplicate.
        if item.matched and not current.matched:
            selected[item.constraint_id] = item
            continue

        # For duplicate matched rows, keep the stronger structured evidence.
        # This is deterministic and affects presentation only.
        if item.matched and current.matched:
            current_key = (
                _field_rank(current.evidence_field),
                -float(current.confidence),
                current.match_type,
                current.evidence_text or "",
            )
            item_key = (
                _field_rank(item.evidence_field),
                -float(item.confidence),
                item.match_type,
                item.evidence_text or "",
            )
            if item_key < current_key:
                selected[item.constraint_id] = item

    return selected


def _field_rank(field: str | None) -> int:
    try:
        return _FIELD_ORDER.index(field or "")
    except ValueError:
        return len(_FIELD_ORDER)


def _build_tags(
    matched_constraints: Sequence[CanonicalConstraint],
    matched_evidence: Sequence[ConstraintEvidence],
) -> tuple[str, ...]:
    tags: list[str] = []

    # Stable order follows canonical constraint order.
    for constraint in matched_constraints:
        tag = _CONSTRAINT_TYPE_TAGS.get(constraint.constraint_type)
        if tag and tag not in tags:
            tags.append(tag)

    # Stable source order is title -> abstract -> concept.
    fields = {item.evidence_field for item in matched_evidence if item.evidence_field}
    for field in _FIELD_ORDER:
        if field in fields:
            tag = _FIELD_TAGS[field]
            if tag not in tags:
                tags.append(tag)

    return tuple(tags)


def _build_reason_text(
    *,
    matched_constraints: Sequence[CanonicalConstraint],
    matched_evidence: Sequence[ConstraintEvidence],
    constraint_count: int,
) -> str:
    if not matched_constraints:
        return "No verifiable constraint-matching evidence was found."

    names = [constraint.canonical_text for constraint in matched_constraints]
    constraint_phrase = _format_quoted_list(names)

    if len(names) <= 2:
        prefix = (
            f"Matches the query constraint {constraint_phrase}"
            if len(names) == 1
            else f"Matches the query constraints {constraint_phrase}"
        )
    else:
        prefix = (
            f"Matches {len(names)} of {constraint_count} query constraints, "
            f"including {constraint_phrase}"
        )

    fields = {
        item.evidence_field
        for item in matched_evidence
        if item.evidence_field in _FIELD_LABELS
    }
    ordered_fields = [field for field in _FIELD_ORDER if field in fields]
    if not ordered_fields:
        return prefix + "."

    labels = [_FIELD_LABELS[field] for field in ordered_fields]
    source_phrase = _format_plain_list(labels)
    verb = "is" if len(labels) == 1 else "is"
    return f"{prefix}; supporting evidence {verb} present in the {source_phrase}."


def _format_quoted_list(items: Sequence[str]) -> str:
    quoted = [f"“{item}”" for item in items]
    return _format_plain_list(quoted)


def _format_plain_list(items: Sequence[str]) -> str:
    values = list(items)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"
