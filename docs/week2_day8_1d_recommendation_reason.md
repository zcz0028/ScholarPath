# Day8-1D — Deterministic Recommendation Reason Builder

## Frozen design

Day8-1D is a presentation/explanation layer only.

It consumes:
- `CanonicalConstraint[]`
- `ConstraintEvidence[]`

It does **not**:
- inspect paper title/abstract/concepts;
- redo evidence matching;
- call an LLM/network;
- modify E3 ranking;
- infer relevance beyond evidence.

Canonical constraints remain the source of truth. Unknown evidence IDs are ignored.
Duplicate evidence cannot duplicate tags/reasons.

## Output

`RecommendationReason`:
- `reason_tags`
- `reason_text`
- `matched_constraints`
- `unmatched_constraints`
- `matched_count`
- `constraint_count`

Machine-readable tags:
- `model_or_entity_match`
- `task_or_modality_match`
- `method_or_property_match`
- `topic_match`
- `title_evidence`
- `abstract_evidence`
- `concept_evidence`

`reason_text` is deterministic English. UI localization should use the structured
fields/tags rather than LLM translation.

## Boundary behavior

- Empty canonical constraints -> empty reason text and empty collections.
- Constraints present but no matched evidence ->
  `No verifiable constraint-matching evidence was found.`
- Confidence is never presented as a probability.
- Unmatched constraints are not claimed in `reason_text`.

## Verification

```cmd
pytest -q
python scripts\inspect_day8_recommendation_reason.py
```

Expected fixed cases:
1. Generic LLM survey -> 1/3 matched, model/entity + title tags.
2. Multimodal LLM -> 2/3 matched, model/entity + task/modality + title tags.
3. GNN molecular property -> 2/2 matched, method/property + topic + title tags.
