# Week 2 Day8-1B — Constraint Evidence Matcher

## Scope

Day8-1B adds deterministic evidence matching on top of the accepted Day8-1A V2 canonical-constraint layer.

It does **not**:
- modify semantic ranking;
- calculate the final evidence-aware score;
- generate `reason_tags` or `reason_text`;
- call an LLM;
- call a remote embedding service;
- change the API or frontend.

## Evidence priority

Field priority:

1. title
2. abstract
3. individual OpenAlex concept `display_name`

Within one field:

1. canonical phrase
2. alias phrase
3. token coverage

Only the best evidence for each canonical constraint is retained.

## Phrase matching

Raw-string `contains` is forbidden.

Both constraint phrases and paper text pass through Day8 normalization first. Matching then uses token boundaries.

Canonical text is explicitly tested before aliases. Matcher behavior does not depend on the stored alias tuple order.

## Token coverage

Let `n` be the number of unique normalized constraint tokens.

- `n = 1`: a generic token cannot create evidence by itself.
- `n = 2`: require `2/2`.
- `n >= 3`: require `ceil(0.8 * n)` and at least 2 tokens.

Examples with threshold `0.8`:

- 3 tokens -> 3 required
- 4 tokens -> 4 required
- 5 tokens -> 4 required

Token coverage must be satisfied inside one field. Tokens are never combined across title, abstract, or multiple concepts.

## Generic one-token protection

Generic single-token constraints such as:

`model`, `method`, `system`, `learning`, `network`, `document`, `prediction`, `analysis`

cannot produce evidence by themselves, including direct one-word phrase matching.

## Abstract handling

Priority:

1. use `paper["abstract"]` when non-empty;
2. otherwise reconstruct OpenAlex `abstract_inverted_index`;
3. otherwise use empty text.

An empty abstract never aborts the matcher. Title and concepts are still checked.

## OpenAlex concepts

`raw["concepts"]` must be a list.

Each dictionary item's `display_name` is matched independently.

Concept strings are never concatenated for evidence matching, preventing false cross-concept token coverage.

## Empty constraints

`build_constraint_evidence([], paper)` returns `[]`.

This guarantees that Day8-1C can safely define coverage `0.0` without dividing by zero.

## Match types

- `title_phrase_match`
- `title_alias_match`
- `title_token_match`
- `abstract_phrase_match`
- `abstract_alias_match`
- `abstract_token_match`
- `concept_phrase_match`
- `concept_alias_match`
- `concept_token_match`
- `none`

## Verification

Run:

```cmd
pytest -q
```

Then:

```cmd
python scripts\inspect_day8_evidence.py
```

Expected manual behavior:

### Generic LLM survey

For the multimodal/scientific-document query:

- large language model -> matched
- multimodal -> unmatched
- scientific document understanding -> unmatched

### MM-LLMs

- large language model -> matched
- multimodal -> matched
- scientific document understanding -> unmatched

### GNN molecular property paper

- graph neural network -> matched
- molecular property prediction -> matched

Only after this layer is accepted should Day8-1C coverage/reranking begin.
