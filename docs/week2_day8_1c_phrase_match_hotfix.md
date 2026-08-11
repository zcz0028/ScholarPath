# Day8-1C Hotfix — Phrase Matching Performance

## Symptom

Full Day8 evaluation over Top100 predictions was interrupted inside:

`constraint_evidence._normalized_phrase_in_text()`

The Day8-1B implementation compiled/executed a regular expression for every
constraint × paper × field × alias check. This is correct on small unit tests
but unnecessarily expensive on the full benchmark, especially when abstracts
are long.

## Fix

Replace regex phrase matching with normalized, token-boundary padded-string
matching:

```python
needle = f" {normalized_phrase} "
haystack = f" {normalized_text} "
return needle in haystack
```

This is **not raw source-string contains matching**.

Both values still pass through ScholarPath normalization first:

- Unicode normalization;
- case folding;
- hyphen/dash normalization;
- punctuation normalization;
- whitespace normalization;
- conservative token normalization.

The surrounding spaces require the complete normalized token sequence, so:

- `model` matches `a model for reasoning`;
- `model` does not match `modeling`;
- `graph neural network` matches `Graph-Neural Networks`;
- `large language model` does not match `large multimodal language model`.

## Scope

No change to:

- constraint normalization/dedup;
- residual recovery;
- evidence priority;
- token-coverage thresholds;
- E0/E1/E2/E3;
- coverage formula;
- NDCG evaluation;
- API/frontend.

## Verify

```cmd
pytest -q
```

Then rerun the exact Day8-1C command.
