# Day9-2C Retry + Resume + Debug Cleanup

This patch adds three stability improvements:

1. OpenAlex transient retry with exponential backoff.
2. Resume/merge for `outputs/week2_day9_citation/citation_paths.jsonl`, so a later unstable run cannot overwrite a previous 19/20 artifact with lower coverage.
3. Cleanup of temporary `DAY9 ... DEBUG` prints from runtime modules.

Benchmark mode still uses artifact-only resolution with `fallback_builder=None`, so runtime citation API calls remain zero.
