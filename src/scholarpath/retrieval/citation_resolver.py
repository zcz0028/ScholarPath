from __future__ import annotations

from typing import Any


def resolve_citation_path(
    *,
    paper,
    qid,
    seed_rank,
    artifact_paths,
    fallback_builder=None,
):

    openalex_id = str(
        paper.get("openalex_id") or ""
    )


    # 1. 优先 artifact
    if openalex_id:

        path = artifact_paths.get(
            openalex_id
        )

        if path:
            return path


    # 2. 没有 artifact 才 fallback
    if fallback_builder:

        try:
            return fallback_builder(
                paper=paper,
                qid=qid,
                seed_rank=seed_rank,
            )

        except Exception:
            return None


    return None