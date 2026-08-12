from scholarpath.retrieval.citation_resolver import (
    resolve_citation_path,
)


def test_artifact_priority():

    result = resolve_citation_path(
        paper={
            "openalex_id":"W1"
        },
        qid="q1",
        seed_rank=1,
        artifact_paths={
            "W1":{
                "edge_type":"references"
            }
        },
        fallback_builder=None,
    )


    assert result["edge_type"]=="references"



def test_missing_returns_none():

    result = resolve_citation_path(
        paper={
            "openalex_id":"W2"
        },
        qid="q1",
        seed_rank=1,
        artifact_paths={},
        fallback_builder=None,
    )


    assert result is None