from scholarpath.fusion.candidate_fusion import (
    CandidateSource,
    FusionConfig,
    fuse_candidate_records,
    merge_paper_metadata,
)


def test_merge_paper_metadata_preserves_longer_abstract_and_ids() -> None:
    base = {
        "title": "Paper",
        "arxiv_id": "2401.00001",
        "abstract": "short",
        "raw": {"a": 1, "b2_1_covered_constraints": ["c1"]},
    }
    incoming = {
        "title": "Paper",
        "doi": "10.123/test",
        "abstract": "this is a longer abstract",
        "raw": {"b": 2, "b2_1_covered_constraints": ["c2"]},
    }

    merged = merge_paper_metadata(base, incoming)
    assert merged["doi"] == "10.123/test"
    assert merged["abstract"] == "this is a longer abstract"
    assert merged["raw"]["a"] == 1
    assert merged["raw"]["b"] == 2
    assert merged["raw"]["b2_1_covered_constraints"] == ["c1", "c2"]


def test_fuse_candidate_records_deduplicates_and_tracks_sources() -> None:
    b0 = CandidateSource(name="b0", directory="unused", weight=1.0)
    b1 = CandidateSource(name="b1_2", directory="unused", weight=1.0)

    record0 = {
        "qid": "q1",
        "question": "test",
        "papers": [
            {"title": "Shared Paper", "arxiv_id": "2401.00001"},
            {"title": "Only B0", "arxiv_id": "2401.00002"},
        ],
    }
    record1 = {
        "qid": "q1",
        "question": "test",
        "papers": [
            {"title": "Shared Paper", "arxiv_id": "2401.00001v2"},
            {"title": "Only B1", "arxiv_id": "2401.00003"},
        ],
    }

    fused, stats = fuse_candidate_records(
        qid="q1",
        question="test",
        source_records=[(b0, record0), (b1, record1)],
        config=FusionConfig(max_fused_candidates=10),
    )

    assert len(fused["papers"]) == 3
    shared = next(p for p in fused["papers"] if p["title"] == "Shared Paper")
    assert set(shared["raw"]["b4_sources"]) == {"b0", "b1_2"}
    assert shared["raw"]["b4_source_count"] == 2
    assert stats["multi_source_candidates"] >= 1


def test_fuse_candidate_records_respects_cap() -> None:
    source = CandidateSource(name="b0", directory="unused", weight=1.0)
    record = {
        "qid": "q1",
        "question": "test",
        "papers": [
            {"title": f"Paper {i}", "arxiv_id": f"2401.{i:05d}"}
            for i in range(20)
        ],
    }

    fused, _ = fuse_candidate_records(
        qid="q1",
        question="test",
        source_records=[(source, record)],
        config=FusionConfig(max_fused_candidates=5),
    )

    assert len(fused["papers"]) == 5
