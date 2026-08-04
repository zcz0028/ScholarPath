from scholarpath.fusion.fallback_fusion import FusionConfig, fuse_papers, fuse_prediction_records, is_high_confidence_supplement, paper_key


def paper(title, arxiv=None, raw=None):
    return {"title": title, "arxiv_id": arxiv, "raw": raw or {}}


def test_paper_key_normalizes_arxiv_version() -> None:
    assert paper_key(paper("A", "2401.01234v2")) == "arxiv:2401.01234"


def test_high_confidence_by_occurrence() -> None:
    candidate = paper("Candidate", "2401.00001", {"b1_occurrence_count": 2, "b1_best_anchor_rank": 999999, "b1_raw_anchor_score": 0.0, "b1_variant_types": ["keyword_core"]})
    assert is_high_confidence_supplement(candidate, FusionConfig())


def test_high_confidence_by_anchor_rank() -> None:
    candidate = paper("Candidate", "2401.00001", {"b1_occurrence_count": 1, "b1_best_anchor_rank": 12, "b1_raw_anchor_score": 0.0, "b1_variant_types": ["cleaned"]})
    assert is_high_confidence_supplement(candidate, FusionConfig())


def test_fuse_papers_keeps_base_prefix_then_inserts_supplement() -> None:
    base = [paper("Base 1", "2401.00001"), paper("Base 2", "2401.00002"), paper("Base 3", "2401.00003")]
    supplement = [
        paper("New High Confidence", "2401.99999", {"b1_occurrence_count": 2, "b1_best_anchor_rank": 30, "b1_raw_anchor_score": 0.05, "b1_variant_types": ["cleaned", "keyword_core"]}),
        paper("Low Confidence", "2401.88888", {"b1_occurrence_count": 1}),
    ]
    fused, stats = fuse_papers(base, supplement, FusionConfig(base_keep_top=2, supplement_slots=1, max_output=4))
    assert [item["title"] for item in fused] == ["Base 1", "Base 2", "New High Confidence", "Base 3"]
    assert stats.base_kept == 2
    assert stats.supplement_inserted == 1


def test_fuse_prediction_records() -> None:
    base_record = {"qid": "q1", "question": "Question", "papers": [paper("Base", "2401.00001")]}
    supplement_record = {"qid": "q1", "question": "Question", "papers": [paper("Supplement", "2401.00002", {"b1_occurrence_count": 2, "b1_best_anchor_rank": 20, "b1_raw_anchor_score": 0.05, "b1_variant_types": ["cleaned", "keyword_core"]})]}
    fused, stats = fuse_prediction_records(base_record, supplement_record, FusionConfig(base_keep_top=1, supplement_slots=1, max_output=2))
    assert fused["qid"] == "q1"
    assert len(fused["papers"]) == 2
    assert stats.supplement_inserted == 1
