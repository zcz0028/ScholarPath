from scholarpath.evaluation.matching import (
    audit_possible_matches,
    deduplicate_predictions,
    deduplicate_predictions_v2,
    match_papers,
)
from scholarpath.paper.schema import PaperRecord


def test_strict_v2_links_arxiv_doi_to_arxiv_id() -> None:
    predictions = [
        PaperRecord(title="Different metadata title", doi="10.48550/arXiv.2401.01234")
    ]
    gold = [PaperRecord(title="Gold title", arxiv_id="2401.01234v2")]
    assert match_papers(predictions, gold, mode="strict").tp == 0
    result = match_papers(predictions, gold, mode="strict_v2")
    assert result.tp == 1
    assert result.pairs[0].match_type == "strong_identifier_v2"


def test_strict_v2_extracts_arxiv_from_url() -> None:
    predictions = [PaperRecord(title="A", url="https://arxiv.org/abs/2301.00001v3")]
    gold = [PaperRecord(title="B", arxiv_id="2301.00001")]
    assert match_papers(predictions, gold, mode="strict_v2").tp == 1


def test_legacy_strict_is_unchanged() -> None:
    predictions = [PaperRecord(title="A", url="https://arxiv.org/abs/2301.00001")]
    gold = [PaperRecord(title="B", arxiv_id="2301.00001")]
    assert match_papers(predictions, gold, mode="strict").tp == 0


def test_v2_deduplicates_preprint_and_formal_version() -> None:
    predictions = [
        PaperRecord(
            title="A Reliable Retrieval Method",
            authors=["Alice Smith"],
            year=2024,
            arxiv_id="2401.01234",
        ),
        PaperRecord(
            title="A Reliable Retrieval Method",
            authors=["Alice Smith", "Bob Lee"],
            year=2025,
            doi="10.1000/reliable",
        ),
    ]
    legacy_kept, _ = deduplicate_predictions(predictions)
    v2_kept, removed = deduplicate_predictions_v2(predictions)
    assert len(legacy_kept) == 2
    assert len(v2_kept) == 1
    assert removed[0]["reason"].startswith("preprint_formal_version")


def test_v2_does_not_merge_conflicting_first_authors() -> None:
    predictions = [
        PaperRecord(title="Identical Long Paper Title", authors=["Alice"], year=2024, arxiv_id="2401.1"),
        PaperRecord(title="Identical Long Paper Title", authors=["Bob"], year=2024, doi="10.1000/x"),
    ]
    kept, _ = deduplicate_predictions_v2(predictions)
    assert len(kept) == 2


def test_near_title_is_audit_only() -> None:
    predictions = [PaperRecord(title="Efficient Scholarly Retrieval Agents")]
    gold = [PaperRecord(title="Efficient Scholarly Retrieval Agent")]
    assert match_papers(predictions, gold, mode="strict_v2").tp == 0
    candidates = audit_possible_matches(predictions, gold, title_similarity_threshold=0.8)
    assert len(candidates) == 1
