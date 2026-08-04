from scholarpath.paper.schema import PaperRecord
from scholarpath.retrieval.aggregation import (
    aggregate_candidate_hits,
    candidate_key,
    raw_anchor_score,
)


def test_candidate_key_prefers_strong_id() -> None:
    paper = PaperRecord(title="A Paper", arxiv_id="2401.01234v2")
    assert candidate_key(paper) == "arxiv:2401.01234"


def test_aggregation_deduplicates_and_ranks_occurrences() -> None:
    paper_a1 = PaperRecord(title="Relevant Paper", arxiv_id="2401.01234", source="openalex")
    paper_a2 = PaperRecord(title="Relevant Paper", arxiv_id="2401.01234v2", source="openalex")
    paper_b = PaperRecord(title="Other Paper", arxiv_id="2402.00001", source="openalex")

    merged = aggregate_candidate_hits(
        [
            (paper_b, 1, "raw", "raw query"),
            (paper_a1, 10, "raw", "raw query"),
            (paper_a2, 1, "expanded", "expanded query"),
        ]
    )

    assert len(merged) == 2
    assert merged[0].arxiv_id == "2401.01234"
    assert merged[0].raw["b1_occurrence_count"] == 2
    assert "expanded" in merged[0].raw["b1_variant_types"]


def test_aggregation_uses_title_when_no_identifier() -> None:
    paper_a = PaperRecord(title="Same Title", source="a")
    paper_b = PaperRecord(title="Same   Title!", source="b")
    merged = aggregate_candidate_hits(
        [
            (paper_a, 1, "raw", "q1"),
            (paper_b, 2, "cleaned", "q2"),
        ]
    )
    assert len(merged) == 1
    assert merged[0].raw["b1_occurrence_count"] == 2


def test_raw_anchor_downweights_keyword_only_candidates() -> None:
    raw_candidate = PaperRecord(title="Raw Rank Twenty", arxiv_id="2401.00020")
    keyword_candidate = PaperRecord(title="Keyword Rank One", arxiv_id="2402.00001")

    merged = aggregate_candidate_hits(
        [
            (keyword_candidate, 1, "keyword_core", "keyword query"),
            (raw_candidate, 20, "raw", "raw query"),
        ],
        mode="raw_anchor",
    )

    assert merged[0].title == "Raw Rank Twenty"
    assert merged[0].raw["b1_aggregation_mode"] == "raw_anchor"
    assert "b1_raw_anchor_score" in merged[0].raw


def test_raw_anchor_still_rewards_multi_variant_support() -> None:
    paper_a = PaperRecord(title="Multi Support", arxiv_id="2401.00001")
    paper_b = PaperRecord(title="Raw Low Rank", arxiv_id="2401.00002")

    merged = aggregate_candidate_hits(
        [
            (paper_b, 70, "raw", "raw query"),
            (paper_a, 3, "keyword_core", "keyword query"),
            (paper_a, 8, "cleaned", "cleaned query"),
        ],
        mode="raw_anchor",
    )

    assert merged[0].title == "Multi Support"


def test_raw_anchor_score_positive() -> None:
    paper = PaperRecord(title="A", arxiv_id="2401.00001")
    merged = aggregate_candidate_hits([(paper, 1, "raw", "query")], mode="raw_anchor")
    assert raw_anchor_score  # function is importable
    assert merged[0].raw["b1_raw_anchor_score"] > 0
