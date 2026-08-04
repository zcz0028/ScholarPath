from scholarpath.paper.schema import PaperRecord
from scholarpath.query.constraints import QueryConstraint
from scholarpath.retrieval.constraint_aggregation import (
    aggregate_constraint_hits,
    textual_constraint_coverage,
)


def test_textual_constraint_coverage() -> None:
    constraints = [
        QueryConstraint(
            id="c1_scaling_law",
            text="scaling law",
            constraint_type="method_or_property",
            terms=["scaling", "law"],
        ),
        QueryConstraint(
            id="c2_video_text",
            text="video text",
            constraint_type="task_or_modality",
            terms=["video", "text"],
        ),
    ]
    paper = PaperRecord(title="Scaling Laws for Video Text Models")
    covered = textual_constraint_coverage(paper, constraints)
    assert "c1_scaling_law" in covered
    assert "c2_video_text" in covered


def test_constraint_aggregation_prefers_more_coverage() -> None:
    constraints = [
        QueryConstraint(
            id="c1_scaling_law",
            text="scaling law",
            constraint_type="method_or_property",
            terms=["scaling", "law"],
        ),
        QueryConstraint(
            id="c2_video_text",
            text="video text",
            constraint_type="task_or_modality",
            terms=["video", "text"],
        ),
    ]
    high = PaperRecord(title="Scaling Laws for Video Text Models", arxiv_id="2401.00001")
    low = PaperRecord(title="Generic Video Models", arxiv_id="2401.00002")

    merged = aggregate_constraint_hits(
        [
            (low, 1, "pairwise_constraint", "video text", ["c2_video_text"]),
            (high, 10, "pairwise_constraint", "scaling law video text", ["c1_scaling_law", "c2_video_text"]),
        ],
        constraints=constraints,
    )

    assert merged[0].title == "Scaling Laws for Video Text Models"
    assert merged[0].raw["b2_coverage_count"] >= 2
    assert "b2_covered_constraints" in merged[0].raw


def test_constraint_aggregation_deduplicates() -> None:
    constraints = [
        QueryConstraint(
            id="c1",
            text="retrieval augmented generation",
            constraint_type="method_or_property",
            terms=["retrieval", "augmented", "generation"],
        )
    ]
    p1 = PaperRecord(title="RAG Paper", arxiv_id="2401.00001")
    p2 = PaperRecord(title="RAG Paper", arxiv_id="2401.00001v2")

    merged = aggregate_constraint_hits(
        [
            (p1, 3, "constraint_core", "retrieval augmented generation", ["c1"]),
            (p2, 5, "keyword_fallback", "RAG", ["c1"]),
        ],
        constraints=constraints,
    )

    assert len(merged) == 1
    assert merged[0].raw["b2_occurrence_count"] == 2
