from pathlib import Path
import json


from scholarpath.evaluation.residual_failure_analysis import (
    analyze_residual_failure,
)



def test_residual_analysis_detects_gap():

    item = {
        "qid": "RealScholarQuery_38",
        "question":
            "Show me research on image encoding distributions.",

        "b4_failure_family":
            "retrieval",

        "remaining_false_negatives": [
            {
                "paper": {
                    "title":
                    "Learned Compression of Encoding Distributions"
                }
            }
        ],
    }


    result = analyze_residual_failure(
        item
    )


    assert result.qid == (
        "RealScholarQuery_38"
    )

    assert (
        "encoding"
        in result.query_terms
    )

    assert (
        result.failure_family
        ==
        "retrieval"
    )



def test_reason_exists():

    item = {
        "qid": "q1",
        "question":
            "image distributions",

        "remaining_false_negatives":
        [],
    }


    result = analyze_residual_failure(
        item
    )


    assert result.failure_reason
    assert len(
        result.recommended_rescue
    ) > 0