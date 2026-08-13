from scholarpath.query.expansion import (
    expand_query,
)



def test_image_encoding_expansion():

    result = expand_query(
        "image encoding distributions"
    )


    assert (
        "compression"
        in result.expanded_terms
    )


    assert (
        "latent representation"
        in result.expanded_terms
    )



def test_llm_expansion():

    result = expand_query(
        "large language model pretraining"
    )


    assert (
        "LLM"
        in result.expanded_terms
    )



def test_no_match_keeps_query():

    result = expand_query(
        "unknown topic"
    )


    assert (
        result.expanded_terms
        ==
        [
            "unknown topic"
        ]
    )