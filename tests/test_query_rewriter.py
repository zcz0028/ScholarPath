from scholarpath.query.rewriter import (
    QueryRewriter,
    clean_question_text,
    expand_terms,
    extract_keywords,
)


def test_clean_question_text_removes_question_prefix() -> None:
    text = "Is there any work that analyzes the scaling law of video-text models?"
    cleaned = clean_question_text(text)
    assert cleaned.startswith("analyzes") or cleaned.startswith("scaling")
    assert "?" not in cleaned


def test_extract_keywords_keeps_technical_terms() -> None:
    keywords = extract_keywords("scaling law of video text image text models")
    assert "scaling" in keywords
    assert "video" in keywords
    assert "image" in keywords


def test_expand_terms_for_multimodal_query() -> None:
    expansions = expand_terms(
        "scaling law of multi-module video-text and image-text models"
    )
    joined = " ".join(expansions).casefold()
    assert "scaling laws" in joined
    assert "vision language" in joined
    assert "video language" in joined
    assert "multimodal" in joined


def test_rewriter_generates_distinct_variants() -> None:
    variants = QueryRewriter().rewrite(
        "Is there any work that analyzes the scaling law of the multi-module models, such as video-text, image-text models?"
    )
    assert len(variants) >= 3
    assert variants[0].variant_type == "raw"
    assert len({item.text.casefold() for item in variants}) == len(variants)
    assert any(item.variant_type == "expanded" for item in variants)
