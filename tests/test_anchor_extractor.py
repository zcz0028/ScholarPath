from scholarpath.query.anchor_extractor import extract_anchors


def by_type(question: str, anchor_type: str):
    result = extract_anchors(question)
    return [anchor for anchor in result.anchors if anchor.anchor_type == anchor_type]


def test_extracts_known_benchmarks_and_preserves_source_span():
    question = "Find a code benchmark harder than HumanEval and MBPP."
    anchors = by_type(question, "benchmark")
    texts = {anchor.text.casefold() for anchor in anchors}
    assert "humaneval" in texts
    assert "mbpp" in texts
    assert all(anchor.source_span for anchor in anchors)


def test_extracts_method():
    anchors = by_type("Compare RLHF and DPO for language-model alignment.", "method")
    assert {anchor.text.casefold() for anchor in anchors} >= {"rlhf", "dpo"}


def test_extracts_negative_survey_constraint():
    anchors = by_type("Find event extraction papers, but exclude surveys.", "negative_constraint")
    assert anchors
    assert anchors[0].metadata["normalized_value"] == "exclude_survey"


def test_extracts_difficulty_comparison():
    anchors = by_type(
        "Find a benchmark harder than HumanEval but easier than CodeContests.",
        "comparison",
    )
    assert anchors
    assert anchors[0].metadata["relation"] == "difficulty_between"
    assert anchors[0].metadata["arguments"] == ["HumanEval", "CodeContests"]


def test_prompt_noise_is_not_an_anchor():
    result = extract_anchors("Show me papers and share insights about robot task planning.")
    normalized = {anchor.normalized_text for anchor in result.anchors}
    assert "show me" not in normalized
    assert "share insights" not in normalized
    assert "robot task planning" in normalized


def test_extracts_explicit_paper_title():
    anchors = by_type(
        'Find the paper titled "Attention Is All You Need" and related work.',
        "paper_title",
    )
    assert any(anchor.text == "Attention Is All You Need" for anchor in anchors)


def test_extracts_temporal_constraint():
    anchors = by_type("Find multimodal-agent papers after 2022.", "temporal_constraint")
    assert anchors
    assert anchors[0].metadata["relation"] == "year_from"
    assert anchors[0].metadata["years"] == [2022]
