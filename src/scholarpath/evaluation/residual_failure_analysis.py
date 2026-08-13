from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


_STOPWORDS = {
    "the",
    "a",
    "an",
    "for",
    "of",
    "and",
    "with",
    "using",
    "research",
    "paper",
    "papers",
}


_CONCEPT_EXPANSIONS = {
    "image": {
        "image",
        "visual",
        "pixel",
        "vision",
    },
    "encoding": {
        "encoding",
        "encoder",
        "compression",
        "entropy",
        "latent",
        "representation",
    },
    "distribution": {
        "distribution",
        "probability",
        "density",
        "prior",
        "generative",
    },
    "model": {
        "model",
        "network",
        "architecture",
    },
}


@dataclass(slots=True)
class ResidualFailureDiagnosis:
    qid: str
    question: str

    failure_family: str
    failure_reason: str

    query_terms: list[str]
    missing_academic_concepts: list[str]

    gold_count: int

    recommended_rescue: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def normalize_tokens(text: str) -> list[str]:
    tokens = re.findall(
        r"[a-zA-Z][a-zA-Z\-]+",
        text.lower(),
    )

    return [
        x
        for x in tokens
        if x not in _STOPWORDS
    ]



def expand_term(term: str) -> set[str]:
    for key, values in _CONCEPT_EXPANSIONS.items():
        if term in values:
            return values

    return {term}



def extract_gold_terms(records: Iterable[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()

    for item in records:
        paper = item.get("paper", {})

        title = paper.get("title", "")

        terms.update(
            normalize_tokens(title)
        )

    return terms



def analyze_residual_failure(
    item: dict[str, Any],
) -> ResidualFailureDiagnosis:

    question = item.get("question", "")

    query_terms = normalize_tokens(question)

    gold_records = (
        item.get("remaining_false_negatives")
        or []
    )

    gold_terms = extract_gold_terms(
        gold_records
    )

    missing = []

    for token in query_terms:

        expanded = expand_term(token)

        candidates = expanded & gold_terms

        if not candidates:
            missing.append(token)


    if len(query_terms) <= 4:
        reason = (
            "underspecified academic query "
            "with vocabulary mismatch"
        )
    else:
        reason = (
            "query terminology does not "
            "cover gold paper concepts"
        )


    rescue = []

    if any(
        x in query_terms
        for x in [
            "encoding",
            "encode",
            "distribution",
        ]
    ):
        rescue.extend(
            [
                "expand surface terms into academic concepts",
                "add latent representation and compression terminology",
            ]
        )

    else:
        rescue.append(
            "generate academic synonym expansion"
        )


    return ResidualFailureDiagnosis(
        qid=item.get("qid", ""),
        question=question.strip(),

        failure_family=item.get(
            "b4_failure_family",
            "retrieval",
        ),

        failure_reason=reason,

        query_terms=query_terms,

        missing_academic_concepts=sorted(
            missing
        ),

        gold_count=len(gold_records),

        recommended_rescue=list(
            dict.fromkeys(rescue)
        ),
    )



def analyze_file(
    input_path: Path,
) -> list[dict[str, Any]]:

    results = []

    with input_path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            item = json.loads(line)

            diagnosis = analyze_residual_failure(
                item
            )

            results.append(
                diagnosis.to_dict()
            )

    return results



def write_report(
    path: Path,
    records: list[dict[str, Any]],
):

    lines = [
        "# Day12-2 Residual Failure Analysis",
        "",
        f"Residual cases: {len(records)}",
        "",
    ]

    for item in records:

        lines.extend(
            [
                "## " + item["qid"],
                "",
                "Question:",
                item["question"],
                "",
                "Failure reason:",
                item["failure_reason"],
                "",
                "Query terms:",
                ", ".join(item["query_terms"]),
                "",
                "Missing academic concepts:",
                ", ".join(
                    item[
                        "missing_academic_concepts"
                    ]
                ),
                "",
                "Recommended rescue:",
            ]
        )

        for r in item["recommended_rescue"]:
            lines.append(
                "- " + r
            )

        lines.append("")


    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )