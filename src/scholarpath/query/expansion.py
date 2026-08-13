from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


ACADEMIC_EXPANSION_DICT = {

    "image": [
        "image",
        "visual",
        "vision",
        "pixel",
    ],

    "encoding": [
        "encoding",
        "encoder",
        "compression",
        "entropy coding",
        "latent representation",
    ],

    "distribution": [
        "distribution",
        "probability distribution",
        "density",
        "prior",
        "generative",
    ],

    "large language model": [
        "large language model",
        "LLM",
        "foundation model",
        "generative model",
    ],

    "graph neural network": [
        "graph neural network",
        "GNN",
        "graph representation learning",
        "message passing neural network",
    ],

    "molecular property": [
        "molecular property prediction",
        "chemical property prediction",
        "molecular modeling",
    ],
}


@dataclass(slots=True)
class QueryExpansionResult:

    original_query: str

    expanded_terms: list[str]

    expanded_query: str


    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def expand_query(
    query: str,
) -> QueryExpansionResult:

    lower = query.lower()

    terms = []


    for key, values in ACADEMIC_EXPANSION_DICT.items():

        if key in lower:

            for item in values:

                if item not in terms:
                    terms.append(item)


    if not terms:
        terms = [
            query
        ]


    expanded_query = (
        query
        +
        " "
        +
        " ".join(terms)
    )


    return QueryExpansionResult(
        original_query=query,
        expanded_terms=terms,
        expanded_query=expanded_query,
    )