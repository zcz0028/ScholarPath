from __future__ import annotations


import json
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC = PROJECT_ROOT / "src"


if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )


from scholarpath.query.expansion import (
    expand_query,
)



INPUT = (
    PROJECT_ROOT
    /
    "data/processed/realscholarquery_gold.jsonl"
)


OUTPUT = (
    PROJECT_ROOT
    /
    "outputs/week2_day12_query_expansion"
)



def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )


    cases = []


    with INPUT.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue


            item = json.loads(line)


            result = expand_query(
                item["question"]
            )


            cases.append(
                {
                    "qid": item["qid"],

                    **result.to_dict()
                }
            )



    case_file = (
        OUTPUT
        /
        "query_expansion_cases.jsonl"
    )


    with case_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in cases:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


    summary = {

        "query_count":
            len(cases),

        "expanded_query_count":
            sum(
                bool(
                    x["expanded_terms"]
                )
                for x in cases
            ),

        "method":
            "rule_based_academic_expansion",

        "llm_calls":
            0,

        "network_calls":
            0,
    }


    (
        OUTPUT
        /
        "ablation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "[Day12-3] Query expansion ablation prepared."
    )

    print(
        "queries =",
        len(cases),
    )


if __name__ == "__main__":
    main()