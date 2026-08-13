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


from scholarpath.evaluation.residual_failure_analysis import (
    analyze_file,
    write_report,
)



def main():

    input_file = (
        PROJECT_ROOT
        /
        "outputs/week2_day12_failure_analysis/"
        "residual_failures.jsonl"
    )


    output_dir = (
        PROJECT_ROOT
        /
        "outputs/week2_day12_failure_analysis"
    )


    records = analyze_file(
        input_file
    )


    json_path = (
        output_dir
        /
        "residual_failure_analysis.json"
    )


    json_path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    write_report(
        output_dir
        /
        "residual_failure_report.md",
        records,
    )


    print(
        "[Day12-2] Residual failure analysis generated."
    )

    print(
        f"[Day12-2] cases = {len(records)}"
    )



if __name__ == "__main__":
    main()