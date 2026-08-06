from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.ranking.seed_gate import SeedGateConfig, select_seed_papers
from scholarpath.retrieval.citation_expansion import (
    CitationBudget,
    CitationBudgetExhausted,
    CitationExpansionConfig,
    OpenAlexCitationProvider,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL record in {path}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["qid"]): row for row in rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ScholarPath Week 2 Day 5 controlled citation expansion"
    )
    parser.add_argument(
        "--unrecovered",
        default="outputs/week2_day4_rescue/unrecovered_queries.jsonl",
    )
    parser.add_argument(
        "--day4-pool",
        default=(
            "outputs/week2_day4_rescue/"
            "merged_candidates_before_rerank.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/week2_day5_citation",
    )
    parser.add_argument("--max-seeds-per-query", type=int, default=3)
    parser.add_argument("--max-seed-rank", type=int, default=15)
    parser.add_argument("--max-references-per-seed", type=int, default=15)
    parser.add_argument("--max-cited-by-per-seed", type=int, default=15)
    parser.add_argument(
        "--max-unique-candidates-per-query",
        type=int,
        default=80,
    )
    parser.add_argument("--max-api-calls", type=int, default=60)
    parser.add_argument(
        "--cache-dir",
        default="data/cache/openalex_day5_citation",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    unrecovered = read_jsonl(args.unrecovered)
    target_qids = {str(row["qid"]) for row in unrecovered}
    pools = by_qid(read_jsonl(args.day4_pool))
    missing_pool_qids = sorted(target_qids - pools.keys())
    if missing_pool_qids:
        raise KeyError(
            "Target qids missing from Day 4 candidate pool: "
            + ", ".join(missing_pool_qids)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_config = SeedGateConfig(
        max_seeds_per_query=args.max_seeds_per_query,
        max_seed_rank=args.max_seed_rank,
    )

    selected_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    selected_by_qid: dict[str, list[Any]] = {}

    for qid in sorted(target_qids):
        record = pools[qid]
        question = str(record.get("question") or "")
        selected, rejected = select_seed_papers(
            qid=qid,
            question=question,
            papers=record.get("papers") or [],
            config=seed_config,
        )
        selected_by_qid[qid] = selected
        selected_rows.append(
            {
                "qid": qid,
                "question": question,
                "seeds": [item.to_dict() for item in selected],
            }
        )
        rejected_rows.extend(item.to_dict() for item in rejected)

    selected_seed_count = sum(len(items) for items in selected_by_qid.values())

    # One cited-by call plus up to two reference calls per seed:
    # 1) fetch seed metadata; 2) batch-fetch referenced works.
    estimated_cited_by_calls = selected_seed_count
    estimated_seed_metadata_calls = selected_seed_count
    estimated_reference_batch_calls = selected_seed_count
    estimated_api_calls_uncapped = (
        estimated_cited_by_calls
        + estimated_seed_metadata_calls
        + estimated_reference_batch_calls
    )
    estimated_api_calls = min(
        estimated_api_calls_uncapped,
        args.max_api_calls,
    )

    manifest = {
        "strategy": "day5_controlled_citation_expansion",
        "target_query_count": len(target_qids),
        "preserved_query_count": max(0, len(pools) - len(target_qids)),
        "selected_seed_count": selected_seed_count,
        "estimated_api_calls": estimated_api_calls,
        "estimated_api_calls_uncapped": estimated_api_calls_uncapped,
        "estimated_cited_by_calls": estimated_cited_by_calls,
        "estimated_seed_metadata_calls": estimated_seed_metadata_calls,
        "estimated_reference_batch_calls": estimated_reference_batch_calls,
        "max_api_calls": args.max_api_calls,
        "max_hops": 1,
        "max_seeds_per_query": args.max_seeds_per_query,
        "production_expansion_uses_gold": False,
        "target_qids": sorted(target_qids),
    }

    (output_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "selected_seeds.jsonl", selected_rows)
    write_jsonl(output_dir / "seed_rejections.jsonl", rejected_rows)

    if args.dry_run:
        print(f"[DRY RUN] Target queries: {len(target_qids)}")
        print(f"[DRY RUN] Preserved queries: {manifest['preserved_query_count']}")
        print(f"[DRY RUN] Selected seeds: {selected_seed_count}")
        print(
            "[DRY RUN] Estimated API calls uncapped: "
            f"{estimated_api_calls_uncapped}"
        )
        print(
            "[DRY RUN] Estimated API calls under budget: "
            f"{estimated_api_calls}"
        )
        print(f"[DRY RUN] Maximum API calls: {args.max_api_calls}")
        print("[DRY RUN] API calls executed: 0")
        print("[DRY RUN] Maximum hop: 1")
        print("[DRY RUN] Production expansion uses gold: false")
        return 0

    if not os.getenv("OPENALEX_API_KEY"):
        print("[ERROR] OPENALEX_API_KEY is not set")
        return 2

    config = CitationExpansionConfig(
        max_references_per_seed=args.max_references_per_seed,
        max_cited_by_per_seed=args.max_cited_by_per_seed,
        max_unique_candidates_per_query=args.max_unique_candidates_per_query,
        max_api_calls=args.max_api_calls,
        cache_dir=args.cache_dir,
    )
    budget = CitationBudget(max_api_calls=args.max_api_calls)
    provider = OpenAlexCitationProvider(config, budget)

    paths: list[dict[str, Any]] = []
    raw_candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_expansions: list[dict[str, Any]] = []

    budget_exhausted = False
    for index, qid in enumerate(sorted(target_qids), start=1):
        seeds = selected_by_qid[qid]
        print(f"[{index}/{len(target_qids)}] qid={qid} seeds={len(seeds)}")

        for seed_decision in seeds:
            for edge_type, fn in (
                ("cited_by", provider.get_cited_by),
                ("reference", provider.get_references),
            ):
                if budget_exhausted or not budget.can_call():
                    budget_exhausted = True
                    skipped_expansions.append(
                        {
                            "qid": qid,
                            "seed": seed_decision.paper.to_dict(include_raw=True),
                            "edge_type": edge_type,
                            "reason": "api_budget_exhausted",
                        }
                    )
                    continue

                try:
                    papers = fn(
                        seed_decision.paper,
                        refresh_cache=args.refresh_cache,
                    )
                except CitationBudgetExhausted:
                    budget_exhausted = True
                    skipped_expansions.append(
                        {
                            "qid": qid,
                            "seed": seed_decision.paper.to_dict(include_raw=True),
                            "edge_type": edge_type,
                            "reason": "api_budget_exhausted",
                        }
                    )
                    continue
                except Exception as exc:
                    failures.append(
                        {
                            "qid": qid,
                            "seed": seed_decision.paper.to_dict(include_raw=True),
                            "edge_type": edge_type,
                            "error": str(exc),
                        }
                    )
                    if not args.continue_on_error:
                        write_jsonl(
                            output_dir / "failed_expansions.jsonl",
                            failures,
                        )
                        write_jsonl(
                            output_dir / "skipped_expansions.jsonl",
                            skipped_expansions,
                        )
                        raise
                    continue

                for edge_rank, paper in enumerate(papers, start=1):
                    paths.append(
                        {
                            "qid": qid,
                            "seed_openalex_id": seed_decision.paper.openalex_id,
                            "seed_title": seed_decision.paper.title,
                            "expanded_openalex_id": paper.openalex_id,
                            "expanded_title": paper.title,
                            "edge_type": edge_type,
                            "seed_rank": seed_decision.rank,
                            "edge_rank": edge_rank,
                            "hop": 1,
                            "trigger_reason": "day4_zero_recall",
                        }
                    )
                    raw_candidates.append(
                        {
                            "qid": qid,
                            "seed_score": seed_decision.seed_score,
                            "seed_rank": seed_decision.rank,
                            "edge_type": edge_type,
                            "edge_rank": edge_rank,
                            "paper": paper.to_dict(include_raw=True),
                        }
                    )

    write_jsonl(output_dir / "citation_paths.jsonl", paths)
    write_jsonl(output_dir / "expanded_candidates_raw.jsonl", raw_candidates)
    write_jsonl(output_dir / "failed_expansions.jsonl", failures)
    write_jsonl(output_dir / "skipped_expansions.jsonl", skipped_expansions)

    summary = {
        **manifest,
        "actual_api_calls": budget.actual_api_calls,
        "cache_hits": budget.cache_hits,
        "raw_citation_candidates": len(raw_candidates),
        "citation_paths": len(paths),
        "failed_expansions": len(failures),
        "skipped_due_to_budget": len(skipped_expansions),
        "budget_exhausted": budget_exhausted,
        "next_step": (
            "Run filtering, fusion, reranking and evaluation after reviewing "
            "seed quality."
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Target queries: {len(target_qids)}")
    print(f"[OK] Selected seeds: {selected_seed_count}")
    print(f"[OK] Actual API calls: {budget.actual_api_calls}")
    print(f"[OK] Cache hits: {budget.cache_hits}")
    print(f"[OK] Raw citation candidates: {len(raw_candidates)}")
    print(f"[OK] Citation paths: {len(paths)}")
    print(f"[OK] Failed expansions: {len(failures)}")
    print(f"[OK] Skipped due to budget: {len(skipped_expansions)}")
    print(f"[OUTPUT] Summary: {output_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
