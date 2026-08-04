from scholarpath.retrieval.dates import parse_benchmark_cutoff


def test_parse_cutoff_matches_pasa_offset() -> None:
    assert parse_benchmark_cutoff("20241001", 7) == "2024-09-24"
    assert parse_benchmark_cutoff("20241001", 0) == "2024-10-01"
