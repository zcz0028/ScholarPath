from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_day12_qe_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("day12_qe_finalize", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline():
    def item(tp,p,r,f):
        return {
            "counts":{"tp":tp,"fp":0,"fn":0},
            "macro":{"precision":p,"recall":r,"f1":f,f"recall@{20 if tp==58 else 50 if tp==81 else 100}":r},
            "micro":{"precision":p,"recall":r,"f1":f},
        }
    return {"evaluation":{
        "top20_strict":item(58,0.058,0.080,0.040),
        "top50_strict":item(81,0.032,0.140,0.046),
        "top100_strict":item(105,0.021,0.1675,0.0342),
    }}


def metric(tp,p,r,f):
    return {
        "tp":tp,"fp":0,"fn":0,
        "macro_precision":p,"macro_recall":r,"macro_f1":f,
        "micro_precision":p,"micro_recall":r,"micro_f1":f,
        "recall_at_k":r,"duplicates_removed":0,"predictions_after_dedup":100,
    }


def test_no_gain_is_rejected():
    m=load_module()
    c=m.build_comparison(
        baseline(),
        {"20":metric(58,0.058,0.080,0.040),"50":metric(81,0.032,0.140,0.046),"100":metric(105,0.021,0.1675,0.0342)},
        {"20":1.0,"50":1.0,"100":1.0},
        {"actual_api_calls":9,"cache_hits":0,"retries":0,"estimated_cost_usd":0.009},
        9,
    )
    assert c["metrics_by_k"]["100"]["delta"]["tp"] == 0
    assert c["adopt_candidate"] is False
    assert c["decision"] == "reject_qe_r9_keep_frozen_pipeline"
    assert c["efficiency"]["api_calls_per_added_top100_tp"] is None


def test_positive_gain_can_be_candidate():
    m=load_module()
    c=m.build_comparison(
        baseline(),
        {"20":metric(60,0.057,0.085,0.041),"50":metric(85,0.033,0.150,0.048),"100":metric(112,0.022,0.180,0.037)},
        {"20":1.0,"50":1.0,"100":1.0},
        {"actual_api_calls":9,"cache_hits":0,"retries":0,"estimated_cost_usd":0.009},
        9,
    )
    assert c["adopt_candidate"] is True
    assert c["decision"] == "candidate_for_e3_integration"


def test_log_aggregation():
    m=load_module()
    r=m.aggregate_retrieval_logs([
        {"actual_api_calls":1,"cache_hits":0,"retries":1,"estimated_cost_usd":0.001},
        {"actual_api_calls":0,"cache_hits":1,"retries":0,"estimated_cost_usd":0.0},
    ])
    assert r["actual_api_calls"] == 1
    assert r["cache_hits"] == 1
    assert r["retries"] == 1


def test_report_contains_final_decision():
    m=load_module()
    c=m.build_comparison(
        baseline(),
        {"20":metric(58,0.058,0.080,0.040),"50":metric(81,0.032,0.140,0.046),"100":metric(105,0.021,0.1675,0.0342)},
        {"20":1.0,"50":1.0,"100":1.0},
        {"actual_api_calls":9,"cache_hits":0,"retries":0,"estimated_cost_usd":0.009},
        9,
    )
    report=m.render_report(c)
    assert "Final QE-R9 Ablation Report" in report
    assert "reject_qe_r9_keep_frozen_pipeline" in report
    assert "Frozen Day8 E3 modified: **No**" in report
