from __future__ import annotations
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_day13_f1_diagnostic.py"

def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_day13_f1_diagnostic.py"
    )

    spec = importlib.util.spec_from_file_location(
        "day13diag",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    # Python 3.13 dataclasses may resolve postponed annotations through
    # sys.modules while the class decorator is being evaluated.
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Do not leave a partially initialized module behind.
        sys.modules.pop(spec.name, None)
        raise

    return module

def test_f1_score_basic():
    m=load_module()
    assert m.f1_score(0.5,0.5)==0.5
    assert m.f1_score(0,0.5)==0

def test_oracle_bucket():
    m=load_module()
    assert m.oracle_bucket(1)=="1-5"
    assert m.oracle_bucket(6)=="6-10"
    assert m.oracle_bucket(20)=="11-20"
    assert m.oracle_bucket(21)=="21-50"
    assert m.oracle_bucket(100)=="51-100"

def test_global_best_tie_prefers_smaller_k():
    rows=[{"k":10,"macro_f1":0.1},{"k":20,"macro_f1":0.1},{"k":5,"macro_f1":0.09}]
    best=max(rows,key=lambda r:(float(r["macro_f1"]),-int(r["k"])))
    assert best["k"]==10

def test_report_marks_gold_boundary():
    m=load_module()
    global_rows=[{"k":20,"tp":58,"fp":942,"fn":733,"macro_precision":0.058,"macro_recall":0.08089,"macro_f1":0.05757,"micro_f1":0.06477,"zero_hit_queries":20}]
    gs={"best_fixed":global_rows[0],"top20_reference":global_rows[0],"best_fixed_macro_f1_gain_vs_top20":0.0}
    osum={"oracle_macro_f1":0.08,"oracle_macro_f1_gain_vs_best_fixed":0.02243,"queries_with_no_strict_hit_in_top100":10,
          "oracle_best_k_distribution":{"1-5":10,"6-10":10,"11-20":10,"21-50":10,"51-100":10}}
    report=m.render_report(global_rows=global_rows,global_summary=gs,oracle_summary=osum)
    assert "offline evaluation/analysis only" in report
    assert "must not be reported as competition score" in report
    assert "material adaptive-cutoff headroom signal" in report
