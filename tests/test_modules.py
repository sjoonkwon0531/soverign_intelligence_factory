"""거시·평가·전략 모듈의 성질 점검."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.params import default_params, default_scenarios
from core.macro import run_macro, compare
from core.appraisal import ahp, pv
from core.strategy import option_value, discover

P = default_params(); S = default_scenarios()


def test_macro_gdp_effect_independent_of_scenario():
    o = run_macro({"a": dict(p_success=0.0, coverage=0.0, trackB=False), "b": dict(p_success=0.3, coverage=0.6, trackB=True)}, N=2000)
    assert o["a"]["gdp_gain"] == o["b"]["gdp_gain"]
    row = [r for r in compare(o, "a") if r["시나리오"] == "b"][0]
    assert row["주권 배당 합계 평균(조원)"] > 0


def test_ahp_sums_to_one():
    r = ahp({"x": {"e": 2, "p": 1}, "y": {"e": 1, "p": 3}}, {"e": 0.6, "p": 0.4})
    assert abs(sum(r["total"].values()) - 1) < 1e-9


def test_pv():
    assert abs(pv([1, 1], 0.1) - (1 + 1 / 1.1)) < 1e-12


def test_option_value_no_div_by_zero():
    r = option_value(S["S4b"], P, (2029, 5.0), N=300)
    assert r["gate_loss"] == 0 and r["saved_per_gate_point"] is None and r["gap_saved"] > 0


def test_discover_dominance():
    d = discover(S["S4b"], S["S1"], P, N=300, metric="cum_gap")
    assert d["share_worse"] == 0 and d["share_better"] > 0.9
