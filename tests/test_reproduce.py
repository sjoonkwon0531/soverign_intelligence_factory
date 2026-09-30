"""본보고서 r1(2026. 9. 30.) 수치 재현 점검. 기본 모수·기본 시드에서 아래 값이 나와야 한다."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.params import default_params, default_scenarios, SEED
from core.sim import run_scenario
from core.costbenefit import run_cb, default_cb
from core.data_io import load_baseline
from core.gap import analyze
from core.gva_b2g import run_gva, b2g, B2G_ITEMS

P = default_params(); S = default_scenarios()


@pytest.fixture(scope="module")
def s4b():
    return run_scenario(S["S4b"], P, 3000, SEED, keep_paths=False)


def test_sim_s4b(s4b):
    assert s4b["G2029_dom"] == pytest.approx(0.30633, abs=1e-4)
    assert s4b["gap"][1] == pytest.approx(3.2737, abs=1e-3)
    assert s4b["lag_front"][1] == pytest.approx(0.99434, abs=1e-4)


def test_sim_s1():
    r = run_scenario(S["S1"], P, 3000, SEED, keep_paths=False)
    assert r["G2029_dom"] == pytest.approx(0.038, abs=1e-3)
    assert r["gap"][1] == pytest.approx(8.3626, abs=1e-3)


def test_costbenefit():
    # 본보고서 WP5 r1 입력: 누적 갭 (1.8, 3.3, 4.7)조원, 성공 확률 0.31
    cb = run_cb((1.8e12, 3.3e12, 4.7e12), 0.31, S["S4b"]["capex_krw"][:4], default_cb())
    assert cb["spc"]["net_benefit"][1] == pytest.approx(-0.20816, abs=2e-3)
    assert cb["spc"]["P_nb_pos"] == pytest.approx(0.4244, abs=2e-3)


def test_gap_layers():
    data, _ = load_baseline()
    R = analyze(data, asof="2026-09-22")
    assert R["g_fr"] == pytest.approx(15.7575, abs=1e-3)
    assert R["a1"] == pytest.approx(3.0868, abs=1e-3)


def test_gva():
    G = run_gva(["S1", "S4b"], S, P)
    assert G["S1"]["gva_total"][1] == pytest.approx(5.744, abs=0.01)
    assert G["S4b"]["gva_total"][1] == pytest.approx(10.476, abs=0.01)


def test_b2g_totals():
    t = b2g(B2G_ITEMS)
    assert [round(t[f"조달 가능 {k}"].sum(), 1) for k in "저중고"] == [7.7, 13.8, 20.7]
    assert [round(t[f"SPC 귀속 {k}"].sum(), 1) for k in "저중고"] == [3.1, 7.6, 14.5]
