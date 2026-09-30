"""앱 공통: 상태 관리, 캐시된 계산, 차트 스타일."""
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.params import default_params, default_scenarios, SEED
from core.data_io import load_baseline
from core import sim as SIM, gap as GAP, costbenefit as CB, grid as GRID, gva_b2g as GB

# 범주 색(시나리오에 고정 배정, 순위와 무관)
SC_COLOR = {"S4b": "#2a78d6", "S1": "#eb6834", "S2b": "#1baf7a", "S2": "#eda100", "S4": "#e87ba4", "S3": "#008300", "S1B": "#4a3aa7"}
INK = "#52514e"
SEQ = [[0, "#eaf2fb"], [0.5, "#6ea8e8"], [1, "#0b4a94"]]           # 크기: 단일 색상 명암
DIV = [[0, "#c2410c"], [0.5, "#e5e5e3"], [1, "#1d5fb4"]]           # 부호: 음(주황)·0(회색)·양(파랑)


def init_state():
    ss = st.session_state
    if "data" not in ss:
        ss.data, ss.meta = load_baseline()
        ss.data_label = "기준본(2026. 9. 22. 조회)"
    if "P" not in ss: ss.P = default_params()
    if "scen" not in ss: ss.scen = default_scenarios()
    if "runs" not in ss: ss.runs = []
    if "asof" not in ss: ss.asof = "2026-09-22"


def to_json(x):
    def conv(o):
        if isinstance(o, tuple): return list(o)
        if isinstance(o, (np.floating, np.integer)): return o.item()
        raise TypeError
    return json.dumps(x, default=conv, sort_keys=True, ensure_ascii=False)


def _P(pj):
    P = json.loads(pj)
    for k, v in P.items():
        if isinstance(v, list) and k != "L_choices": P[k] = tuple(v)
    P["L_choices"] = tuple(P["L_choices"])
    return P


@st.cache_data(show_spinner=False, max_entries=64)
def cached_run(sc_json, P_json, N, seed, pacing, closure=False):
    sc = json.loads(sc_json); P = _P(P_json)
    tw = (lambda m: m.update(diff_rate=0.0)) if closure else None
    return SIM.run_scenario(sc, P, N, seed, pacing, tweak=tw)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_levers(sc_json, P_json, N, seed):
    return SIM.run_levers(json.loads(sc_json), _P(P_json), N, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_grid_cell(T, s, P_json, N, seed, pacing, mode):
    return GRID.run_cell(T, s, _P(P_json), None, N, seed, pacing, mode)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_cb(gap_tri, p_success, capex_path, C_json, N, seed, keep=False):
    C = json.loads(C_json)
    for k, v in C.items():
        if isinstance(v, list): C[k] = tuple(v)
    return CB.run_cb(tuple(gap_tri), p_success, list(capex_path), C, N, seed, keep=keep)


def run_sc(key, N=1000, seed=SEED, pacing=1.0, closure=False, P=None):
    ss = st.session_state
    return cached_run(to_json(ss.scen[key]), to_json(P or ss.P), N, seed, pacing, closure)


def analyze(asof=None, window="2024-01-01", price=(21.6, 40.0)):
    """격차 산출(세션 안에서 입력이 같으면 다시 계산하지 않음)."""
    ss = st.session_state
    sig = to_json({k: v.get("sha16") for k, v in ss.meta.items()}) + ss.data["korea"].to_json() + ss.data["aa_pairs"].to_json() + str(asof or ss.asof) + window + str(price)
    memo = ss.setdefault("_gap_memo", {})
    if sig not in memo:
        memo.clear(); memo[sig] = GAP.analyze(ss.data, asof=asof or ss.asof, window=window, price_decline=tuple(price))
    return memo[sig]


def fig_base(title=None, ytitle=None, xtitle=None, h=380):
    f = go.Figure()
    f.update_layout(height=h + (30 if title else 0), margin=dict(l=10, r=10, t=85 if title else 40, b=10),
                    title=dict(text=title, x=0, xanchor="left", y=0.99, yanchor="top", font=dict(size=15)) if title else None, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), xaxis_title=xtitle, yaxis_title=ytitle)
    return f


def heatmap(z, x, y, fmt="{:.2f}", title=None, diverging=False, xtitle=None, ytitle=None, h=420):
    z = np.array(z, dtype=float)
    kw = dict(colorscale=DIV, zmid=0) if diverging else dict(colorscale=SEQ)
    f = go.Figure(go.Heatmap(z=z, x=x, y=y, text=[[fmt.format(v) for v in row] for row in z], texttemplate="%{text}",
                             hovertemplate="%{y} · %{x}: %{text}<extra></extra>", xgap=2, ygap=2, **kw))
    f.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 10, b=10), title=title, xaxis_title=xtitle, yaxis_title=ytitle)
    f.update_yaxes(type="category"); f.update_xaxes(type="category")
    return f


REGIMES = {"기준(추세 유지)": 1.0, "감속(2027년부터 속도 절반)": 0.5, "가속(2027년부터 속도 1.3배)": 1.3, "직접 입력": None}


def regime_picker(container, key="regime"):
    """프런티어 진보 속도 체제. 2027년 이후 연 상승 속도에 곱하는 배수를 돌려준다."""
    lab = container.selectbox("프런티어 진보 체제", list(REGIMES), key=key,
                              help="감속은 스케일링 둔화, 가속은 추론 연산·자동화 연구로 진보가 빨라지는 경우. 2027년 이후 프런티어 상승 속도에 곱합니다.")
    v = REGIMES[lab]
    if v is None:
        v = container.number_input("속도 배수", 0.2, 2.5, 1.0, 0.1, key=key + "_v")
    return float(v), lab


def jo(x):
    """조원 표기."""
    return "해당 없음" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.1f}조원"


def rng3(v, d=2, unit=""):
    if not v: return "해당 없음"
    return f"{v[1]:.{d}f}{unit} ({v[0]:.{d}f}～{v[2]:.{d}f})"


def save_run(kind, label, payload):
    st.session_state.runs.append(dict(kind=kind, label=label, payload=payload))


def data_badge():
    ss = st.session_state
    st.caption(f"데이터: {ss.data_label} · 기준일 {ss.asof} · 모수: {'기본(본보고서 r1)' if ss.P == default_params() else '사용자 수정'}")


# 거시·전략 모듈 캐시
from core import macro as MAC, strategy as STR, factory as FAC


def scen_stats(key, N=1000, pacing=1.0):
    """시나리오 시뮬레이션에서 거시 모듈 입력(성공 확률, 주권 커버리지, 오픈웨이트 트랙 여부)을 만든다."""
    ss = st.session_state
    if key == "무대응":
        return dict(p_success=0.0, coverage=0.0, trackB=False)
    sc = ss.scen[key]; r = run_sc(key, N=N, pacing=pacing)
    return dict(p_success=float(r["G2029_dom"]) if sc["own_frontier"] else 0.0, coverage=float(r["sov"]),
                trackB=bool(sc.get("open_fallback") or not sc["own_frontier"]))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_macro(stats_json, M_json, N, seed):
    M = json.loads(M_json)
    for k, v in M.items():
        if isinstance(v, list): M[k] = tuple(v)
    return MAC.run_macro(json.loads(stats_json), M, N, seed)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_sens(sc_json, P_json, N):
    return STR.sensitivity(json.loads(sc_json), _P(P_json), N)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_discover(a_json, b_json, P_json, N, metric, depth):
    return STR.discover(json.loads(a_json), json.loads(b_json), _P(P_json), N, metric=metric, max_depth=depth)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_options(sc_json, P_json, rules_json, N):
    return STR.option_table(json.loads(sc_json), _P(P_json), [tuple(r) for r in json.loads(rules_json)], N)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_factory_run(sc_json, P_json, fk, N, pacing):
    sc = json.loads(sc_json)
    return SIM.run_scenario(sc, _P(P_json), N, SEED, pacing, keep_paths=False, xdraw=lambda rng: FAC.draw_factory(rng, fk), xseed=7)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_factory_econ(fk, N):
    return FAC.factory_economics(fk, N=N)


def pick(container, label, options, labels, default=None, multi=False, **kw):
    """표시 이름으로 고르고 내부 키를 돌려주는 선택 상자(format_func 없이 동작해 테스트와 공유 링크에서 안정적)."""
    disp = [labels(o) if callable(labels) else labels.get(o, o) for o in options]
    back = dict(zip(disp, options))
    if multi:
        dflt = [disp[options.index(d)] for d in (default or []) if d in options]
        return [back[v] for v in container.multiselect(label, disp, default=dflt, **kw)]
    idx = options.index(default) if default in options else 0
    v = container.selectbox(label, disp, index=idx, **kw)
    return back[v] if v is not None else None


def sc_label(k):
    return k if k == "무대응" else f"{k} {st.session_state.scen[k]['name']}"
