import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import pick, sc_label, regime_picker, run_sc, cached_levers, to_json, data_badge, fig_base, SC_COLOR, INK, save_run
from core.params import PARAM_LABELS, YEARS, default_params, default_scenarios, SEED
from core.sim import summary_row, run_scenario

ss = st.session_state
st.title("시나리오 시뮬레이션")
data_badge()

with st.expander("모수 설정(확률분포의 하한과 상한)", expanded=False):
    rows = []
    for k, lab in PARAM_LABELS.items():
        v = ss.P[k]
        lo, hi = (v if isinstance(v, (tuple, list)) else (v, v))
        rows.append({"키": k, "모수": lab, "하한": float(lo), "상한": float(hi), "본보고서 r1": str(default_params()[k])})
    ed = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=["키", "모수", "본보고서 r1"], key="param_editor")
    c = st.columns(3)
    if c[0].button("모수 적용"):
        P = dict(ss.P)
        for _, r in ed.iterrows():
            lo, hi = sorted([float(r["하한"]), float(r["상한"])]); P[r["키"]] = (lo, hi)
        ss.P = P; st.success("적용했습니다.")
    if c[1].button("본보고서 r1 모수로"):
        ss.P = default_params(); st.rerun()
    ss.P["krw_per_usd"] = c[2].number_input("환율(원/달러)", 800.0, 2000.0, float(ss.P["krw_per_usd"]), 10.0)
    st.caption("하한과 상한을 같게 두면 그 값으로 고정됩니다. 고정해도 다른 모수의 표본은 바뀌지 않도록 난수 순서를 유지합니다.")

with st.expander("시나리오 정의 편집", expanded=False):
    k = pick(st, "편집할 시나리오", list(ss.scen.keys()), sc_label)
    sc = ss.scen[k]
    st.caption(sc["desc"])
    df = pd.DataFrame({"연도": YEARS, "설비투자(조원)": [v / 1e12 for v in sc["capex_krw"]], "B2G 계획(조원)": [v / 1e12 for v in sc["B2G"]], "기존 전력(MW)": sc["E_pipeline_mw"]})
    ed2 = st.data_editor(df, hide_index=True, disabled=["연도"], key=f"sc_ed_{k}")
    c = st.columns(4)
    n_teams = c[0].number_input("팀 수(0이면 자체 모델 없음)", 0, 5, int(sc["n_teams"]))
    ts = c[1].number_input("학습 투입 비중", 0.0, 1.0, float(sc["train_share"]), 0.05)
    fast = c[2].checkbox("계통 패스트트랙", sc["fast_track"]); stop = c[3].checkbox("미달 시 자동 중단", sc["gate_stop"])
    c = st.columns(4)
    verif = c[0].checkbox("국가 검증 루프", sc["verif"]); fb = c[1].checkbox("오픈웨이트 배치 트랙", sc.get("open_fallback", False))
    newname = c[2].text_input("이름", sc["name"])
    c2 = st.columns(3)
    upd = dict(sc, name=newname, n_teams=int(n_teams), train_share=float(ts), fast_track=fast, gate_stop=stop, verif=verif, open_fallback=fb,
               own_frontier=int(n_teams) > 0, capex_krw=[float(v) * 1e12 for v in ed2["설비투자(조원)"]], B2G=[float(v) * 1e12 for v in ed2["B2G 계획(조원)"]],
               E_pipeline_mw=[float(v) for v in ed2["기존 전력(MW)"]])
    if c2[0].button("이 시나리오에 저장"):
        ss.scen[k] = upd; st.success("저장했습니다.")
    nk = c2[1].text_input("새 시나리오 키", "U1")
    if c2[2].button("새 시나리오로 추가"):
        ss.scen[nk] = dict(upd, desc=f"사용자 정의(원본 {k})"); st.success(f"{nk}를 추가했습니다.")
    if st.button("시나리오 정의를 기본값으로"):
        ss.scen = default_scenarios(); st.rerun()

st.subheader("실행")
c = st.columns([3, 1, 1, 1])
keys = pick(c[0], "시나리오", list(ss.scen.keys()), sc_label, default=["S1", "S1B", "S2", "S2b", "S3", "S4", "S4b"], multi=True)
N = c[1].selectbox("반복 횟수", [300, 1000, 3000], index=1, help="3,000회이면 본보고서 값을 그대로 재현합니다(기본 모수와 시드일 때).")
pacing, regime = regime_picker(c[2], "sc_regime")
seed = c[3].number_input("난수 시드", value=SEED, step=1)
if keys:
    prog = st.progress(0.0, "계산 중")
    res = {}
    for i, k in enumerate(keys):
        res[k] = run_sc(k, N=N, seed=int(seed), pacing=pacing); prog.progress((i + 1) / len(keys), f"{k} 완료")
    prog.empty()
    tab = pd.DataFrame([summary_row(k, ss.scen[k], res[k]) for k in keys])
    st.dataframe(tab.round(3), hide_index=True)
    if st.button("이 결과를 비교 목록에 저장"):
        save_run("시나리오", f"{regime}, {N}회, 시드 {seed}, 모수 {'기본' if ss.P == default_params() else '수정'}", tab.to_dict("records")); st.success("저장했습니다.")

    c1, c2 = st.columns(2)
    f = fig_base(title="2029년 도메인 게이트 통과 확률", h=360)
    f.add_trace(go.Bar(x=[f"{k}" for k in keys], y=[res[k]["G2029_dom"] for k in keys], marker_color=[SC_COLOR.get(k, "#2a78d6") for k in keys],
                       text=[f"{res[k]['G2029_dom']:.2f}" for k in keys], textposition="outside", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    f.update_layout(showlegend=False, yaxis_range=[0, max(0.1, max(res[k]["G2029_dom"] for k in keys) * 1.25)])
    c1.plotly_chart(f)
    f = fig_base(title="누적 재무 갭(조원, 중앙값과 10～90 백분위)", h=360)
    f.add_trace(go.Bar(x=keys, y=[res[k]["gap"][1] for k in keys], marker_color=[SC_COLOR.get(k, "#2a78d6") for k in keys],
                       error_y=dict(type="data", symmetric=False, array=[res[k]["gap"][2] - res[k]["gap"][1] for k in keys], arrayminus=[res[k]["gap"][1] - res[k]["gap"][0] for k in keys], color=INK),
                       hovertemplate="%{x}: %{y:.1f}조원<extra></extra>"))
    f.update_layout(showlegend=False); c2.plotly_chart(f)

    own = [k for k in keys if ss.scen[k]["own_frontier"]]
    if own:
        f = fig_base(title="프런티어 대비 격차의 평균 경로(년)", h=360)
        for k in own:
            f.add_trace(go.Scatter(x=YEARS, y=res[k]["traj"]["lag_f"], mode="lines+markers", line=dict(color=SC_COLOR.get(k), width=2), marker=dict(size=8), name=f"{k} {ss.scen[k]['name']}"))
        f.add_hline(y=1.0, line=dict(color=INK, width=1, dash="dot"))
        st.plotly_chart(f)
        kf = st.selectbox("팬차트 시나리오", own, index=own.index("S4b") if "S4b" in own else 0)
        t = res[kf]["traj"]; col = SC_COLOR.get(kf, "#2a78d6")
        f = fig_base(title=f"{kf} 자국 역량 경로(중앙값과 10～90 백분위)", ytitle="Epoch 역량지수", h=380)
        f.add_trace(go.Scatter(x=YEARS, y=t["F_p50"], line=dict(color="#0b0b0b", width=2), name="프런티어"))
        f.add_trace(go.Scatter(x=YEARS, y=t["Q_p50"], line=dict(color=INK, width=2, dash="dash"), name="오픈웨이트 최상위"))
        f.add_trace(go.Scatter(x=YEARS + YEARS[::-1], y=t["A_p90"] + t["A_p10"][::-1], fill="toself", fillcolor=col, opacity=0.18, line=dict(width=0), name="10～90 백분위"))
        f.add_trace(go.Scatter(x=YEARS, y=t["A_p50"], line=dict(color=col, width=2), mode="lines+markers", marker=dict(size=8), name="중앙값"))
        st.plotly_chart(f)

st.subheader("레버 분석")
st.caption("선택한 시나리오에서 한 요인만 바꿨을 때 2029년 도메인 게이트 확률의 변화")
lk = st.selectbox("기준 시나리오", [k for k in ss.scen if ss.scen[k]["own_frontier"]], index=[k for k in ss.scen if ss.scen[k]["own_frontier"]].index("S2b") if "S2b" in ss.scen else 0)
lN = st.select_slider("레버 반복 횟수", [300, 1000, 3000], 1000)
if st.button("레버 분석 실행"):
    with st.spinner("계산 중"):
        L = cached_levers(to_json(ss.scen[lk]), to_json(ss.P), lN, SEED)
    base = L["기준"]
    f = fig_base(h=340)
    names = list(L.keys()); vals = [L[n] for n in names]
    f.add_trace(go.Bar(y=names, x=vals, orientation="h", marker_color=["#52514e"] + ["#2a78d6" if v >= base else "#eb6834" for v in vals[1:]],
                       text=[f"{v:.2f}" for v in vals], textposition="outside", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    f.add_vline(x=base, line=dict(color=INK, dash="dot")); f.update_layout(showlegend=False, xaxis_range=[0, 1], yaxis=dict(autorange="reversed"), xaxis_title="2029년 도메인 게이트 확률")
    st.plotly_chart(f)
    save_run("레버", f"{lk}, {lN}회", L)

st.subheader("강건성 점검")
st.caption("가장 불확실한 두 모수(연산량 효과, 초기 흡수율)를 분포의 양 끝에 고정했을 때 순위가 유지되는지 확인합니다.")
if st.button("강건성 점검 실행(각 1,000회)"):
    cases = {"연산량 효과 하한": dict(a1=None, lo=True), "연산량 효과 상한": dict(a1=None, lo=False), "흡수율 하한": dict(z=True), "흡수율 상한": dict(z=False)}
    a1 = ss.P["a1"]; z0 = ss.P["zeta0"]
    tw = {"연산량 효과 하한": lambda m: m.update(a1=a1[0]), "연산량 효과 상한": lambda m: m.update(a1=a1[1]),
          "초기 흡수율 하한": lambda m: m.update(zeta0=z0[0]), "초기 흡수율 상한": lambda m: m.update(zeta0=z0[1]),
          "비관(둘 다 하한)": lambda m: m.update(a1=a1[0], zeta0=z0[0]), "낙관(둘 다 상한)": lambda m: m.update(a1=a1[1], zeta0=z0[1])}
    rows = []
    kk = [k for k in ["S1", "S2", "S2b", "S4", "S4b"] if k in ss.scen]
    with st.spinner("계산 중"):
        for name, f_ in tw.items():
            row = {"조건": name}
            for k in kk: row[k] = run_scenario(ss.scen[k], ss.P, 1000, SEED, tweak=f_, keep_paths=False)["G2029_dom"]
            rows.append(row)
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
