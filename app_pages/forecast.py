import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import pick, sc_label, regime_picker, analyze, run_sc, data_badge, fig_base, SC_COLOR, INK
from core.gap import forecast
from core.params import LAG_DIVISOR

ss = st.session_state
st.title("격차 전망")
st.write("현재 자료의 추세로 프런티어를 연장하고, 시나리오별로 한국 역량의 경로를 시뮬레이션해 2031년까지의 격차를 전망합니다.")
data_badge()
R = analyze()

st.subheader("프런티어 추세의 연장")
fc = forecast(R)
f = fig_base(ytitle="Epoch 역량지수", h=360)
f.add_trace(go.Scatter(x=fc["연도"], y=fc["프런티어_상한"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
f.add_trace(go.Scatter(x=fc["연도"], y=fc["프런티어_하한"], fill="tonexty", fillcolor="rgba(82,81,78,0.15)", line=dict(width=0), name="프런티어(기간 민감도 범위)"))
f.add_trace(go.Scatter(x=fc["연도"], y=fc["프런티어"], line=dict(color="#0b0b0b", width=2), name=f"프런티어(연 {R['g_fr']:.1f}점)"))
f.add_trace(go.Scatter(x=fc["연도"], y=fc["오픈웨이트"], line=dict(color=INK, width=2, dash="dash"), name=f"오픈웨이트 최상위(연 {R['g_op']:.1f}점)"))
st.plotly_chart(f)
st.caption("각 연도 3분기 말 기준. 범위는 폐쇄형 상승 속도의 기간 민감도(2023～2025 시작, 최근 12～36개월 평균)의 최소～최대.")

st.subheader("시나리오별 한국 역량과 격차")
c = st.columns([3, 1, 1, 1])
keys = pick(c[0], "시나리오", list(ss.scen.keys()), sc_label, default=["S1", "S4b"], multi=True)
N = c[1].selectbox("반복 횟수", [500, 1000, 3000], index=1)
pacing, _reg = regime_picker(c[3], "fc_regime")
closure = c[2].toggle("흡수 경로 차단", help="해외 진보의 흡수항을 0으로 두는 극단 조건(본보고서 Ⅳ장 8절)")
keys = [k for k in keys if ss.scen[k]["own_frontier"]]
if keys:
    res = {k: run_sc(k, N=N, closure=closure, pacing=pacing) for k in keys}
    f = fig_base(ytitle="Epoch 역량지수", h=420)
    t0 = res[keys[0]]["traj"]
    f.add_trace(go.Scatter(x=t0["years"], y=t0["F_p50"], line=dict(color="#0b0b0b", width=2), name="프런티어(중앙값)"))
    f.add_trace(go.Scatter(x=t0["years"], y=t0["Q_p50"], line=dict(color=INK, width=2, dash="dash"), name="오픈웨이트 최상위(중앙값)"))
    for k in keys:
        t = res[k]["traj"]; col = SC_COLOR.get(k, "#2a78d6")
        f.add_trace(go.Scatter(x=t["years"] + t["years"][::-1], y=t["A_p90"] + t["A_p10"][::-1], fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=t["years"], y=t["A_p50"], line=dict(color=col, width=2), marker=dict(size=8), mode="lines+markers", name=f"{k} {ss.scen[k]['name']}"))
    st.plotly_chart(f)
    f2 = fig_base(ytitle="프런티어 대비 격차(년)", h=340)
    for k in keys:
        t = res[k]["traj"]
        f2.add_trace(go.Scatter(x=t["years"], y=t["lag_f"], mode="lines+markers", line=dict(color=SC_COLOR.get(k), width=2), marker=dict(size=8), name=f"{k} {ss.scen[k]['name']}"))
    f2.add_hline(y=1.0, line=dict(color=INK, width=1, dash="dot"), annotation_text="게이트 상한 1년")
    st.plotly_chart(f2)
    rows = []
    for k in keys:
        r = res[k]
        rows.append({"시나리오": f"{k} {ss.scen[k]['name']}", "2029 프런티어 격차(년)": r["lag_front"][1], "10%": r["lag_front"][0], "90%": r["lag_front"][2],
                     "1년 이하 확률": r["P_lagF_le1"], "2031 평균 격차(년)": r["traj"]["lag_f"][-1], "2029 도메인 게이트 확률": r["G2029_dom"]})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
    if closure:
        st.warning("흡수 경로 차단은 확률을 부여하지 않은 조건부 추정입니다. 인용 시 '흡수 경로가 막힐 경우'라는 조건을 함께 적어 주세요.")
else:
    st.info("자체 모델이 있는 시나리오를 하나 이상 고르세요.")
