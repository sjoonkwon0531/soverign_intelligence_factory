import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import analyze, data_badge, fig_base, INK
from core.gap import suggest_params
from core.params import PARAM_LABELS, default_params

ss = st.session_state
st.title("격차 진단과 모수 보정")
data_badge()
c = st.columns(4)
window = c[0].selectbox("추세 기준 기간 시작", ["2024-01-01", "2023-01-01", "2025-01-01"], help="프런티어 상승 속도를 추정할 기간. 본보고서 기준은 2024-01-01")
p_lo = c[1].number_input("연간 가격 하락(보수)", 1.0, 1000.0, 21.6, help="GPT-3.5 수준 280배/22개월의 연율")
p_hi = c[2].number_input("연간 가격 하락(중간)", 1.0, 1000.0, 40.0, help="Epoch AI, 과학 문제(GPQA) 기준")
R = analyze(window=window, price=(p_lo, p_hi))

st.subheader("격차의 다층 구조")
kr_rec = sum(v for k, v in R["records_all"].items() if "Korea" in k)
layers = [
    ["진보의 생산 주체", f"{R['window'][:7]} 이후 프런티어 기록 경신 {sum(R['records_all'].values())}건 중 한국 {kr_rec}건. 국적: {R['records_all']}. 오픈웨이트 기록 경신: {R['records_open']}. 역량지수 수록 한국 모델 {R['kr_in_eci']}개", "진보를 만드는 쪽인가, 흡수하는 쪽인가"],
    ["표면 격차", f"오픈웨이트 대비 {R['lag_o']:.2f}년, 프런티어 대비 {R['lag_f']:.2f}～{R['lag_f_hi']:.2f}년(환산 오차 ±{R['lag_band']:.2f}년)", "현재 공개적으로 관측되는 격차"],
    ["정부 사업 내부 격차", f"결선 팀 모델 {R['fin_lag'][0]:.2f}～{R['fin_lag'][1]:.2f}년" if "fin_lag" in R else "결선 팀 미지정", "정부 사업 결선 모델 기준"],
]
if "a1" in R:
    layers.append(["컴퓨팅 환산 격차", f"연산량 10배 = {R['ten_x_years']:.2f}년. 1년치 진보 = 연산량 10^{R['one_year_decades']:.1f}배(약 {10**R['one_year_decades']:,.0f}배)", "연산 증설만으로는 격차를 좁히기 어려움"])
if "kr_share" in R:
    layers.append(["생산 능력 격차", f"보유 연산 세계의 {R['kr_share']}%(미국의 1/{R['us_over_kr']}), 최대 클러스터 1/{R['cluster_gap']}, 투입 격차 매년 {R['input_gap_growth'][0]:.2f}～{R['input_gap_growth'][1]:.2f}배 확대", "물리적 생산 기반"])
layers.append(["가격 환산 격차", f"표면 격차를 가격으로 환산하면 약 {R['price_x'][0]:.0f}～{R['price_x'][1]:.0f}배", "1년 늦은 모델이 받는 가격"])
layers.append(["자체 역량 격차(조건부)", "격차 전망 화면에서 흡수 경로 차단 조건으로 계산", "흡수 경로가 막힐 경우"])
st.dataframe(pd.DataFrame(layers, columns=["층위", "값", "의미"]), hide_index=True)

st.subheader("역량지수 최고 기록의 추이")
eci = ss.data["eci"]; eci = eci[eci["date"] <= pd.Timestamp(R["asof"])]
f = fig_base(ytitle="Epoch 역량지수", h=430)
for grp, col, dash, lab in (("Closed weights", "#0b0b0b", "solid", "폐쇄형"), ("Open weights", INK, "dash", "오픈웨이트")):
    s = eci[eci["Accessibility group"] == grp]
    f.add_trace(go.Scatter(x=s["date"], y=s["eci"], mode="markers", marker=dict(size=6, color="#c3c2b7"), name=f"{lab} 모델", text=s["Model"],
                           hovertemplate="%{text}<br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>", showlegend=False))
    env = R["env_closed"] if grp == "Closed weights" else R["env_open"]
    f.add_trace(go.Scatter(x=env["date"], y=env["eci"], mode="lines+markers", line=dict(shape="hv", color=col, width=2, dash=dash), marker=dict(size=8),
                           name=f"{lab} 최고 기록", text=env["Model"], hovertemplate="%{text}: %{y:.1f}<extra></extra>"))
    g = R["g_fr"] if grp == "Closed weights" else R["g_op"]; a = R["g_fr_a"] if grp == "Closed weights" else R["g_op_a"]
    xs = pd.date_range(R["window"], R["asof"], periods=20); t = xs.year + (xs.dayofyear - 1) / 365.25
    f.add_trace(go.Scatter(x=xs, y=a + g * t, mode="lines", line=dict(color=col, width=1), opacity=0.5, name=f"{lab} 추세 연 {g:.1f}점", hoverinfo="skip"))
f.update_layout(hovermode="closest")
st.plotly_chart(f)
c = st.columns(4)
c[0].metric("폐쇄형 상승 속도", f"연 {R['g_fr']:.2f}점", help=f"표준오차 {R['g_fr_se']:.2f}, 기록 {R['g_fr_n']}개")
c[1].metric("오픈웨이트 상승 속도", f"연 {R['g_op']:.2f}점", help=f"표준오차 {R['g_op_se']:.2f}")
c[2].metric("2차항(가속 검정)", f"{R['quad']:.2f}", help=f"|t| = {R['quad_t']:.2f}. 2 미만이면 가속 증거 없음")
c[3].metric("프런티어와 오픈웨이트 간격", f"{R['fr_now']-R['op_now']:.1f}점")
st.markdown("**기간 민감도**")
st.dataframe(R["slope_sens"].round(2), hide_index=True)

st.subheader("한국 모델의 위치")
st.caption(f"환산식: 역량지수 = {R['map_b']:.3f} × AA + {R['map_a']:.2f} (모델 {R['map_n']}개, 결정계수 {R['map_r2']:.3f}, 95% 예측구간 ±{R['map_pi']:.1f}점)")
kr = R["korea"].copy()
st.dataframe(kr[["model", "aa", "aa_low", "finalist", "eci", "오픈웨이트 격차(년)", "프런티어 격차(년)", "프런티어 격차 상한(년)"]].round(2), hide_index=True)

if "compute_sens" in R:
    st.subheader("연산량과 역량의 관계")
    st.dataframe(R["compute_sens"].round(2), hide_index=True)

st.subheader("시뮬레이션 모수 자동 보정")
st.write("위 산출값으로 시뮬레이션 입력 범위를 다시 정합니다. 규칙: 프런티어 상승은 추정치 ±1.25점, 시간항은 추정치 ±표준오차, 연산 효과는 표본별 추정치 범위, 출발점은 결선 최저～최상위 환산값, 간격은 현재 간격 −2.7～+2.3점.")
Q = suggest_params(R, ss.P)
rows = []
for k in ("fr_drift", "diff_rate", "a1", "A_kr0", "Afor0", "gap_open"):
    rows.append({"모수": PARAM_LABELS.get(k, "프런티어 출발 역량지수"), "현재": str(ss.P[k]), "보정 제안": str(Q[k]), "본보고서 r1": str(default_params()[k])})
st.dataframe(pd.DataFrame(rows), hide_index=True)
c = st.columns(2)
if c[0].button("보정값을 시뮬레이션에 적용", type="primary"):
    ss.P = Q; st.success("적용했습니다. 시나리오, 격자, 전망, 경제성 화면이 새 모수로 계산됩니다.")
if c[1].button("본보고서 r1 모수로 되돌리기"):
    ss.P = default_params(); st.success("되돌렸습니다.")

st.subheader("산출 추적표")
st.dataframe(R["trace"], hide_index=True)
st.download_button("산출 추적표 내려받기(csv)", R["trace"].to_csv(index=False).encode("utf-8-sig"), "gap_trace.csv", "text/csv")
