import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import pick, sc_label, scen_stats, cached_macro, to_json, data_badge, fig_base, save_run, regime_picker, SC_COLOR, INK
from core.macro import MACRO_PARAMS, HORIZON

ss = st.session_state
st.title("국가 거시경제 파급")
st.write("AI가 경제 전체에 주는 효과는 공급원과 무관하게 대부분 발생합니다. 추진 방식이 바꾸는 것은 그 효과 가운데 국내에 귀속되는 몫, "
         "해외 AI 서비스 수입의 크기, 접근이 차단될 때의 손실입니다. 이 화면은 시나리오 시뮬레이션의 성공 확률과 주권 커버리지를 받아 "
         "2026～2035년 국내 귀속 부가가치(GNI 관점), 서비스 수입, 차단 손실을 현재가치로 비교합니다.")
data_badge()
if "MAC" not in ss: ss.MAC = dict(MACRO_PARAMS)
M = ss.MAC

with st.expander("거시 모수(범위 가정). 값을 바꾸면 모든 결과가 다시 계산됩니다", expanded=False):
    c = st.columns(3)
    gdp = c[0].number_input("2025 명목 GDP(조원)", 1000.0, 5000.0, M["gdp0_krw"] / 1e12, 10.0,
                            help="한국은행 국민계정 확정치로 바꿔 주세요. 기본값 2,600조원은 잠정 가정입니다.")
    gn = c[1].number_input("명목 성장률(연)", 0.0, 0.1, M["g_nominal"], 0.005, format="%.3f")
    dr = c[2].number_input("사회적 할인율", 0.0, 0.1, M["discount"], 0.005, format="%.3f", help="기획재정부 예비타당성조사 지침 4.5%(2017. 9. 이후)")
    labels = {"G_LR": "AI의 장기 GDP 수준 효과", "adopt_mid": "확산 중간 시점(연도)", "adopt_k": "확산 속도", "jcurve": "보완 무형투자 비용(J-곡선)",
              "f_front": "프런티어급이 필요한 효과 비중", "spend_ratio": "AI 서비스 지출/AI 효과", "dom_share0": "현재 국내 공급 비중",
              "capture": "SPC 성공 시 문턱 위 국산 전환율", "trackB_gain": "오픈웨이트 트랙의 국내 공급 증가", "v_dom": "국내 공급의 국내 부가가치 비율",
              "v_loc": "해외 공급의 국내 부가가치 비율", "pi_us": "미국 접근 유지 확률(4년)", "T_block": "차단 지속(년)"}
    rows = [{"키": k, "모수": lab, "하한": float(M[k][0]), "상한": float(M[k][1])} for k, lab in labels.items()]
    ed = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=["키", "모수"], key="mac_ed")
    cc = st.columns(2)
    if cc[0].button("거시 모수 적용"):
        new = dict(M, gdp0_krw=gdp * 1e12, g_nominal=gn, discount=dr)
        for _, r in ed.iterrows():
            new[r["키"]] = tuple(sorted([float(r["하한"]), float(r["상한"])]))
        ss.MAC = new; st.rerun()
    if cc[1].button("기본값으로", key="mac_reset"):
        ss.MAC = dict(MACRO_PARAMS); st.rerun()
    st.caption("장기 수준 효과 4.2～12.6%는 한국은행(2025) 이슈노트의 AI 도입 시 장기 GDP 효과 범위. 나머지는 본보고서 가정이며 근거는 방법과 출처 화면에 정리.")

c = st.columns([3, 1, 1, 1])
opts = ["무대응"] + list(ss.scen.keys())
keys = pick(c[0], "비교할 시나리오", opts, sc_label, default=["무대응", "S1", "S2b", "S3", "S4b"], multi=True,
            help="무대응은 자체 역량도 오픈웨이트 배치도 없는 가상의 비교 기준(예타의 사업 미시행)")
base = c[1].selectbox("기준", keys, index=keys.index("S1") if "S1" in keys else 0) if keys else None
N = c[2].selectbox("거시 반복", [2000, 5000, 20000], index=1)
pacing, reg = regime_picker(c[3], "mac_regime")

if keys and base:
    with st.spinner("시나리오 통계와 거시 경로 계산 중"):
        stats = {k: scen_stats(k, N=1000, pacing=pacing) for k in keys}
        out = cached_macro(to_json(stats), to_json(ss.MAC), N, 31)
    from core.macro import compare
    cmp = compare(out, base)
    st.subheader(f"{base} 대비 주권 배당")
    st.caption("주권 배당 = 국내 귀속 부가가치 증가 + 회피한 차단 손실(2026～2035 현재가치, 같은 난수 표본의 경로별 차이). 괄호는 10～90% 범위.")
    k1 = [r for r in cmp if r["시나리오"] != base]
    if k1:
        cols = st.columns(min(4, len(k1)))
        for col, r in zip(cols * 3, k1):
            col.metric(f"{r['시나리오']} 주권 배당 평균", f"{r['주권 배당 합계 평균(조원)']:+.1f}조원",
                       help=f"국내 귀속 부가가치 증가 중앙 {r['국내 귀속 부가가치 증가(조원)'][1]:+.1f}조원, 95% 꼬리 차단 손실 감소 {r['차단 손실 95% 꼬리 감소(조원)']:+.1f}조원")
    f = fig_base(title=f"주권 배당 구성({base} 대비, 평균, 조원)", h=360)
    names = [r["시나리오"] for r in k1]
    ret_mean = [float((out[n]["_retained"] - out[base]["_retained"]).mean()) for n in names]
    blk_mean = [float((out[base]["_loss"] - out[n]["_loss"]).mean()) for n in names]
    f.add_trace(go.Bar(x=names, y=ret_mean, name="국내 귀속 부가가치 증가", marker_color="#2a78d6"))
    f.add_trace(go.Bar(x=names, y=blk_mean, name="회피한 차단 손실", marker_color="#9ec5f0"))
    f.update_layout(barmode="relative", yaxis_title="조원(현재가치)"); f.add_hline(y=0, line=dict(color=INK, width=1))
    st.plotly_chart(f)
    tab = []
    for r in cmp:
        n = r["시나리오"]; o = out[n]; s = stats[n]
        tab.append({"시나리오": n, "성공 확률": s["p_success"], "주권 커버리지": s["coverage"],
                    "AI 효과(GDP) 중앙": o["gdp_gain"][1], "국내 귀속 부가가치 중앙": o["retained"][1], "서비스 수입 중앙": o["imports"][1],
                    "차단 손실 평균": o["block_loss_mean"], "차단 손실 95%": o["block_loss_p95"],
                    "부가가치 증가 중앙(기준 대비)": r["국내 귀속 부가가치 증가(조원)"][1], "회피 차단 손실 중앙": r["회피한 차단 손실(조원)"][1],
                    "주권 배당 평균": r["주권 배당 합계 평균(조원)"], "꼬리 위험 감소(95%)": r["차단 손실 95% 꼬리 감소(조원)"]})
    df = pd.DataFrame(tab)
    st.dataframe(df.round(2), hide_index=True)
    st.caption("금액은 조원, 2026～2035년 현재가치. AI 효과(GDP)는 공급원과 무관하므로 시나리오 간에 같습니다. 이 값이 같다는 점이 핵심입니다: 추진 방식의 차이는 효과의 크기가 아니라 귀속과 위험에서 나옵니다.")

    c1, c2 = st.columns(2)
    f = fig_base(title="국내 공급 비중(중앙값)", ytitle="비중", h=340)
    for n in keys:
        f.add_trace(go.Scatter(x=HORIZON, y=out[n]["path_dshare"], mode="lines+markers", name=n, line=dict(color=SC_COLOR.get(n, INK), width=2, dash="dot" if n == "무대응" else "solid"), marker=dict(size=6)))
    c1.plotly_chart(f)
    f = fig_base(title="AI의 측정 GDP 효과와 국내 귀속분(중앙값, 조원/년)", ytitle="조원", h=340)
    f.add_trace(go.Scatter(x=HORIZON, y=out[keys[0]]["path_gain"], name="AI 효과(모든 시나리오 공통)", line=dict(color="#0b0b0b", width=2)))
    for n in keys:
        f.add_trace(go.Scatter(x=HORIZON, y=out[n]["path_retained"], name=f"{n} 국내 귀속", line=dict(color=SC_COLOR.get(n, INK), width=2, dash="dot" if n == "무대응" else "solid")))
    c2.plotly_chart(f)
    st.caption("J-곡선: 도입이 빠른 시기에는 보완 무형투자가 측정 성과를 일부 가립니다(Brynjolfsson, Rock and Syverson 2021). 국내 귀속분은 AI 서비스 지출 중 국내 공급분과 해외 공급분의 국내 부가가치 비율로 계산합니다.")

    with st.expander("해석 원칙과 한계"):
        st.markdown("""
- 거시 효과의 절대 수준은 한국은행 추정 범위에 전적으로 의존합니다. 정책 비교에는 **시나리오 간 차이**만 쓰는 것이 원칙입니다.
- 국내 귀속 부가가치 증가는 수입 대체에 따른 GNI 관점의 이득이며, 투입 측 산업연관 효과(건설·장비)와 더하면 이중 계산입니다.
- 차단 손실은 프런티어급 모델이 필요한 효과 비중만큼 발생하며 주권 커버리지가 있으면 70% 완화된다고 가정합니다. 커버리지가 차단 시 실제로 작동하는지는 검증 인프라와 운영 준비에 달려 있습니다.
- 한국 모델이 성공해도 프런티어 품질에 못 미치면 국산 전환은 문턱 위 지출 일부에 그칩니다(전환율 20～40%). 전환율은 결론을 가장 크게 좌우하는 모수이므로 전략 분석 화면의 민감도와 함께 보세요.
""")
    if st.button("거시 결과 저장"):
        save_run("거시", f"기준 {base}, {reg}, {N}회", df.to_dict("records")); st.success("저장했습니다.")
else:
    st.info("비교할 시나리오와 기준을 고르세요.")
