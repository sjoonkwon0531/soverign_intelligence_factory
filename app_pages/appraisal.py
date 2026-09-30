import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import run_sc, scen_stats, cached_macro, cached_cb, to_json, data_badge, fig_base, heatmap, save_run, SC_COLOR, INK
from core.appraisal import pv, ahp
from core.costbenefit import default_cb
from core.macro import MACRO_PARAMS
from core.params import LAG_DIVISOR

ss = st.session_state
st.title("예비타당성조사식 종합 평가")
st.write("예비타당성조사의 틀(사회적 할인율로 현재가치화한 편익비용비, 경제성·정책성·기술성의 다기준 종합)에 맞춰 대안을 평가합니다. "
         "편익은 두 가지 방식으로 따로 계산해 서로 견줍니다. 거시 방식은 국내 귀속 부가가치와 회피한 차단 손실, 미시 방식은 비용편익 모듈의 회피한 종속비용입니다. "
         "두 방식은 같은 편익을 다른 경로로 잰 것이므로 더하지 않습니다.")
data_badge()
if "MAC" not in ss: ss.MAC = dict(MACRO_PARAMS)

c = st.columns(4)
own = [k for k in ss.scen]
alts = c[0].multiselect("평가 대안", own, default=[k for k in ["S1", "S2b", "S3", "S4b"] if k in own])
comp = c[1].selectbox("비교 기준", ["무대응"] + own, index=0, help="무대응: 사업 미시행(비용 0, 자체 역량과 배치 트랙 없음). 특정 시나리오를 고르면 증분 분석")
r_s = c[2].number_input("사회적 할인율", 0.0, 0.1, 0.045, 0.005, format="%.3f")
Nm = c[3].selectbox("거시 반복", [2000, 5000], index=0)
c = st.columns(4)
ext = c[0].number_input("2032～2035 연 운영 순비용(조원)", 0.0, 5.0, 0.75, 0.05, help="비용편익 모듈의 가정 0.3～1.2조원의 중앙. 자체 모델이 있는 시나리오에만 적용")
ext_open = c[1].number_input("오픈웨이트 전용 시나리오의 연 운영비(조원)", 0.0, 3.0, 0.3, 0.05)
gov = c[2].number_input("재정 조달 비중", 0.0, 1.0, 0.30, 0.05, help="SPC 정부 출자 30% 가정. 분산 지속은 전액 재정이므로 1.0으로 바꿔 비교할 수 있음")
mcpf = c[3].number_input("공공자금의 한계비용(MCPF)", 1.0, 2.0, 1.0, 0.05, help="재정 조달 1원의 사회적 비용(Dahlby 2008). 국내 예타는 기본 적용하지 않으므로 1.0이 기본, 민감도 확인용")
fiscal_full = st.toggle("분산 지속(S1, S1B)은 전액 재정으로 계산", value=True)

if not alts:
    st.info("평가할 대안을 고르세요."); st.stop()

keys = list(dict.fromkeys(["무대응", comp] + alts))
with st.spinner("시나리오와 거시 경로 계산 중"):
    runs = {k: run_sc(k, N=1000) for k in keys if k != "무대응"}
    stats = {k: scen_stats(k, N=1000) for k in keys}
    mac = cached_macro(to_json(stats), to_json(dict(ss.MAC, discount=r_s)), Nm, 31)


def cost_pv(k):
    if k == "무대응": return 0.0
    r = runs[k]; sc = ss.scen[k]
    capex = np.median(r["paths"]["capex"], 0); opex = np.median(r["paths"]["opex"], 0)
    base = pv(capex + opex, r_s)
    e = ext if sc["own_frontier"] else ext_open
    base += sum(e * 1e12 / (1 + r_s) ** (6 + j) for j in range(4))
    g = 1.0 if (fiscal_full and k in ("S1", "S1B")) else gov
    return base * (1 + (mcpf - 1) * g) / 1e12


rows = []
b0 = mac[comp]
for k in alts:
    if k == comp: continue
    dC = cost_pv(k) - cost_pv(comp)
    dret = float((mac[k]["_retained"] - b0["_retained"]).mean()); dblk = float((b0["_loss"] - mac[k]["_loss"]).mean())
    dB = dret + dblk
    rows.append({"대안": k, "비용 증분(조원)": dC, "국내 귀속 부가가치 증가": dret, "회피한 차단 손실": dblk, "편익 증분(조원)": dB,
                 "B/C": dB / dC if dC > 0 else np.nan, "순현재가치(조원)": dB - dC,
                 "판정": ("비용 절감과 편익 증가(우월)" if dC <= 0 and dB >= 0 else ("열위" if dC >= 0 and dB <= 0 else ("B/C 1 이상" if dB / dC >= 1 else "B/C 1 미만")) if dC != 0 else "비용 동일")})
df = pd.DataFrame(rows)

t1, t4, t2, t3 = st.tabs(["경제성(거시 편익)", "트랙 A의 증분 가치", "경제성 교차 확인(미시 편익)", "종합 평가(AHP)"])
with t1:
    st.caption(f"비교 기준: {comp}. 비용은 2026～2031 설비투자와 운영비의 경로 중앙값에 2032～2035 운영비를 더한 현재가치, 편익은 2026～2035 현재가치(할인율 {r_s:.1%}). "
               "B2G 매출은 정부와 SPC 사이의 이전이므로 편익에서 제외, 산업연관 파급효과는 편익에 더하지 않음.")
    if len(df):
        cols = st.columns(min(4, len(df)))
        for col, (_, r) in zip(cols * 3, df.iterrows()):
            col.metric(f"{r['대안']} B/C", "정의 불가" if not np.isfinite(r["B/C"]) else f"{r['B/C']:.2f}", help=f"순현재가치 {r['순현재가치(조원)']:+.1f}조원, {r['판정']}")
        st.dataframe(df.round(2), hide_index=True)
        f = fig_base(title="비용 증분과 편익 증분(조원, 현재가치)", xtitle="비용 증분", ytitle="편익 증분", h=420)
        mx = max(1.0, float(np.nanmax(np.abs(df[["비용 증분(조원)", "편익 증분(조원)"]].values))) * 1.15)
        f.add_trace(go.Scatter(x=[-mx, mx], y=[-mx, mx], mode="lines", line=dict(color=INK, dash="dot", width=1), name="B/C = 1", hoverinfo="skip"))
        for _, r in df.iterrows():
            f.add_trace(go.Scatter(x=[r["비용 증분(조원)"]], y=[r["편익 증분(조원)"]], mode="markers+text", text=[r["대안"]], textposition="top center",
                                   marker=dict(size=14, color=SC_COLOR.get(r["대안"], "#2a78d6"), line=dict(color="white", width=2)), name=r["대안"]))
        f.update_layout(hovermode="closest", showlegend=False); f.add_hline(y=0, line=dict(color="#c9c8c4", width=1)); f.add_vline(x=0, line=dict(color="#c9c8c4", width=1))
        st.plotly_chart(f)
        st.caption("점선 위쪽은 편익 증분이 비용 증분보다 큰 영역. 왼쪽 위는 비용을 줄이면서 편익이 늘어나는 우월 대안.")

        st.subheader("편익비용비 민감도")
        tgt = st.selectbox("대상 대안", df["대안"].tolist(), index=df["대안"].tolist().index("S4b") if "S4b" in df["대안"].tolist() else 0)
        caps = [0.1, 0.2, 0.3, 0.4, 0.6]; G = {"하단(4.2～6%)": (0.042, 0.06), "중앙(6～9%)": (0.06, 0.09), "상단(9～12.6%)": (0.09, 0.126)}
        if st.button("전환율 × AI 효과 크기 격자 계산", help="격자점마다 거시 모형을 다시 돌립니다(약 15회)"):
            Z = []
            dC = cost_pv(tgt) - cost_pv(comp)
            for gl, gv in G.items():
                row = []
                for cp in caps:
                    M2 = dict(ss.MAC, discount=r_s, capture=(cp, cp), G_LR=gv)
                    o = cached_macro(to_json({comp: stats[comp], tgt: stats[tgt]}), to_json(M2), 2000, 31)
                    dB = float((o[tgt]["_retained"] - o[comp]["_retained"]).mean() + (o[comp]["_loss"] - o[tgt]["_loss"]).mean())
                    row.append(dB / dC if dC > 0 else np.nan)
                Z.append(row)
            st.plotly_chart(heatmap(Z, [f"{int(c*100)}%" for c in caps], list(G), fmt="{:.2f}", title=f"{tgt}의 B/C({comp} 대비)", xtitle="성공 시 국산 전환율", ytitle="AI의 장기 GDP 효과", h=340))
            st.caption("B/C가 1을 넘는 영역이 정책의 성립 조건입니다. 전환율은 조달 설계(품질 연동, 보호 물량)와 검증 인프라로 정부가 움직일 수 있는 변수입니다.")

with t2:
    st.caption("비용편익 모듈(할인율 5%, 2031년 지평)의 편익비용비: (성공 확률 × 성공 시 회피 종속비용 + 국가 잔존가치) / 누적 재무 갭. 거시 방식과 편익의 경로가 다르므로 둘의 방향이 같은지만 확인합니다.")
    rows2 = []
    for k in alts:
        if k == "무대응" or not ss.scen[k]["own_frontier"]: continue
        r = runs[k]; C = default_cb(); C["krw_per_usd"] = ss.P["krw_per_usd"]
        cb = cached_cb(tuple(v * 1e12 for v in r["gap"]), r["G2029_dom"], tuple(float(v) for v in ss.scen[k]["capex_krw"][:4]), to_json(C), 10000, 20260922)
        av = cb["spc"]["avoided_if_success"][1] if cb["spc"]["avoided_if_success"] else 0.0
        B = r["G2029_dom"] * av + cb["residual"]["total_nation"][1]; Cc = r["gap"][1]
        rows2.append({"대안": k, "성공 확률": r["G2029_dom"], "성공 시 회피 종속비용 중앙": av, "국가 잔존가치 중앙": cb["residual"]["total_nation"][1],
                      "누적 재무 갭 중앙": Cc, "B/C(미시)": B / Cc if Cc > 0 else np.nan, "기대 순편익 2031": cb["spc"]["net_benefit"][1], "순편익 양일 확률": cb["spc"]["P_nb_pos"]})
    if rows2:
        st.dataframe(pd.DataFrame(rows2).round(3), hide_index=True)
        st.caption("미시 방식은 2031년까지의 좁은 지평과 문턱 위 지출만을 편익으로 보므로 거시 방식보다 보수적입니다. 두 방식의 부호가 다르면 결론은 지평과 전환율 가정에 달려 있다는 뜻입니다.")

with t3:
    st.caption("예타의 AHP는 사업 시행과 미시행을 쌍대 비교하지만, 여기서는 대안 간 우선순위를 같은 틀로 단순화해 계산합니다. 가중치와 판정 기준은 사용자가 정합니다(예타 지침의 가중치 범위는 사업 유형과 지역에 따라 다름).")
    c = st.columns(3)
    we = c[0].slider("경제성 가중치", 0.0, 1.0, 0.5, 0.05); wp = c[1].slider("정책성 가중치", 0.0, 1.0, 0.3, 0.05); wt = c[2].slider("기술성 가중치", 0.0, 1.0, 0.2, 0.05)
    cand = [k for k in alts]
    ref = mac["무대응"]
    sc = {}
    for k in cand:
        dC = cost_pv(k); dB = float((mac[k]["_retained"] - ref["_retained"]).mean() + (ref["_loss"] - mac[k]["_loss"]).mean())
        tail = float(np.percentile(ref["_loss"], 95) - np.percentile(mac[k]["_loss"], 95))
        r = runs[k]
        lag = r["lag_front"][1] if r["lag_front"] else None
        sc[k] = dict(bc=dB / dC if dC > 0 else 0.0, cov=stats[k]["coverage"], tail=tail, gate=stats[k]["p_success"], lag=lag)
    lags = [v["lag"] for v in sc.values() if v["lag"] is not None] or [1.0]; lmax = max(lags) + 0.5
    for v in sc.values():
        if v["lag"] is None: v["lag"] = lmax
    tails = [v["tail"] for v in sc.values()]; tmax = max(max(tails), 1e-9)
    scores = {k: {"경제성": v["bc"], "정책성": 0.5 * v["cov"] + 0.5 * max(v["tail"], 0) / tmax, "기술성": 0.5 * v["gate"] + 0.5 * (lmax - v["lag"]) / lmax} for k, v in sc.items()}
    res = ahp(scores, {"경제성": we, "정책성": wp, "기술성": wt})
    tab = pd.DataFrame([{"대안": k, "B/C(무대응 대비)": sc[k]["bc"], "주권 커버리지": sc[k]["cov"], "차단 꼬리 위험 감소(조원)": sc[k]["tail"],
                         "도메인 게이트 확률": sc[k]["gate"], "2029 프런티어 격차(년)": sc[k]["lag"],
                         "경제성 점수": res["local"]["경제성"][k], "정책성 점수": res["local"]["정책성"][k], "기술성 점수": res["local"]["기술성"][k], "종합": res["total"][k]} for k in cand])
    tab = tab.sort_values("종합", ascending=False)
    st.dataframe(tab.round(3), hide_index=True)
    f = fig_base(title="종합 우선순위(가중합, 합계 1)", h=320)
    for crit, col in (("경제성", "#2a78d6"), ("정책성", "#9ec5f0"), ("기술성", "#0b4a94")):
        f.add_trace(go.Bar(y=tab["대안"], x=[res["weights"][crit] * res["local"][crit][k] for k in tab["대안"]], name=crit, orientation="h", marker_color=col))
    f.update_layout(barmode="stack", yaxis=dict(autorange="reversed"), hovermode="closest")
    st.plotly_chart(f)
    st.caption("정책성 = 주권 커버리지와 차단 꼬리 위험 감소의 평균, 기술성 = 도메인 게이트 확률과 격차 역수 척도의 평균(자체 모델이 없는 대안은 격차 척도 0점: 기술성은 국내 자체 역량을 평가). 점수는 기준별로 대안 합이 1이 되도록 정규화.")
    if st.button("종합 평가 저장"):
        save_run("종합 평가", f"기준 {comp}, 가중치 {we}/{wp}/{wt}", dict(econ=df.to_dict("records"), ahp=tab.to_dict("records"))); st.success("저장했습니다.")

with t4:
    st.caption("이중 트랙 권고안(S4b)에서 자체 추격(트랙 A)을 뺀 것이 오픈웨이트 배치와 검증(S3)입니다. 둘의 차이가 트랙 A의 증분 비용과 증분 편익입니다. "
               "트랙 A의 편익은 성공 확률과 성공 시 국산 전환율의 곱에 비례하므로, 두 변수의 격자에서 증분 B/C가 1을 넘는 조건을 찾습니다.")
    if "S4b" in ss.scen and "S3" in ss.scen:
        dCa = cost_pv("S4b") - cost_pv("S3")
        c = st.columns(3)
        c[0].metric("트랙 A 증분 비용(조원, 현재가치)", f"{dCa:.1f}")
        c[1].metric("현재 모수의 성공 확률", f"{stats['S4b']['p_success']:.2f}" if "S4b" in stats else "해당 없음")
        if st.button("성공 확률 × 전환율 격차 계산", key="ta_btn"):
            st_S3 = scen_stats("S3", N=1000); st_A = scen_stats("S4b", N=1000)
            ps_l = [round(st_A["p_success"], 2), 0.4, 0.5, 0.6, 0.7]; cp_l = [0.2, 0.3, 0.4, 0.6, 0.8]
            Z = []
            for ps in ps_l:
                row = []
                for cp in cp_l:
                    o = cached_macro(to_json({"S3": st_S3, "S4b": dict(st_A, p_success=ps)}), to_json(dict(ss.MAC, discount=r_s, capture=(cp, cp))), 3000, 31)
                    dB = float((o["S4b"]["_retained"] - o["S3"]["_retained"]).mean() + (o["S3"]["_loss"] - o["S4b"]["_loss"]).mean())
                    row.append(dB / dCa if dCa > 0 else np.nan)
                Z.append(row)
            ss._ta = (Z, ps_l, cp_l)
        if "_ta" in ss:
            Z, ps_l, cp_l = ss._ta
            st.plotly_chart(heatmap(Z, [f"{int(c*100)}%" for c in cp_l], [f"{p:.2f}" for p in ps_l], fmt="{:.2f}", title="트랙 A의 증분 B/C(S3 대비)",
                                    xtitle="성공 시 국산 전환율", ytitle="2029 도메인 게이트 확률", h=380))
            st.markdown("**읽는 법**: 오픈웨이트 배치와 검증 인프라(트랙 B)는 단독으로도 편익비용비가 높은 무후회 투자입니다. 트랙 A는 성공 확률과 전환율이 함께 높아야 증분 B/C가 1을 넘으므로, "
                        "정책의 초점은 트랙 A의 규모가 아니라 (1) 전환율을 올리는 조달·검증 설계, (2) 성공 확률이 낮게 관측될 때 트랙 A를 줄이는 중단·재편 규칙에 있어야 합니다. "
                        "이 계산은 학습 파급, 안보, 수출 같은 비시장 가치를 포함하지 않으므로 그 가치가 B/C 1까지의 부족분을 메우는지가 정책 판단의 쟁점입니다.")
