import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import pick, sc_label, run_sc, cached_cb, to_json, data_badge, fig_base, heatmap, save_run, SC_COLOR, INK, rng3
from core.costbenefit import default_cb, CB_LABELS, breakeven_capture, fmt_breakeven, tornado
from core.econ import spc_financials, cost_effectiveness
from core.gva_b2g import run_gva, ALLOC, GVA_PARAMS

ss = st.session_state
st.title("비용편익과 ROI")
st.write("국가 관점(실패 시 잔존가치, 임차 경로 종속비용, 순편익, 손익분기)과 SPC 재무 관점(NPV, IRR, 회수기간, 편익비용비, ROI), 국내 부가가치를 계산합니다. 누적 재무 갭과 성공 확률은 시나리오 시뮬레이션에서 자동으로 가져옵니다.")
data_badge()
if "CB" not in ss: ss.CB = default_cb()

own = [k for k in ss.scen if ss.scen[k]["own_frontier"]]
c = st.columns(3)
k = pick(c[0], "평가할 SPC 시나리오", own, sc_label, default="S4b")
N = c[1].selectbox("시나리오 반복", [1000, 3000], index=1)
Ncb = c[2].selectbox("비용편익 반복", [5000, 20000], index=1)
r = run_sc(k, N=N)
gap_tri = tuple(v * 1e12 for v in r["gap"]); p_s = r["G2029_dom"]; capex4 = [float(v) for v in ss.scen[k]["capex_krw"][:4]]
with st.expander("연계 값과 비용편익 모수", expanded=False):
    c = st.columns(3)
    g1 = c[0].number_input("누적 재무 갭 10%(조원)", value=round(r["gap"][0], 2)); g2 = c[1].number_input("중앙(조원)", value=round(r["gap"][1], 2)); g3 = c[2].number_input("90%(조원)", value=round(r["gap"][2], 2))
    p_s = st.number_input("성공 확률(2029 도메인 게이트)", 0.0, 1.0, float(round(p_s, 3)), 0.01)
    gap_tri = tuple(sorted([g1 * 1e12, g2 * 1e12, g3 * 1e12]))
    rows = []
    for kk, lab in CB_LABELS.items():
        v = ss.CB[kk]; lo, hi = (v if isinstance(v, (tuple, list)) else (v, v))
        rows.append({"키": kk, "모수": lab, "하한": float(lo), "상한": float(hi)})
    ed = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=["키", "모수"], key="cb_editor")
    cc = st.columns(2)
    if cc[0].button("비용편익 모수 적용"):
        C = dict(ss.CB)
        for _, row in ed.iterrows():
            lo, hi = sorted([float(row["하한"]), float(row["상한"])])
            C[row["키"]] = lo if not isinstance(ss.CB[row["키"]], (tuple, list)) else (lo, hi)
        ss.CB = C; st.success("적용했습니다.")
    if cc[1].button("기본 모수로"):
        ss.CB = default_cb(); st.rerun()
ss.CB["krw_per_usd"] = ss.P["krw_per_usd"]
cb = cached_cb(gap_tri, p_s, tuple(capex4), to_json(ss.CB), Ncb, 20260922)

t1, t2, t3, t4, t5 = st.tabs(["국가 관점 순편익", "SPC 재무 지표", "민감도", "국내 부가가치", "비용 효과성"])
with t1:
    s = cb["spc"]
    c = st.columns(4)
    c[0].metric("실패 시 잔존가치(국가)", f"{cb['residual']['total_nation'][1]:.2f}조원", help=rng3(cb["residual"]["total_nation"]))
    c[1].metric("SPC 순비용(갭−잔존)", f"{s['net_cost'][1]:.2f}조원", help=rng3(s["net_cost"]))
    c[2].metric("2031 기대 순편익", f"{s['net_benefit'][1]:+.2f}조원", help=f"양(+)일 확률 {s['P_nb_pos']:.2f}")
    c[3].metric("2035 기대 순편익", f"{cb['h2035']['net_benefit'][1]:+.2f}조원", help=f"양(+)일 확률 {cb['h2035']['P_nb_pos']:.2f}")
    be31 = breakeven_capture(cb["grid2031"], round(p_s, 2) if round(p_s, 2) in (0.2, 0.31, 0.4, 0.5, 0.6, 0.7) else p_s)
    st.markdown(f"**손익분기 국산 전환율**(성공 확률 격자에서 가장 가까운 값 기준): 2031년 지평 {fmt_breakeven(be31)}, 2035년 지평 {fmt_breakeven(breakeven_capture(cb['grid2035'], p_s))}")
    rows = [["잔존가치: 장비 회수가", *cb["residual"]["equipment_salvage"]], ["잔존가치: 데이터센터·전력", *cb["residual"]["dc_power"]],
            ["잔존가치: 역량(국가)", *cb["residual"]["capability_nation"]], ["잔존가치 합계(국가)", *cb["residual"]["total_nation"]],
            ["잔존가치 합계(SPC 주주)", *cb["residual"]["total_spc"]], ["임차 경로 총 유출(참고, 비용 아님)", *cb["lockin"]["gross_outflow"]],
            ["지대 이전", *cb["lockin"]["rent"]], ["접근 차단 기대손실", *cb["lockin"]["block_loss"]], ["강제 전환비용", *cb["lockin"]["switch"]],
            ["실질 종속비용 합계", *cb["lockin"]["total"]], ["SPC 순비용", *s["net_cost"]], ["순편익 2031", *s["net_benefit"]],
            ["순편익 2031, 성공 시", *(s["nb_success"] or [np.nan] * 3)], ["순편익 2031, 실패 시", *(s["nb_fail"] or [np.nan] * 3)],
            ["순편익 2035", *cb["h2035"]["net_benefit"]], ["순편익 2035, 성공 시", *(cb["h2035"]["nb_success"] or [np.nan] * 3)]]
    st.dataframe(pd.DataFrame(rows, columns=["항목(조원, 현재가치)", "10%", "중앙값", "90%"]).round(2), hide_index=True)
    st.caption(f"2031년까지 접근 차단 발생 확률 {cb['lockin']['P_block_2031']:.2f}, 2035년까지 {cb['h2035']['P_block_2035']:.2f}. 안보·정보 주권 등 비시장 가치는 계상하지 않음.")
    c1, c2 = st.columns(2)
    for col, key, ttl, capk in ((c1, "grid2031", "2031년 지평", (0.1, 0.2, 0.3, 0.4, 0.5)), (c2, "grid2035", "2035년 지평", (0.2, 0.3, 0.4, 0.5))):
        g = pd.DataFrame(cb[key]).pivot(index="P_success", columns="capture", values="E_nb")
        col.plotly_chart(heatmap(g.values, [f"{int(v*100)}%" for v in g.columns], [f"{v:.2f}" for v in g.index], fmt="{:+.2f}", diverging=True,
                                 title=f"기대 순편익(조원), {ttl}", xtitle="국산 전환율(성공 시)", ytitle="성공 확률", h=380))
    st.markdown("**미국 접근 유지 확률 구간별 결과**: 차단 위험이 커져도 순편익이 양일 확률은 거의 변하지 않으면, 결정 변수는 차단 확률이 아니라 성공 확률과 전환율임")
    st.dataframe(pd.DataFrame(cb["pi_sensitivity"]).round(2), hide_index=True)
    if st.button("국가 관점 결과 저장"):
        save_run("비용편익", f"{k}, 성공 확률 {p_s:.2f}", dict(rows=rows, breakeven=[fmt_breakeven(be31)])); st.success("저장했습니다.")

with t2:
    st.caption("SPC를 하나의 사업으로 보고 연도별 매출에서 설비투자와 운영비를 뺀 현금흐름으로 계산합니다. 2031년 이후 가치는 아래 방식 중 하나로 반영합니다.")
    c = st.columns(4)
    rate = c[0].number_input("할인율", 0.0, 0.3, 0.05, 0.01)
    term = c[1].selectbox("종료가치", ["잔존가치(누적 설비투자의 일정 비율)", "2032～2035 운영 지속"])
    rf = c[2].number_input("잔존 비율", 0.0, 1.0, 0.30, 0.05)
    eg = c[3].number_input("운영 지속 시 연 성장률", -0.5, 1.0, 0.10, 0.05)
    fin, sm = spc_financials(r["paths"], r=rate, terminal="residual" if term.startswith("잔존") else "operate", residual_frac=rf, ext_growth=eg)
    c = st.columns(5)
    c[0].metric("NPV 중앙값", f"{sm['npv'][1]:+.2f}조원", help=rng3(sm["npv"]))
    c[1].metric("NPV가 양일 확률", f"{sm['P_npv_pos']:.2f}")
    c[2].metric("IRR 중앙값", f"{sm['irr'][1]*100:.1f}%" if np.isfinite(sm["irr"][1]) else "정의 불가", help=f"IRR이 정의되는 경로 비율 {sm['irr_defined']:.2f}")
    c[3].metric("편익비용비 중앙값", f"{sm['bcr'][1]:.2f}")
    c[4].metric("기간 내 회수 확률", f"{sm['P_payback']:.2f}")
    f = fig_base(title="NPV 분포(조원)", h=320)
    s_ = r["paths"]["success"]
    f.add_trace(go.Histogram(x=fin["npv"][~s_] / 1e12, name="도메인 게이트 미달 경로", marker_color="#eb6834", opacity=0.75, nbinsx=40))
    f.add_trace(go.Histogram(x=fin["npv"][s_] / 1e12, name="도메인 게이트 통과 경로", marker_color="#2a78d6", opacity=0.75, nbinsx=40))
    f.update_layout(barmode="overlay", hovermode="closest"); f.add_vline(x=0, line=dict(color=INK, dash="dot"))
    st.plotly_chart(f)
    st.dataframe(pd.DataFrame([["NPV(조원)", *sm["npv"]], ["NPV, 통과 경로", *(sm["npv_success"] or [np.nan] * 3)], ["NPV, 미달 경로", *(sm["npv_fail"] or [np.nan] * 3)],
                               ["IRR", *sm["irr"]], ["편익비용비", *sm["bcr"]], ["ROI", *sm["roi"]]], columns=["지표", "10%", "중앙값", "90%"]).round(3), hide_index=True)
    st.info("SPC 재무 지표는 민간 출자자 관점의 참고값입니다. 국가 관점의 편익(종속비용 회피, 국내 부가가치)은 포함하지 않으므로, 정책 판단은 국가 관점 순편익과 함께 보아야 합니다.")

with t3:
    hz = st.radio("지평", ["2031", "2035"], horizontal=True)
    if st.button("민감도(토네이도) 계산", help="각 모수를 분포 하한·상한에 고정해 기대 순편익 변화를 계산(모수당 6,000회)"):
        with st.spinner("계산 중"):
            tr = tornado(gap_tri, p_s, capex4, ss.CB, N=6000, horizon=hz)
        base = tr[0]["base"]
        f = fig_base(title=f"기대 순편익 민감도({hz}년 지평, 기준 {base:+.2f}조원)", h=40 * len(tr) + 120)
        f.add_trace(go.Bar(y=[t["label"] for t in tr], x=[t["E_low"] - base for t in tr], base=base, orientation="h", name="하한", marker_color="#eb6834",
                           hovertemplate="%{y} 하한: %{x:+.2f}<extra></extra>"))
        f.add_trace(go.Bar(y=[t["label"] for t in tr], x=[t["E_high"] - base for t in tr], base=base, orientation="h", name="상한", marker_color="#2a78d6",
                           hovertemplate="%{y} 상한: %{x:+.2f}<extra></extra>"))
        f.update_layout(barmode="overlay", yaxis=dict(autorange="reversed"), xaxis_title="기대 순편익(조원)", hovermode="closest")
        f.add_vline(x=base, line=dict(color=INK, dash="dot"))
        st.plotly_chart(f)
        st.dataframe(pd.DataFrame(tr)[["label", "low_val", "high_val", "E_low", "E_high"]].rename(columns={"label": "모수", "low_val": "하한값", "high_val": "상한값", "E_low": "하한 시 순편익", "E_high": "상한 시 순편익"}).round(3),
                     hide_index=True)

with t4:
    st.caption("한국은행 2020년 기준년 산업연관표 SW산업 부가가치유발계수 0.893, 취업유발계수 10억원당 8.7명. 투입 측 국내 부가가치 비율은 가정")
    keys = st.multiselect("비교 시나리오", list(ss.scen.keys()), default=["S1", "S4b"])
    alloc = {}
    cols = st.columns(max(1, len(keys)))
    for col, kk in zip(cols, keys):
        a = ALLOC.get(kk, (0.40, 0.175, 0.425))
        col.markdown(f"**{kk} 배분**")
        w_eq = col.number_input("장비", 0.0, 1.0, a[0], 0.01, key=f"aeq{kk}"); w_dc = col.number_input("데이터센터", 0.0, 1.0, a[1], 0.01, key=f"adc{kk}")
        alloc[kk] = (w_eq, w_dc, max(0.0, 1 - w_eq - w_dc))
    Ng = st.selectbox("반복", [1000, 3000], index=0, key="gvaN")
    if st.button("부가가치 계산") and keys:
        with st.spinner("계산 중"):
            G = run_gva(keys, ss.scen, ss.P, alloc, GVA_PARAMS, N=Ng)
        rows = [{"시나리오": kk, "부가가치 합계(조원)": G[kk]["gva_total"][1], "10%": G[kk]["gva_total"][0], "90%": G[kk]["gva_total"][2],
                 "산출 측(조원)": G[kk]["gva_output"][1], "투입 측(조원)": G[kk]["gva_invest"][1], "취업유발(천 인년)": G[kk]["jobs"][1] / 1e3,
                 "성공 확률": G[kk]["P_success"]} for kk in keys]
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
        f = fig_base(title="국내 부가가치 구성(조원, 2026～2031 현재가치)", h=340)
        f.add_trace(go.Bar(x=keys, y=[G[kk]["gva_invest"][1] for kk in keys], name="투입 측", marker_color="#9ec5f0"))
        f.add_trace(go.Bar(x=keys, y=[G[kk]["gva_output"][1] for kk in keys], name="산출 측", marker_color="#2a78d6"))
        f.update_layout(barmode="stack"); st.plotly_chart(f)
        st.caption("부가가치 차이의 대부분은 조달 구조 전환(B2G 경로) 가정에서 나오므로, 결합 자체의 성과가 아니라 결합이 실행하는 전환의 크기로 해석해야 함")

with t5:
    keys = st.multiselect("시나리오", list(ss.scen.keys()), default=[x for x in ["S1", "S1B", "S2", "S2b", "S4", "S4b"] if x in ss.scen], key="ce_keys")
    base = st.selectbox("기준 시나리오", keys, index=0) if keys else None
    if keys and base:
        res = {kk: run_sc(kk, N=1000) for kk in keys}
        ce = pd.DataFrame(cost_effectiveness(res, base))
        st.dataframe(ce.round(3), hide_index=True)
        f = fig_base(title="총투입과 2029 도메인 게이트 확률", xtitle="총투입(조원)", ytitle="게이트 확률", h=380)
        for kk in keys:
            f.add_trace(go.Scatter(x=[res[kk]["capex_total"]], y=[res[kk]["G2029_dom"]], mode="markers+text", text=[kk], textposition="top center",
                                   marker=dict(size=14, color=SC_COLOR.get(kk, "#2a78d6"), line=dict(color="white", width=2)), name=kk))
        f.update_layout(hovermode="closest", showlegend=False); st.plotly_chart(f)
