import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import pick, sc_label, cached_sens, cached_discover, cached_options, to_json, data_badge, fig_base, save_run, SC_COLOR, INK

ss = st.session_state
st.title("전략 분석")
st.write("결과를 좌우하는 불확실성이 무엇인지, 어떤 세계에서 권고안이 다른 대안보다 나빠지는지, 중간 점검에서 멈출 권리가 얼마의 가치를 갖는지를 계산합니다. "
         "모두 같은 난수 표본에서 대안을 비교하므로 차이는 우연이 아니라 구조에서 나옵니다.")
data_badge()
own = [k for k in ss.scen if ss.scen[k]["own_frontier"]]
t1, t2, t3 = st.tabs(["무엇이 결과를 좌우하나(전역 민감도)", "권고안이 지는 세계(시나리오 발견)", "멈출 권리의 가치(실물옵션)"])

with t1:
    st.caption("모든 불확실 모수를 동시에 뽑은 몬테카를로 표본에서, 결과를 표준화한 모수에 회귀한 표준화 회귀계수(SRC)와 순위상관을 봅니다(Saltelli et al. 2008). "
               "설명력이 0.7 이상이면 SRC 순위를 믿을 수 있고, 낮으면 순위상관을 함께 봅니다.")
    c = st.columns(3)
    k = pick(c[0], "시나리오", own, sc_label, default="S4b", key="sens_k")
    N = c[1].selectbox("표본", [1000, 3000], index=0, key="sens_N")
    out_lab = {"success": "2029 도메인 게이트 통과", "lag_f": "2029 프런티어 격차(년)", "cum_gap": "누적 재무 갭(조원)"}
    o = pick(c[2], "결과 지표", list(out_lab), out_lab)
    with st.spinner("표본 계산 중"):
        S = cached_sens(to_json(ss.scen[k]), to_json(ss.P), N)
    sub = S[S["결과"] == o].copy()
    if len(sub):
        sub["abs"] = sub["표준화회귀계수"].abs(); sub = sub.sort_values("abs", ascending=False).head(12)
        r2 = sub["설명력"].iloc[0]
        f = fig_base(title=f"{out_lab[o]}에 대한 표준화 회귀계수(설명력 {r2:.2f})", h=40 * len(sub) + 120)
        f.add_trace(go.Bar(y=sub["모수"], x=sub["표준화회귀계수"], orientation="h", marker_color=["#2a78d6" if v > 0 else "#eb6834" for v in sub["표준화회귀계수"]],
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>", name="SRC"))
        f.update_layout(yaxis=dict(autorange="reversed"), hovermode="closest", showlegend=False, xaxis_title="표준화 회귀계수(파랑: 지표를 키움, 주황: 줄임)")
        f.add_vline(x=0, line=dict(color=INK, width=1))
        st.plotly_chart(f)
        st.dataframe(sub[["모수", "표준화회귀계수", "순위상관", "설명력"]].round(3), hide_index=True)
        st.markdown("**읽는 법**: 상위 모수는 정책으로 바꿀 수 있는 것(B2G 실현율, 흡수율 개선, 설비 규모)과 바꿀 수 없는 것(프런티어 속도, 연산 효율)으로 나뉩니다. "
                    "바꿀 수 없는 모수가 상위이면 조기 관측과 중단 규칙이, 바꿀 수 있는 모수가 상위이면 그 레버에 대한 집중이 전략의 핵심입니다.")
    else:
        st.info("이 시나리오에서는 해당 지표의 변동이 없습니다.")

with t2:
    st.caption("같은 표본에서 대안 A와 B의 성과를 경로별로 비교해, A가 B보다 나빠지는 모수 영역을 얕은 결정나무로 찾습니다(Lempert, Popper and Bankes 2003의 강건 의사결정 방식을 단순화). "
               "A가 모든 경로에서 B 이상이면 A가 B를 지배한다고 보고, 대신 A가 더 나은 영역을 설명합니다.")
    c = st.columns(4)
    a = c[0].selectbox("대안 A", own, index=own.index("S4b") if "S4b" in own else 0, key="dA")
    b = c[1].selectbox("대안 B", own, index=own.index("S2b") if "S2b" in own else 0, key="dB")
    met = pick(c[2], "지표", ["success", "lag_f", "cum_gap"], {"success": "도메인 게이트 통과", "lag_f": "프런티어 격차", "cum_gap": "누적 재무 갭"}, key="dM")
    dep = c[3].selectbox("나무 깊이", [2, 3], key="dD")
    if a == b:
        st.info("서로 다른 두 대안을 고르세요.")
    else:
        with st.spinner("같은 표본으로 두 대안 계산 중"):
            D = cached_discover(to_json(ss.scen[a]), to_json(ss.scen[b]), to_json(ss.P), 2000, met, dep)
        c = st.columns(4)
        c[0].metric("A가 나쁜 경로 비율", f"{D['share_worse']:.1%}"); c[1].metric("A가 나은 경로 비율", f"{D['share_better']:.1%}")
        c[2].metric("평균 차이(양수: A 유리)", f"{D['mean_diff']:+.3f}"); c[3].metric("평균 후회(A 선택 시)", f"{D['regret_mean']:.3f}")
        st.markdown(f"**설명 대상**: {D['explained']}")
        st.code(D["rules"], language=None)
        if D["importance"]:
            imp = pd.DataFrame(sorted(D["importance"].items(), key=lambda x: -x[1]), columns=["모수", "중요도"])
            st.dataframe(imp.round(3), hide_index=True)
        st.markdown("**활용**: 나무의 분기 모수는 조기 경보 지표입니다. 예컨대 연산 효율(연산량 10배당 역량지수)이 낮게 관측되면 학습 비중이 큰 대안이 유리해지는 식입니다. "
                    "이 모수들을 격차 진단 화면에서 새 자료로 다시 추정하면, 어느 세계에 있는지를 매년 확인할 수 있습니다.")

with t3:
    st.caption("해당 연도에 한국 모델이 오픈웨이트 최상위보다 정해진 여유 이상 뒤지면 트랙 A(자체 추격)를 멈추는 규칙을 넣고, 규칙이 없을 때와 같은 표본에서 비교합니다(Dixit and Pindyck 1994의 단계 투자 논리). "
               "오판 중단은 멈췄으나 계속했다면 도메인 게이트를 통과했을 경로, 적중 중단은 계속했어도 미달이었을 경로입니다.")
    c = st.columns(2)
    k = pick(c[0], "시나리오", own, sc_label, default="S4b", key="opt_k")
    N = c[1].selectbox("표본", [1000, 2000, 3000], index=1, key="opt_N")
    default_rules = pd.DataFrame([(2027, 3.0), (2027, 6.0), (2028, 0.0), (2028, 3.0), (2028, 6.0), (2029, 5.0)], columns=["점검 연도", "여유(역량지수)"])
    rules = st.data_editor(default_rules, num_rows="dynamic", hide_index=True, key="opt_rules")
    rl = [[int(r["점검 연도"]), float(r["여유(역량지수)"])] for _, r in rules.dropna().iterrows() if int(r["점검 연도"]) in (2026, 2027, 2028, 2029, 2030, 2031)]
    if st.button("중단 규칙 평가", type="primary") and rl:
        with st.spinner("규칙별 계산 중"):
            T = cached_options(to_json(ss.scen[k]), to_json(ss.P), json.dumps(rl), N)
        df = pd.DataFrame([{"규칙": f"{t['rule'][0]}년, 여유 {t['rule'][1]:g}", "중단 비율": t["stop_share"], "누적 갭 절감(조원)": t["gap_saved"],
                            "게이트 손실(%p)": t["gate_loss"] * 100, "적중 중단": t["true_stop"], "오판 중단": t["false_stop"],
                            "게이트 1%p당 절감(조원)": t["saved_per_gate_point"]} for t in T])
        ss._opt_df = df
    if "_opt_df" in ss:
        df = ss._opt_df
        st.dataframe(df.round(3), hide_index=True)
        f = fig_base(title="중단 규칙의 교환: 누적 갭 절감과 게이트 손실", xtitle="게이트 손실(%p)", ytitle="누적 갭 절감(조원)", h=380)
        f.add_trace(go.Scatter(x=df["게이트 손실(%p)"], y=df["누적 갭 절감(조원)"], mode="markers+text", text=df["규칙"], textposition="top center",
                               marker=dict(size=14, color="#2a78d6", line=dict(color="white", width=2)), name="규칙"))
        f.update_layout(hovermode="closest", showlegend=False)
        st.plotly_chart(f)
        st.markdown("**읽는 법**: 왼쪽 위(게이트 손실 없이 절감)가 가장 좋은 규칙입니다. 2029년 도메인 판정 직후의 중단은 정의상 게이트 손실이 없고, 이것이 에스크로와 재편 조항의 재정적 가치입니다. "
                    "더 이른 점검은 절감이 크지만 오판 중단이 늘어나므로, 점검 지표의 측정 오차(AA 환산 ±3.6점)보다 여유를 크게 잡아야 합니다. "
                    "게이트 1%p의 국가 가치는 비용편익 화면의 성공 시 순편익으로 환산할 수 있습니다.")
        if st.button("중단 규칙 결과 저장"):
            save_run("실물옵션", f"{k}, {N}회", df.to_dict("records")); st.success("저장했습니다.")
