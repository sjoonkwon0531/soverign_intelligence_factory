import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import fig_base, save_run
from core.gva_b2g import B2G_ITEMS, b2g

ss = st.session_state
st.title("공공수요(B2G) 상향식 적산")
st.write("예산서와 조달 통계의 항목별 금액에 AI 대체율(저·중·고)을 곱하고 성장률을 적용해 연도별 공공 AI 조달 가능 규모를 적산합니다. 항목, 금액, 대체율을 직접 고칠 수 있습니다.")
if "b2g_items" not in ss: ss.b2g_items = B2G_ITEMS.copy()
items = st.data_editor(ss.b2g_items, num_rows="dynamic", key="b2g_ed",
                       column_config={"문턱 위": st.column_config.CheckboxColumn("문턱 위(국방·의료 등)"), "군": st.column_config.SelectboxColumn("군", options=["A", "B"])})
c = st.columns(5)
gA = c[0].number_input("A군 성장률(기존 IT 지출)", 0.0, 0.5, 0.04, 0.01)
gB = c[1].number_input("B군 성장률(신규 AI 예산)", 0.0, 1.0, 0.30, 0.05)
s_lo = c[2].number_input("SPC 점유 저", 0.0, 1.0, 0.40, 0.05); s_mi = c[3].number_input("SPC 점유 중", 0.0, 1.0, 0.55, 0.05); s_hi = c[4].number_input("SPC 점유 고", 0.0, 1.0, 0.70, 0.05)
cc = st.columns(2)
if cc[0].button("표 적용"): ss.b2g_items = items; st.success("적용했습니다.")
if cc[1].button("기본 항목으로"): ss.b2g_items = B2G_ITEMS.copy(); st.rerun()
t = b2g(items.dropna(subset=["기준금액(억원)"]), gA, gB, spc_share=(s_lo, s_mi, s_hi))
st.subheader("연도별 조달 가능 규모(조원)")
st.dataframe(t.round(2), hide_index=True)
tot = t.drop(columns="연도").sum()
c = st.columns(3)
c[0].metric("5개년 누계(중)", f"{tot['조달 가능 중']:.1f}조원", help=f"저 {tot['조달 가능 저']:.1f}, 고 {tot['조달 가능 고']:.1f}")
c[1].metric("문턱 위 누계(중)", f"{tot['문턱 위 중']:.1f}조원", help=f"저 {tot['문턱 위 저']:.1f}, 고 {tot['문턱 위 고']:.1f}")
c[2].metric("SPC 귀속 누계(중)", f"{tot['SPC 귀속 중']:.1f}조원", help=f"저 {tot['SPC 귀속 저']:.1f}, 고 {tot['SPC 귀속 고']:.1f}")
f = fig_base(title="조달 가능 규모(중): 문턱 위와 문턱 아래(조원)", h=360)
f.add_trace(go.Bar(x=t["연도"], y=t["조달 가능 중"] - t["문턱 위 중"], name="문턱 아래", marker_color="#9ec5f0"))
f.add_trace(go.Bar(x=t["연도"], y=t["문턱 위 중"], name="문턱 위", marker_color="#0b4a94"))
f.update_layout(barmode="stack"); st.plotly_chart(f)
st.caption("총량을 인용할 때는 프런티어급이 필요한 문턱 위 규모를 함께 적어야 함. 조달 구조 전환 3종(신규 정보화의 AI 내재화 의무, 다년도 계약, 국방 AI SW 항목)이 실행되어야 대체율이 저에서 중으로 이동")
if st.button("결과 저장"):
    save_run("공공수요", f"A {gA}, B {gB}", t.to_dict("records")); st.success("저장했습니다.")
