import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import regime_picker, cached_grid_cell, to_json, data_badge, fig_base, heatmap, save_run, SC_COLOR
from core.params import SEED

ss = st.session_state
st.title("SPC 규모·민간 출자 비중 격자")
st.write("권고 구조(이중 트랙, 에스크로)를 유지한 채 총규모와 민간 출자 비중을 바꿔 성과와 비용을 비교합니다. 출자 여력, 흡수 역량, 계통 접속, GPU 공급 상한과 임대 방식을 반영합니다.")
data_badge()
c = st.columns(4)
sizes = c[0].multiselect("총규모(조원)", [5, 10, 16.7, 20, 30, 50, 100, 200], default=[5, 10, 16.7, 20, 30, 50])
shares = c[1].multiselect("민간 출자 비중", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], default=[0.5, 0.6, 0.7, 0.8])
N = c[2].selectbox("격자점당 반복", [200, 500, 1000, 2000], index=1, help="본보고서는 2,000회. 격자점 수 × 반복 횟수만큼 시간이 걸립니다.")
mode = c[3].selectbox("컴퓨팅 확보", ["구매", "임대"])
pacing, regime = regime_picker(st, "grid_regime")
if st.button("격자 실행", type="primary") or "grid_rows" in ss:
    if st.session_state.get("_grid_sig") != (tuple(sizes), tuple(shares), N, mode, regime, to_json(ss.P)):
        prog = st.progress(0.0); rows = []; tot = len(sizes) * len(shares); i = 0
        for T in sorted(sizes):
            for s in sorted(shares):
                rows.append(cached_grid_cell(T, s, to_json(ss.P), N, SEED, pacing, "lease" if mode == "임대" else "buy"))
                i += 1; prog.progress(i / tot, f"{T}조원, 민간 {int(s*100)}%")
        prog.empty(); ss.grid_rows = rows; ss._grid_sig = (tuple(sizes), tuple(shares), N, mode, regime, to_json(ss.P))
    df = pd.DataFrame(ss.grid_rows)
    metric = st.selectbox("지표", ["도메인 게이트 확률", "프런티어 격차 중앙(년)", "격차 1년 이하 확률", "누적 재무 갭(조원)", "2029 잔존가치(조원)",
                                 "2029 전력 구속 확률", "실현 자본(조원)", "정부 출자 실현(조원)", "민간 실현율", "공공성 이탈 위험지수"])
    piv = df.pivot(index="총규모(조원)", columns="민간 비중", values=metric).sort_index()
    fmt = "{:.2f}" if "확률" in metric or "년" in metric or "율" in metric or "지수" in metric else "{:.1f}"
    st.plotly_chart(heatmap(piv.values, [f"{int(s*100)}%" for s in piv.columns], [f"{v:g}" for v in piv.index], fmt=fmt,
                            xtitle="민간 출자 비중", ytitle="총규모(조원)", title=metric))
    sh = st.select_slider("규모별 추이를 볼 민간 비중", sorted(shares), value=0.7 if 0.7 in shares else sorted(shares)[0])
    sub = df[df["민간 비중"] == sh].sort_values("총규모(조원)")
    c1, c2 = st.columns(2)
    for col, (m, ttl) in zip((c1, c2), (("도메인 게이트 확률", "도메인 게이트 확률"), ("누적 재무 갭(조원)", "누적 재무 갭(조원)"))):
        f = fig_base(title=f"{ttl}, 민간 {int(sh*100)}%", h=320)
        f.add_trace(go.Scatter(x=[f"{v:g}" for v in sub["총규모(조원)"]], y=sub[m], mode="lines+markers", line=dict(color="#2a78d6", width=2), marker=dict(size=8), name=ttl))
        f.update_layout(showlegend=False, xaxis_title="총규모(조원)"); col.plotly_chart(f)
    c1, c2 = st.columns(2)
    for col, (m, ttl) in zip((c1, c2), (("프런티어 격차 중앙(년)", "2029 프런티어 격차(년)"), ("2029 전력 구속 확률", "2029 전력 구속 확률"))):
        f = fig_base(title=f"{ttl}, 민간 {int(sh*100)}%", h=320)
        f.add_trace(go.Scatter(x=[f"{v:g}" for v in sub["총규모(조원)"]], y=sub[m], mode="lines+markers", line=dict(color="#eb6834", width=2), marker=dict(size=8), name=ttl))
        f.update_layout(showlegend=False, xaxis_title="총규모(조원)"); col.plotly_chart(f)
    st.dataframe(df.round(3), hide_index=True)
    if st.button("격자 결과를 비교 목록에 저장"):
        save_run("격자", f"{mode}, {regime}, 격자점당 {N}회", df.to_dict("records")); st.success("저장했습니다.")
    st.caption("16.7조원은 보도된 정부 출자 5조원·지분 30%에서 역산한 규모. 공공성 이탈 위험지수는 민간 60%를 0, 80%를 1로 둔 참고 척도로 성과 계산에는 반영하지 않음.")
