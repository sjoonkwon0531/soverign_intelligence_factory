from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_pages._ui import pick, sc_label, cached_factory_run, cached_factory_econ, cached_cb, to_json, data_badge, fig_base, heatmap, save_run, regime_picker, INK
from core.factory import FACTORY_PATHS, FACTORY_PARAMS
from core.costbenefit import default_cb, breakeven_capture, fmt_breakeven

ss = st.session_state
st.title("민간 인텔리전스 팩토리와 B2G 전용 추진")
st.write("국내 대기업이 정부 출자 없이 프런티어 랩과 함께 짓는 대규모 AI 데이터센터(민간 팩토리)가 있을 때, 국가 SPC의 추진 방식별 성과를 비교합니다. "
         "민간 팩토리는 국가 할당 용량(전력·부지 지원의 대가)과 인재 파급을 주지만, 국내 문턱 위 상업 시장을 잠식합니다. "
         "B2G 전용은 SPC가 공공 조달 매출만으로 운영되는 경우입니다.")
data_badge()
FLOOR = 0.4

st.subheader("민간 팩토리 경로")
Ne = st.selectbox("팩토리 경제성 반복", [2000, 5000], index=0)
fe = {fk: cached_factory_econ(fk, Ne) for fk in FACTORY_PATHS}
rows = []
for fk, v in fe.items():
    rows.append({"경로": f"{fk} {FACTORY_PATHS[fk]['name']}", "2031 가동(GW)": v["gw31"][1], "2029 보유 H100e(만)": v["h100e29"][1] / 1e4,
                 "2029 세계 대비 비중": v["share29"][1], "건물·전력 투자(조원)": v["capex_shell"][1] / 1e12, "GPU 포함 투자(조원)": v["capex_full"][1] / 1e12,
                 "국내 부가가치(조원, 현재가치)": v["gva"][1] / 1e12, "2031 전력(TWh)": v["power_twh31"][1], "전력 보조 상당(조원)": v["power_subsidy"][1] / 1e12})
st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
st.caption("실현율 50～100%(계통·부지·앵커 계약 지연), IT MW당 H100e는 연 19% 개선. 2029 세계 대비 비중이 1% 안팎이라는 점이 핵심입니다: 민간 팩토리는 국내 산업 기반이지 연산 규모의 격차를 메우는 수단이 아닙니다.")

st.subheader("추진 방식 × 민간 팩토리")
c = st.columns([2, 2, 1, 1])
fks = pick(c[0], "민간 팩토리 경로", list(FACTORY_PATHS), lambda k: f"{k} {FACTORY_PATHS[k]['name']}", default=list(FACTORY_PATHS), multi=True)
base = ss.scen.get("S4b", list(ss.scen.values())[0]); s1 = ss.scen.get("S1", base)
VAR = {
    "A 분산 지속": dict(s1),
    "B B2G 전용(조달 보장)": dict(base, market_mult=0.0),
    "C B2G 전용(품질 연동)": dict(base, market_mult=0.0, b2g_quality_floor=FLOOR),
    "D B2G 전용(품질 연동, 실현 60%)": dict(base, market_mult=0.0, b2g_quality_floor=FLOOR, b2g_mult=0.6),
    "E 권고안(B2G와 민간·수출, 품질 연동)": dict(base, b2g_quality_floor=FLOOR),
    "F 권고안 경량(설비 절반)과 할당 용량": dict(base, b2g_quality_floor=FLOOR, capex_krw=[v * 0.5 for v in base["capex_krw"]]),
}
vks = c[1].multiselect("추진 방식", list(VAR), default=list(VAR))
N = c[2].selectbox("반복", [500, 1000, 2000], index=1)
pacing, reg = regime_picker(c[3], "fac_regime")
st.caption("품질 연동: 한국 모델이 오픈웨이트 문턱보다 5점 이상 뒤지면 공공 조달 중 보호 물량 40%만 국산으로 발주. 모든 경우에 흡수 역량 상한을 적용하므로 시나리오 화면의 값과 다를 수 있습니다.")
SCOPE = 0.22; PI_SHIFT = {"F0": 0.0, "F1": 0.03, "F2": 0.06}
if st.button("행렬 계산", type="primary") and fks and vks:
    out = []; prog = st.progress(0.0); tot = len(fks) * len(vks); i = 0
    for fk in fks:
        for vk in vks:
            sc = VAR[vk]
            r = cached_factory_run(to_json(sc), to_json(ss.P), fk, N, pacing)
            row = {"팩토리": fk, "추진 방식": vk, "도메인 게이트": r["G2029_dom"], "프런티어 격차(년)": r["lag_front"][1],
                   "격차 1년 이하": r["P_lagF_le1"], "누적 재무 갭(조원)": r["gap"][1], "주권 커버리지": r["sov"]}
            if not vk.startswith("A"):
                C = default_cb(); C["krw_per_usd"] = ss.P["krw_per_usd"]; C["pi_shift"] = PI_SHIFT[fk]
                if "B2G 전용" in vk: C["capture_scope"] = SCOPE
                cb = cached_cb(tuple(v * 1e12 for v in r["gap"]), r["G2029_dom"], tuple(float(v) for v in sc["capex_krw"][:4]), to_json(C), 10000, 20260922)
                row.update({"국가 순편익 2031(조원)": cb["spc"]["net_benefit"][1], "순편익 양일 확률": cb["spc"]["P_nb_pos"], "국가 순편익 2035(조원)": cb["h2035"]["net_benefit"][1]})
            out.append(row); i += 1; prog.progress(i / tot, f"{fk}, {vk}")
    prog.empty(); ss._fac = pd.DataFrame(out)
if "_fac" in ss:
    df = ss._fac
    met = st.selectbox("지표", ["도메인 게이트", "프런티어 격차(년)", "누적 재무 갭(조원)", "국가 순편익 2031(조원)", "국가 순편익 2035(조원)", "주권 커버리지"])
    piv = df.pivot(index="추진 방식", columns="팩토리", values=met)
    st.plotly_chart(heatmap(piv.values, list(piv.columns), list(piv.index), fmt="{:+.2f}" if "순편익" in met else "{:.2f}", diverging="순편익" in met,
                            title=met, xtitle="민간 팩토리 경로", ytitle="추진 방식", h=380))
    st.dataframe(df.round(3), hide_index=True)
    st.markdown("**읽는 법**: B2G 전용은 전환 대상이 공공 몫(문턱 위 지출의 약 22%)에 그치므로 국가 순편익이 구조적으로 음이 되기 쉽습니다. "
                "민간 팩토리의 할당 용량은 게이트 확률을 올리지만, 같은 팩토리가 국내 상업 시장을 잠식하므로 권고안의 민간·수출 매출 가정은 낮춰 보아야 합니다. "
                "경량안(F)은 할당 용량으로 설비투자를 대체해 재무 갭을 크게 줄입니다.")
    if st.button("행렬 결과 저장"):
        save_run("민간 팩토리", f"{reg}, {N}회", df.to_dict("records")); st.success("저장했습니다.")

pre = Path(__file__).resolve().parent.parent / "analysis" / "out" / "factory_b2g_matrix.csv"
if pre.exists():
    with st.expander("검토 메모(2026. 9. 30.)의 기준 계산 결과(2,000회)"):
        st.dataframe(pd.read_csv(pre).round(3), hide_index=True)
