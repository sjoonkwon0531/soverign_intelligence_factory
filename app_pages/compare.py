import io, json
import pandas as pd
import streamlit as st
from app_pages._ui import to_json

ss = st.session_state
st.title("결과 비교와 내보내기")
runs = ss.runs
if not runs:
    st.info("아직 저장한 결과가 없습니다. 각 화면의 '저장' 버튼으로 결과를 모을 수 있습니다.")
for i, r in enumerate(runs):
    with st.expander(f"{i+1}. {r['kind']}: {r['label']}"):
        p = r["payload"]
        if isinstance(p, list): st.dataframe(pd.DataFrame(p), hide_index=True)
        elif isinstance(p, dict) and "rows" in p: st.dataframe(pd.DataFrame(p["rows"]), hide_index=True)
        else: st.json({k: (float(v) if hasattr(v, "__float__") else v) for k, v in p.items()})

sc_runs = [r for r in runs if r["kind"] == "시나리오"]
if len(sc_runs) >= 2:
    st.subheader("시나리오 실행 간 비교")
    frames = []
    for r in sc_runs:
        d = pd.DataFrame(r["payload"]); d.insert(0, "실행", r["label"]); frames.append(d)
    allf = pd.concat(frames)
    metric = st.selectbox("지표", [c for c in allf.columns if c not in ("실행", "시나리오")])
    st.dataframe(allf.pivot_table(index="시나리오", columns="실행", values=metric).round(3))

st.subheader("내보내기")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    pd.DataFrame([{"키": k, "값": json.dumps(v, ensure_ascii=False, default=list)} for k, v in ss.P.items()]).to_excel(xw, sheet_name="시뮬레이션 모수", index=False)
    pd.DataFrame([{"키": k, **{kk: (json.dumps(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}} for k, v in ss.scen.items()]).to_excel(xw, sheet_name="시나리오 정의", index=False)
    for i, r in enumerate(runs):
        p = r["payload"]
        df = pd.DataFrame(p) if isinstance(p, list) else (pd.DataFrame(p["rows"]) if isinstance(p, dict) and "rows" in p else pd.DataFrame([p]))
        df.to_excel(xw, sheet_name=f"{i+1}_{r['kind']}"[:31], index=False)
st.download_button("모수, 시나리오, 저장 결과를 엑셀로 내려받기", buf.getvalue(), "simulator_results.xlsx")
st.download_button("현재 모수와 시나리오(json, 다시 불러오기용)", to_json(dict(P=ss.P, scen=ss.scen)).encode("utf-8"), "simulator_setup.json", "application/json")
up = st.file_uploader("저장해 둔 설정(json) 불러오기", type=["json"])
if up and st.button("설정 불러오기"):
    d = json.loads(up.getvalue().decode("utf-8"))
    P = d["P"]
    for k, v in P.items():
        if isinstance(v, list) and k != "L_choices": P[k] = tuple(v)
    P["L_choices"] = tuple(P["L_choices"]); ss.P = P; ss.scen = d["scen"]; st.success("불러왔습니다.")
if runs and st.button("저장 결과 모두 지우기"):
    ss.runs = []; st.rerun()
