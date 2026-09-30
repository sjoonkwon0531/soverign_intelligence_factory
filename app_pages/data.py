import pandas as pd
import streamlit as st
from app_pages._ui import data_badge
from core.data_io import read_bytes, fetch_url, load_baseline, diff_eci, sha16, KIND_LABEL, DEFAULT_URLS

ss = st.session_state
st.title("데이터 관리")
st.write("새 자료를 파일로 올리거나 링크로 불러오면 형식을 자동으로 판별해 적용합니다. 적용한 자료는 격차 진단, 모수 보정, 전망에 바로 반영됩니다.")
data_badge()

st.subheader("현재 사용 중인 자료")
rows = []
for k, lab in KIND_LABEL.items():
    m = ss.meta.get(k, {})
    rows.append({"자료": lab, "출처": m.get("source"), "행 수": m.get("rows"), "SHA-256 앞 16자리": m.get("sha16")})
st.dataframe(pd.DataFrame(rows), hide_index=True)
c1, c2 = st.columns([1, 3])
ss.asof = c1.text_input("기준일(이 날짜까지의 자료만 사용)", ss.asof, help="YYYY-MM-DD. 새 자료를 올리면 자료의 마지막 날짜로 바꾸는 것을 권장합니다.")
if c2.button("기준본으로 되돌리기"):
    ss.data, ss.meta = load_baseline(); ss.data_label = "기준본(2026. 9. 22. 조회)"; ss.asof = "2026-09-22"; st.rerun()


def apply(items, src_label, raw_bytes):
    if not items:
        st.error("인식할 수 있는 자료가 없습니다. 지원 형식은 아래 설명을 참고하세요."); return
    for kind, obj, desc in items:
        if kind == "eci":
            d = diff_eci(ss.data["eci"], obj)
            st.success(f"{KIND_LABEL[kind]}: {desc}에서 {len(obj):,}행 인식")
            c = st.columns(4)
            c[0].metric("신규 모델", len(d["added"])); c[1].metric("점수 변경", len(d["changed"]))
            c[2].metric("프런티어 최고점", f"{d['frontier_new'][0]:.1f}", delta=f"{d['frontier_new'][0]-d['frontier_old'][0]:+.1f}", help=d["frontier_new"][1])
            c[3].metric("오픈웨이트 최고점", f"{d['open_new'][0]:.1f}", delta=f"{d['open_new'][0]-d['open_old'][0]:+.1f}", help=d["open_new"][1])
            if d["added"]:
                new = obj[obj["Model"].isin(d["added"])].sort_values("eci", ascending=False)[["Model", "Organization", "Country (of organization)", "Accessibility group", "date", "eci"]] if "Organization" in obj.columns else obj[obj["Model"].isin(d["added"])]
                st.dataframe(new.head(30), hide_index=True)
            ss.asof = d["last_date_new"]
        else:
            st.success(f"{KIND_LABEL[kind]}: {desc}" + (f"에서 {len(obj):,}행 인식" if hasattr(obj, "__len__") and not isinstance(obj, dict) else " 인식"))
        ss.data[kind] = obj
        ss.meta[kind] = dict(source=f"{src_label} ({desc})", sha16=sha16(raw_bytes), rows=len(obj) if hasattr(obj, "shape") else None)
    ss.data_label = "사용자 갱신 자료"
    ss.pop("_gap_memo", None)


st.subheader("파일 올리기")
st.caption("csv, xlsx, zip(Epoch benchmark_data.zip 그대로), json(컴퓨팅 보유 요약)을 지원합니다. 여러 파일을 한꺼번에 올려도 됩니다.")
ups = st.file_uploader("자료 파일", accept_multiple_files=True, type=["csv", "xlsx", "zip", "json"])
if ups and st.button("올린 파일 적용", type="primary"):
    for u in ups:
        b = u.getvalue()
        try:
            apply(read_bytes(u.name, b), f"업로드 {u.name}", b)
        except Exception as e:
            st.error(f"{u.name}: 읽지 못했습니다 ({e})")

st.subheader("링크로 불러오기")
st.caption("Epoch AI 자료 내려받기 주소는 바뀔 수 있습니다. 주소가 바뀌면 Epoch 누리집의 Download 버튼 주소를 붙여 넣으세요.")
u1 = st.text_input("Epoch 역량지수(benchmark_data.zip 또는 eci csv)", DEFAULT_URLS["eci"])
u2 = st.text_input("Epoch 주목 모델(notable_ai_models.csv)", DEFAULT_URLS["notable"])
u3 = st.text_input("기타 자료 주소(선택)", "")
if st.button("링크에서 불러와 적용"):
    for u in (u1, u2, u3):
        if not u.strip(): continue
        with st.spinner(f"{u} 불러오는 중"):
            try:
                name, b = fetch_url(u.strip())
                apply(read_bytes(name, b), f"링크 {u.strip()}", b)
            except Exception as e:
                st.error(f"{u}: 불러오지 못했습니다 ({e})")

st.subheader("한국 모델 AA 점수")
st.caption("한국 모델은 Epoch 역량지수에 수록되어 있지 않아 Artificial Analysis 지수를 환산합니다. 새 모델이 나오거나 점수가 재채점되면 여기서 고치세요. aa_low는 조회 시점별 변동의 하단입니다.")
kr = st.data_editor(ss.data["korea"][["model", "developer", "aa", "aa_low", "finalist", "note"]] if "developer" in ss.data["korea"].columns else ss.data["korea"],
                    num_rows="dynamic", key="kr_editor",
                    column_config={"finalist": st.column_config.CheckboxColumn("독자모델 결선"), "aa": st.column_config.NumberColumn("AA 지수"), "aa_low": st.column_config.NumberColumn("AA 하단")})
if st.button("한국 모델 점수 적용"):
    kr = kr.dropna(subset=["model", "aa"]).copy(); kr["aa_low"] = kr["aa_low"].fillna(kr["aa"]); kr["finalist"] = kr["finalist"].fillna(False).astype(bool)
    ss.data["korea"] = kr; ss.meta["korea"] = dict(source="앱에서 편집", sha16=sha16(kr.to_csv().encode()), rows=len(kr)); ss.pop("_gap_memo", None); st.success("적용했습니다.")

st.subheader("AA 지수와 역량지수 대응표")
st.caption("두 지수에 함께 수록된 모델입니다. 환산식(역량지수 = 기울기·AA + 절편)을 여기서 다시 추정합니다.")
pr = st.data_editor(ss.data["aa_pairs"], num_rows="dynamic", key="pairs_editor")
if st.button("대응표 적용"):
    pr = pr.dropna(subset=["aa", "eci"]); ss.data["aa_pairs"] = pr
    ss.meta["aa_pairs"] = dict(source="앱에서 편집", sha16=sha16(pr.to_csv().encode()), rows=len(pr)); ss.pop("_gap_memo", None); st.success("적용했습니다.")

with st.expander("지원 형식과 자동 판별 규칙"):
    st.markdown("""
- **Epoch 역량지수**: `eci`, `date`와 `Accessibility group`(또는 `Model accessibility`) 열이 있으면 인식합니다. zip이면 안의 `eci_scores.csv`를 찾습니다.
- **Epoch 주목 모델**: `Training compute (FLOP)`과 `Publication date` 열이 있으면 인식합니다. 전체 파일을 그대로 올려도 됩니다.
- **AA–역량지수 대응표**: `aa`, `eci` 열.
- **한국 모델 점수**: `model`, `aa` 열(선택: `aa_low`, `finalist`, `developer`).
- **컴퓨팅 보유 요약**: `KR_position_2025`, `global_stock_growth` 키가 있는 json(기준본 형식).
- 엑셀 파일은 시트마다 위 규칙으로 판별합니다.
""")
