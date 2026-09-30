"""국내 부가가치·취업유발(산업연관표 계수)과 공공(B2G) 수요 상향식 적산."""
import numpy as np
import pandas as pd
from .params import YEARS
from .sim import draw_common, simulate

GVA_PARAMS = dict(v_sw=0.893, job_sw=8.7, v_eq=(0.05, 0.15), v_dc=(0.55, 0.75), v_cap=(0.85, 0.95), job_cap=(5.0, 8.0), discount=0.05)
ALLOC = {"S1": (0.483, 0.207, 0.31), "S1B": (0.483, 0.207, 0.31), "S4b": (0.40, 0.175, 0.425), "S2b": (0.40, 0.175, 0.425),
         "S2": (0.40, 0.175, 0.425), "S4": (0.40, 0.175, 0.425), "S3": (0.40, 0.175, 0.425)}


def run_gva(keys, scen, P, alloc=None, G=None, N=3000, seed_sim=20260922, seed_va=20260927):
    """시나리오들을 같은 난수 흐름으로 차례로 실행(본보고서와 같은 순서면 같은 값)."""
    G = G or GVA_PARAMS; alloc = alloc or ALLOC
    rs = np.random.default_rng(seed_sim); rv = np.random.default_rng(seed_va)
    beta = 1 / (1 + G["discount"]); disc = np.array([beta ** i for i in range(len(YEARS))])
    out = {}
    for k in keys:
        sc = scen[k]; w_eq, w_dc, w_cap = alloc[k]
        go, gi, gt, jb, sc_, rp = [], [], [], [], [], []
        for _ in range(N):
            m = draw_common(rs, P); m["pacing"] = 1.0; r = simulate(sc, m, P)
            v = dict(v_eq=rv.uniform(*G["v_eq"]), v_dc=rv.uniform(*G["v_dc"]), v_cap=rv.uniform(*G["v_cap"]), job_cap=rv.uniform(*G["job_cap"]))
            g_out = float(np.sum(disc * r["rev"] * G["v_sw"]))
            g_inv = float(np.sum(disc * r["capex"] * (w_eq * v["v_eq"] + w_dc * v["v_dc"] + w_cap * v["v_cap"])))
            j = float(np.sum(r["rev"] / 1e9 * G["job_sw"]) + np.sum(r["capex"] * w_cap / 1e9 * v["job_cap"]))
            go.append(g_out); gi.append(g_inv); gt.append(g_out + g_inv); jb.append(j); sc_.append(r["gates"]["2029_domain"]); rp.append(float(np.sum(disc * r["rev"])))
        a = lambda x: np.array(x); s = a(sc_).astype(bool); gt = a(gt)
        p = lambda x: [float(np.percentile(x, q)) / 1e12 for q in (10, 50, 90)]
        out[k] = dict(P_success=float(s.mean()), rev_pv=p(a(rp)), gva_output=p(a(go)), gva_invest=p(a(gi)), gva_total=p(gt),
                      gva_success=p(gt[s]) if s.any() else None, gva_fail=p(gt[~s]) if (~s).any() else None,
                      jobs=[float(np.percentile(a(jb), q)) for q in (10, 50, 90)])
    return out


B2G_ITEMS = pd.DataFrame([
    # 군, 항목, 기준금액(억원), 기준연도, 저, 중, 고, 문턱 위, 근거
    ["A", "SW구축 운영·유지관리", 34313, 2026, 0.03, 0.06, 0.10, False, "레거시 운영 인력비 중심, 5년 내 3~10%"],
    ["A", "SW구축 신규 개발(AI 제외)", 11465, 2026, 0.10, 0.20, 0.35, False, "신규 시스템의 AI 내재화 비율"],
    ["A", "SW구축 AI 관련 사업", 3903, 2026, 0.60, 0.80, 1.00, False, "직접 AI 사업의 국산 모델 채택률"],
    ["A", "상용SW·SaaS", 4280, 2026, 0.15, 0.30, 0.45, False, "AI SaaS 전환분"],
    ["A", "ICT 장비", 10776, 2026, 0.05, 0.15, 0.25, False, "AI 서버·NPU 대체분"],
    ["B", "국방 AI", 3610, 2027, 0.25, 0.40, 0.55, True, "모델·플랫폼·SW 비중"],
    ["B", "행정 AI", 815, 2027, 0.50, 0.70, 0.90, False, "범정부 확산 시 국산 모델 채택률"],
    ["B", "모두의 AI", 2500, 2027, 0.40, 0.60, 0.80, False, "국산 모델 탑재 조건 시"],
    ["B", "AI 교육", 21000, 2027, 0.05, 0.10, 0.15, False, "AI 서비스 이용료 분"],
    ["B", "지역의료 AI", 12600, 2027, 0.03, 0.06, 0.10, True, "의료 AI 서비스·검증 분"],
    ["B", "피지컬 AI", 31000, 2027, 0.03, 0.06, 0.10, False, "기반 모델·시뮬레이션 서비스 분"],
    ["B", "디지털서비스 계약", 1500, 2027, 0.20, 0.35, 0.50, False, "AI 서비스 계약 전환분"],
], columns=["군", "항목", "기준금액(억원)", "기준연도", "대체율 저", "대체율 중", "대체율 고", "문턱 위", "근거"])


def b2g(items: pd.DataFrame, gA=0.04, gB=0.30, years=range(2027, 2032), spc_share=(0.40, 0.55, 0.70)):
    """연도별 조달 가능 규모(조원). A군은 연 gA, B군은 연 gB로 기준연도에서 성장."""
    rows = []
    for y in years:
        r = {"연도": y}
        for lab, col in (("저", "대체율 저"), ("중", "대체율 중"), ("고", "대체율 고")):
            g = np.where(items["군"] == "A", (1 + gA) ** (y - items["기준연도"]), (1 + gB) ** (y - items["기준연도"]))
            v = items["기준금액(억원)"] * items[col] * g
            r[f"조달 가능 {lab}"] = float(v.sum()) / 1e4
            r[f"문턱 위 {lab}"] = float(v[items["문턱 위"].astype(bool)].sum()) / 1e4
        for lab, sh in zip(("저", "중", "고"), spc_share):
            r[f"SPC 귀속 {lab}"] = r[f"조달 가능 {lab}"] * sh
        rows.append(r)
    return pd.DataFrame(rows)
