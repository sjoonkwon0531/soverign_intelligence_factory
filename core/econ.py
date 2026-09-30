"""경제성 지표: SPC 재무 관점(NPV, IRR, 회수기간, 편익비용비, ROI)과 비용 효과성.

시뮬레이션 경로의 연도별 설비투자·운영비·매출을 현금흐름으로 쓰고, 2031년 말 종료가치는
(1) 잔존가치 방식 또는 (2) 2032~2035 운영 지속 방식 중 선택한다. 국가 관점 순편익은 costbenefit 모듈이 계산한다.
"""
import numpy as np


def npv(cf, r):
    return float(sum(v / (1 + r) ** i for i, v in enumerate(cf)))


def irr(cf, lo=-0.99, hi=3.0, tol=1e-6):
    f = lambda r: npv(cf, r)
    a, b = f(lo), f(hi)
    if np.isnan(a) or np.isnan(b) or a * b > 0:
        return None
    for _ in range(200):
        m = (lo + hi) / 2; fm = f(m)
        if abs(fm) < tol: return m
        if a * fm < 0: hi, b = m, fm
        else: lo, a = m, fm
    return (lo + hi) / 2


def payback(cf):
    c = np.cumsum(cf)
    idx = np.where(c >= 0)[0]
    return int(idx[0]) if len(idx) else None


def spc_financials(paths, r=0.05, terminal="residual", residual_frac=0.30, ext_years=4, ext_growth=0.0, ext_margin=None):
    """paths: run_scenario(...)['paths']. 반환: 경로별 지표 배열과 요약.

    terminal="residual": 2031년 말에 누적 설비투자의 residual_frac 만큼 회수한다고 가정
    terminal="operate" : 2032~2035에 2031년 영업현금흐름(매출−운영비)이 ext_growth로 변한다고 가정(ext_margin으로 덮어쓰기 가능)
    """
    capex, opex, rev = paths["capex"], paths["opex"], paths["rev"]
    n = capex.shape[0]
    out = dict(npv=np.zeros(n), irr=np.full(n, np.nan), payback=np.full(n, np.nan), bcr=np.zeros(n), roi=np.zeros(n))
    for k in range(n):
        cf = list(rev[k] - capex[k] - opex[k])
        if terminal == "residual":
            cf[-1] += residual_frac * capex[k].sum()
        else:
            op = (rev[k][-1] - opex[k][-1]) if ext_margin is None else ext_margin
            cf += [op * (1 + ext_growth) ** (j + 1) for j in range(ext_years)]
        infl = [max(v, 0) for v in cf]; outf = [max(-v, 0) for v in cf]
        pv_in = npv(rev[k].tolist() + ([0] * (len(cf) - 6)), r) + (npv([0] * 5 + [residual_frac * capex[k].sum()], r) if terminal == "residual" else npv([0] * 6 + cf[6:], r))
        pv_out = npv((capex[k] + opex[k]).tolist(), r)
        out["npv"][k] = npv(cf, r)
        v = irr(cf); out["irr"][k] = np.nan if v is None else v
        pb = payback(cf); out["payback"][k] = np.nan if pb is None else 2026 + pb
        out["bcr"][k] = pv_in / pv_out if pv_out > 0 else np.nan
        out["roi"][k] = (pv_in - pv_out) / pv_out if pv_out > 0 else np.nan
    s = paths["success"]
    q = lambda x: [float(np.nanpercentile(x, p)) for p in (10, 50, 90)] if np.isfinite(x).any() else [np.nan] * 3
    summ = dict(npv=[v / 1e12 for v in q(out["npv"])], P_npv_pos=float(np.mean(out["npv"] > 0)), irr=q(out["irr"]),
                irr_defined=float(np.mean(np.isfinite(out["irr"]))), bcr=q(out["bcr"]), roi=q(out["roi"]),
                P_payback=float(np.mean(np.isfinite(out["payback"]))),
                npv_success=[v / 1e12 for v in q(out["npv"][s])] if s.any() else None, npv_fail=[v / 1e12 for v in q(out["npv"][~s])] if (~s).any() else None)
    return out, summ


def cost_effectiveness(res, base_key="S1"):
    """시나리오별 결과(run_scenario 요약)로 비용 효과성 표를 만든다. 기준 대비 게이트 확률 0.1과 격차 0.1년당 누적 갭."""
    b = res[base_key]
    rows = []
    for k, r in res.items():
        d_gate = r["G2029_dom"] - b["G2029_dom"]
        d_lag = (b["lag_front"][1] - r["lag_front"][1]) if (r.get("lag_front") and b.get("lag_front")) else None
        d_gap = r["gap"][1] - b["gap"][1]
        rows.append({"시나리오": k, "누적 재무 갭(조원)": r["gap"][1], "기준 대비 갭 차이(조원)": d_gap,
                     "기준 대비 게이트 확률 차이": d_gate, "기준 대비 격차 단축(년)": d_lag,
                     "누적 갭 1조원당 게이트 확률": (r["G2029_dom"] / r["gap"][1]) if r["gap"][1] > 0.05 else None,
                     "총투입(조원)": r["capex_total"], "총투입 1조원당 게이트 확률": r["G2029_dom"] / r["capex_total"] if r["capex_total"] > 0 else None})
    return rows
