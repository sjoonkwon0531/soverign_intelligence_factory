"""SPC 규모 × 민간 출자 비중 격자 모형.

권고 구조(이중 트랙, 에스크로, 패스트트랙, 단일 주체)를 상속하고, 출자 여력에 따른 실현 자본, 흡수 역량 상한,
계통 접속 상한, GPU 공급 상한, 임대 방식을 추가한다. 격자점마다 시드를 고정해 재현 가능하게 한다.
"""
import math
import numpy as np
from .params import YEARS, FLOP_PER_H100E_YEAR, K0, GRID_EXTRA, GRID_SHAPE, LAG_DIVISOR, SEED
from .sim import draw_common, pct


def draw_extra(rng, X):
    return {k: float(rng.uniform(*v)) if isinstance(v, (tuple, list)) else float(v) for k, v in X.items()}


def simulate_grid(total_krw, s, m, x, P, mode="buy", shape=None, B2G_base=None):
    KRW = P["krw_per_usd"]
    T = len(YEARS)
    SHAPE = np.array(shape or GRID_SHAPE)
    C_vint = []; C = np.zeros(T); E = np.zeros(T); S = np.zeros(T); K = 0.0; A_own = np.zeros(T)
    capex = np.zeros(T); opex = np.zeros(T); rev = np.zeros(T); realized = np.zeros(T)
    E_committed = {}; E_base = [50, 80, 110, 140, 170, 200]
    fast_L = max(1, m["L"] - 1)
    plan = SHAPE * total_krw
    req_p = plan * s; req_g = plan * (1 - s)
    f_p = np.minimum(1.0, x["Cp"] / np.maximum(req_p, 1e9)); f_g = np.minimum(1.0, x["Cg"] / np.maximum(req_g, 1e9))
    real = plan * (s * f_p + (1 - s) * f_g)
    gz_bonus = P["bonus_consolidation"] + P["bonus_verification"] + 0.04 * (s - 0.5) / 0.3
    mkt_mult = 0.8 + 0.4 * s
    b2g_mult = min(1.0, 0.7 + (1 - s))
    b2g_scale = min(1.0, (total_krw / 10e12) ** 0.5)
    B2G = [v * b2g_scale for v in (B2G_base or [0.5e12, 1.0e12, 1.5e12, 1.8e12, 2.0e12, 2.0e12])]
    train_share = 0.45
    ebind = False
    pace = m.get("pacing", 1.0)
    for i, y in enumerate(YEARS):
        p_t = m["p0_usd"] * math.exp(-m["g_p"] * i) * KRW
        budget = real[i]; realized[i] = budget
        eq_budget, dc_budget = budget * 0.7, budget * 0.3
        if mode == "lease":
            lr = x["lease_rate"]
            leased = min(eq_budget / (p_t * lr), x["Gcap"] * 1.5)
            unspent = eq_budget - leased * p_t * lr; budget -= unspent; realized[i] = budget
            new_h100e = 0.0
        else:
            new_h100e = min(eq_budget / p_t, x["Gcap"])
            unspent = eq_budget - new_h100e * p_t; budget -= unspent; realized[i] = budget
            leased = 0.0
        new_mw = dc_budget / (m["dc_capex_per_mw_usd"] * KRW)
        arrive = y + fast_L
        E_committed[arrive] = E_committed.get(arrive, 0) + new_mw
        C_vint = [(vy, a * (1 - m["delta"])) for vy, a in C_vint] + [(y, new_h100e)]
        C[i] = sum(a for _, a in C_vint) + leased
        E[i] = min(E_base[i] + sum(v for ay, v in E_committed.items() if ay <= y), E_base[i] + x["Ecap_mw"])
        S[i] = min(C[i], E[i] * 1000 / m["eps_it"])
        if i == 3 and S[i] < C[i] * 0.999: ebind = True
        Tcap = x["Tcap0"] * (x["gT"] ** i)
        S_train = min(S[i] * train_share, Tcap)
        eta_t = (m["eta_g"] ** m["zeta"]) ** i
        K += S_train * FLOP_PER_H100E_YEAR * eta_t
        zeta_t = min(1.0, m["zeta0"] + (m["g_zeta"] + gz_bonus) * i)
        tfac = min(i, 1) + pace * max(0, i - 1)
        A_own[i] = m["A_kr0"] + m["a1"] * (math.log10(max(K + K0, K0)) - math.log10(K0)) + zeta_t * m["diff_rate"] * tfac
        Afor_t = m["Afor0"] + m["fr_drift"] * tfac; qc_t = Afor_t - m["gap_open"]
        b2g = B2G[i] * m["b2g_real"] * b2g_mult
        above = max(0.0, A_own[i] - qc_t)
        commercial = m["market_above_thr_krw"] * mkt_mult * min(1.0, above * m["rent_per_point"]) * (1.0 if i >= 2 else 0.3)
        rev[i] = b2g + commercial + (P["verif_revenue_krw"] if i >= 1 else 0.0)
        capex[i] = budget
        opex[i] = C[i] * m["eps_kw"] * 8760 * m["power_krw_kwh"] + (C[i] - leased) * p_t * m["maint"]
    i29 = 3; tf29 = 1 + pace * 2
    Af29 = m["Afor0"] + m["fr_drift"] * tf29; qc29 = Af29 - m["gap_open"]
    p29 = m["p0_usd"] * math.exp(-m["g_p"] * i29) * KRW
    residual = (C[i29] * p29 if mode == "buy" else 0.0) + 0.5 * float(np.sum(realized[:i29 + 1])) * 0.3
    return dict(dom=A_own[i29] >= qc29 - P["gate_2029_domain_margin"], front=A_own[i29] >= Af29 - P["gate_2029_frontier_margin"],
                lag_f=(Af29 - A_own[i29]) / LAG_DIVISOR["frontier"], lag_o=(qc29 - A_own[i29]) / LAG_DIVISOR["open"],
                cum_gap=float(np.sum(capex + opex - rev)), residual=residual, sov=max(1.0 if A_own[i29] >= qc29 else 0.0, m["pi_open"] * 0.7),
                ebind=ebind, realized=float(realized.sum()), gov_out=float((real * (1 - s)).sum()), priv_out=float((real * s).sum()),
                fp=float(np.mean(f_p[:5])), fg=float(np.mean(f_g[:5])))


def run_cell(size_trillion, share, P, X=None, N=1000, seed=SEED, pacing=1.0, mode="buy"):
    X = X or GRID_EXTRA
    rng = np.random.default_rng(seed + int(size_trillion * 10) * 101 + int(share * 100))
    acc = {k: [] for k in ["dom", "front", "lag_f", "lag_o", "cum_gap", "residual", "sov", "ebind", "realized", "gov_out", "priv_out", "fp", "fg"]}
    for _ in range(N):
        m = draw_common(rng, P); m["pacing"] = pacing; x = draw_extra(rng, X)
        r = simulate_grid(size_trillion * 1e12, share, m, x, P, mode)
        for k in acc: acc[k].append(r[k])
    a = {k: np.array(v, dtype=float) for k, v in acc.items()}
    lf = pct(a["lag_f"])
    return {"총규모(조원)": size_trillion, "민간 비중": share, "방식": "임대" if mode == "lease" else "구매",
            "도메인 게이트 확률": a["dom"].mean(), "프런티어 게이트 확률": a["front"].mean(),
            "프런티어 격차 중앙(년)": lf[1], "격차 10%": lf[0], "격차 90%": lf[2], "격차 1년 이하 확률": float((a["lag_f"] <= 1).mean()),
            "누적 재무 갭(조원)": float(np.median(a["cum_gap"])) / 1e12, "2029 잔존가치(조원)": float(np.median(a["residual"])) / 1e12,
            "주권 커버리지": a["sov"].mean(), "2029 전력 구속 확률": a["ebind"].mean(), "실현 자본(조원)": float(np.median(a["realized"])) / 1e12,
            "정부 출자 실현(조원)": float(np.median(a["gov_out"])) / 1e12, "민간 출자 실현(조원)": float(np.median(a["priv_out"])) / 1e12,
            "민간 실현율": float(a["fp"].mean()), "정부 실현율": float(a["fg"].mean()),
            "공공성 이탈 위험지수": float(np.clip((share - 0.6) / 0.2, 0, 1))}


def run_grid(sizes, shares, P, X=None, N=1000, seed=SEED, pacing=1.0, mode="buy", progress=None):
    rows = []; total = len(sizes) * len(shares); k = 0
    for T in sizes:
        for s in shares:
            rows.append(run_cell(T, s, P, X, N, seed, pacing, mode)); k += 1
            if progress: progress(k / total)
    return rows
