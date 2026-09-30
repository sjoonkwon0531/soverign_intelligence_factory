"""시나리오 시뮬레이션 모형(연 단위, 몬테카를로).

물리층 S = min(C, E/ε), 역량층 A = A0 + a1·Δlog10K + ζ·g·t, 재무 = 설비투자 + 운영비 − 매출.
본보고서 r1(2026-09-30)의 모형과 수식·난수 순서가 같으므로 기본 모수와 시드에서 같은 값을 재현한다.
"""
import math
import numpy as np
from .params import YEARS, FLOP_PER_H100E_YEAR, K0, SEED, LAG_DIVISOR

# 원 모형의 추출 순서. (키, 추출 여부)
_ORDER = [("delta", 1), ("g_p", 1), ("p0_usd", 1), ("dc_capex_per_mw_usd", 1), ("eps_kw", 1), ("eps_it", 0),
          ("power_krw_kwh", 1), ("maint", 1), ("L", 1), ("eta_g", 1), ("zeta", 1), ("a1", 1), ("fr_drift", 1),
          ("diff_rate", 1), ("Afor0", 0), ("gap_open", 1), ("A_kr0", 1), ("zeta0", 1), ("g_zeta", 1), ("pi_us", 1),
          ("pi_open", 1), ("b2g_real", 1), ("market_above_thr_krw", 1), ("rent_per_point", 1)]


def draw_common(rng, P):
    """모수 한 벌 추출. 고정값으로 바꾼 모수도 난수를 하나 소비해 다른 모수의 표본을 유지한다."""
    m = {}
    for k, drawn in _ORDER:
        if k == "L":
            ch, pr = P["L_choices"]
            m["L"] = int(rng.choice(ch, p=pr))
            continue
        v = P[k]
        if not drawn:
            m[k] = float(v if not isinstance(v, (tuple, list)) else v[0])
            continue
        if isinstance(v, (tuple, list)):
            m[k] = float(rng.uniform(v[0], v[1]))
        else:
            rng.uniform(); m[k] = float(v)
    return m


def simulate(sc, m, P, x=None):
    """단일 경로. 반환: 연도별 상태, 게이트, 재무.

    x(선택): 민간 인텔리전스 팩토리 연계 입력. slice(연도별 국가 할당 H100e), slice_train, slice_cost(구매가 대비 연 비용률),
    Tcap0·gT(흡수 역량 상한), spill(흡수율 가산), crowd(국내 상업 시장 잠식률). 없으면 본보고서 모형과 같음."""
    KRW = P["krw_per_usd"]
    T = len(YEARS)
    C_vint = []
    C = np.zeros(T); E = np.zeros(T); S = np.zeros(T); K = 0.0; A_own = np.full(T, 20.0)
    capex = np.zeros(T); opex = np.zeros(T); rev = np.zeros(T)
    gates = {2027: False, 2028: False, 2029: False, "2029_domain": False}; stopped_at = None
    E_committed = {}
    used_slice = np.zeros(T)
    E_base = list(sc["E_pipeline_mw"])
    fast_L = max(1, m["L"] - 1) if sc["fast_track"] else m["L"]
    pace = m.get("pacing", 1.0)
    for i, y in enumerate(YEARS):
        if stopped_at is not None:
            C_vint = [(vy, a * (1 - m["delta"])) for vy, a in C_vint]
            C[i] = sum(a for _, a in C_vint); E[i] = E[i - 1]; S[i] = min(C[i], E[i] * 1000 / m["eps_it"])
            opex[i] = C[i] * m["eps_kw"] * 8760 * m["power_krw_kwh"] * 0.7
            rev[i] = rev[i - 1] * 0.8; A_own[i] = A_own[i - 1]
            continue
        p_t = m["p0_usd"] * math.exp(-m["g_p"] * i) * KRW
        budget = sc["capex_krw"][i]
        eq_budget, dc_budget = budget * 0.7, budget * 0.3
        new_h100e = eq_budget / p_t if sc["own_frontier"] or sc.get("open_fallback") or sc["n_teams"] == 0 else 0
        new_mw = dc_budget / (m["dc_capex_per_mw_usd"] * KRW)
        arrive = y + fast_L
        E_committed[arrive] = E_committed.get(arrive, 0) + new_mw
        C_vint = [(vy, a * (1 - m["delta"])) for vy, a in C_vint] + [(y, new_h100e)]
        C[i] = sum(a for _, a in C_vint)
        E[i] = E_base[i] + sum(v for ay, v in E_committed.items() if ay <= y)
        S[i] = min(C[i], E[i] * 1000 / m["eps_it"])
        eta_t = (m["eta_g"] ** m["zeta"]) ** i
        share_max_team = 1.0 / sc["n_teams"] if sc["n_teams"] > 0 else 0.0
        if x is None:
            K += S[i] * sc["train_share"] * share_max_team * FLOP_PER_H100E_YEAR * eta_t
        else:
            sl = x["slice"][i] if sc["own_frontier"] else 0.0
            own_train = S[i] * sc["train_share"] * share_max_team
            cap_i = x["Tcap0"] * x["gT"] ** i if x.get("Tcap0") else float("inf")
            if x.get("on_demand", True):   # 흡수 가능한 만큼만 할당 용량을 쓰고 이용료도 그만큼만 냄
                sl = min(sl, max(0.0, (cap_i - own_train) / (x.get("slice_train", 0.8) * max(share_max_team, 1e-9))))
            used_slice[i] = sl
            train = min(own_train + sl * x.get("slice_train", 0.8) * share_max_team, cap_i)
            K += train * FLOP_PER_H100E_YEAR * eta_t
        gz = m["g_zeta"] + (P["bonus_consolidation"] if sc["n_teams"] == 1 else 0.0) + (P["bonus_verification"] if sc["verif"] else 0.0)
        if x is not None: gz += x.get("spill", 0.0)
        zeta_t = min(1.0, m["zeta0"] + gz * i)
        tfac = min(i, 1) + pace * max(0, i - 1)
        A_own[i] = (m["A_kr0"] + m["a1"] * (math.log10(max(K + K0, K0)) - math.log10(K0)) + zeta_t * m["diff_rate"] * tfac) if sc["own_frontier"] else 0.0
        Afor_t = m["Afor0"] + m["fr_drift"] * tfac
        qc_t = Afor_t - m["gap_open"]
        b2g = sc["B2G"][i] * m["b2g_real"] * sc.get("b2g_mult", 1.0)
        if sc.get("b2g_quality_floor") is not None and sc["own_frontier"]:   # 검증 인증 조달: 문턱 근처 성능이 아니면 보호 물량만 국산
            ok = A_own[i] >= qc_t - P["gate_2029_domain_margin"]
            b2g *= 1.0 if ok else sc["b2g_quality_floor"]
        above = max(0.0, A_own[i] - qc_t) if sc["own_frontier"] else 0.0
        commercial = m["market_above_thr_krw"] * min(1.0, above * m["rent_per_point"]) * (1.0 if i >= 2 else 0.3)
        if "market_mult" in sc or x is not None: commercial *= sc.get("market_mult", 1.0) * (1 - (x.get("crowd", 0.0) if x else 0.0))
        verif_rev = P["verif_revenue_krw"] if sc["verif"] and i >= 1 else 0.0
        rev[i] = b2g + commercial + verif_rev
        capex[i] = budget
        opex[i] = C[i] * m["eps_kw"] * 8760 * m["power_krw_kwh"] + C[i] * p_t * m["maint"]
        if x is not None and sc["own_frontier"]: opex[i] += used_slice[i] * p_t * x.get("slice_cost", 0.0)
        if y == 2027: gates[2027] = sc["own_frontier"] and A_own[i] >= qc_t - P["gate_2027_margin"]
        if y == 2028: gates[2028] = sc["own_frontier"] and A_own[i] >= qc_t and commercial > 0
        if y == 2029:
            gates[2029] = sc["own_frontier"] and A_own[i] >= Afor_t - P["gate_2029_frontier_margin"]
            gates["2029_domain"] = sc["own_frontier"] and A_own[i] >= qc_t - P["gate_2029_domain_margin"]
        if sc["gate_stop"] and y in (2027, 2028) and not gates[y] and stopped_at is None:
            stopped_at = y
        sr = sc.get("stop_rule")   # 합리적 중단 규칙: (연도, 여유) 해당 연도에 오픈웨이트 대비 여유보다 뒤지면 중단
        if sr and y == sr[0] and stopped_at is None and sc["own_frontier"] and A_own[i] < qc_t - sr[1]:
            stopped_at = y
    cum_gap = float(np.sum(capex + opex - rev))
    i29 = YEARS.index(2029)
    p29 = m["p0_usd"] * math.exp(-m["g_p"] * i29) * KRW
    residual = C[i29] * p29 + 0.5 * sum(sc["capex_krw"][:i29 + 1]) * 0.3
    qc29 = m["Afor0"] + m["fr_drift"] * (1 + pace * (i29 - 1)) - m["gap_open"]
    own_ok = 1.0 if (sc["own_frontier"] and A_own[i29] >= qc29) else 0.0
    open_ok = m["pi_open"] * 0.7 if (not sc["own_frontier"] or sc.get("open_fallback")) else 0.0
    return dict(C=C, E=E, S=S, A_own=A_own, capex=capex, opex=opex, rev=rev, gates=gates, stopped_at=stopped_at,
                cum_gap=cum_gap, used_slice=used_slice, residual=residual, sov_cov=max(own_ok, open_ok), A_2029=A_own[i29], qc29=qc29, Afor29=qc29 + m["gap_open"])


def pct(x, q=(10, 50, 90)):
    x = np.asarray(x, dtype=float)
    return [float(np.percentile(x, v)) for v in q]


def run_scenario(sc, P, N=3000, seed=SEED, pacing=1.0, tweak=None, keep_paths=True, xdraw=None, xseed=7, keep_params=False):
    """시나리오 하나를 N회 실행. tweak(m)은 추출된 모수를 경로마다 수정하는 함수(레버·흡수 차단 등)."""
    rng = np.random.default_rng(seed); rx = np.random.default_rng(xseed)
    GF, GO = LAG_DIVISOR["frontier"], LAG_DIVISOR["open"]
    xs = []; mlist = []; per = []
    rec = dict(g27=0, g28=0, g29=0, dom=0, stop=0, eb=0)
    gaps, res, sov, lo, lf, A, F, Q, capex, opex, rev, succ, capex_tot = [], [], [], [], [], [], [], [], [], [], [], [], []
    for _ in range(N):
        m = draw_common(rng, P); m["pacing"] = pacing
        if tweak: tweak(m)
        if keep_params: mlist.append(dict(m))
        x = xdraw(rx) if xdraw else None
        if x is not None: xs.append(x)
        r = simulate(sc, m, P, x)
        rec["g27"] += r["gates"][2027]; rec["g28"] += r["gates"][2028]; rec["g29"] += r["gates"][2029]; rec["dom"] += r["gates"]["2029_domain"]
        rec["stop"] += r["stopped_at"] is not None; rec["eb"] += (r["E"][2] * 1000 / m["eps_it"] < r["C"][2])
        gaps.append(r["cum_gap"]); res.append(r["residual"]); sov.append(r["sov_cov"]); capex_tot.append(float(np.sum(r["capex"])))
        succ.append(bool(r["gates"]["2029_domain"]))
        tf = np.array([min(i, 1) + pacing * max(0, i - 1) for i in range(6)]); Af = m["Afor0"] + m["fr_drift"] * tf; qc = Af - m["gap_open"]
        if keep_paths:
            A.append(r["A_own"]); F.append(Af); Q.append(qc); capex.append(r["capex"]); opex.append(r["opex"]); rev.append(r["rev"])
        if sc["own_frontier"]:
            lo.append((qc[3] - r["A_own"][3]) / GO); lf.append((Af[3] - r["A_own"][3]) / GF)
        if keep_params:
            per.append(dict(success=float(r["gates"]["2029_domain"]), cum_gap=r["cum_gap"] / 1e12, lag_f=(Af[3] - r["A_own"][3]) / GF if sc["own_frontier"] else np.nan,
                            stopped=float(r["stopped_at"] is not None), sov=r["sov_cov"], residual=r["residual"] / 1e12))
    out = dict(N=N, seed=seed, pacing=pacing,
               G2027=rec["g27"] / N, G2028=rec["g28"] / N, G2029_front=rec["g29"] / N, G2029_dom=rec["dom"] / N,
               P_stop=rec["stop"] / N, P_ebind2028=rec["eb"] / N,
               gap=[v / 1e12 for v in pct(gaps)], residual=[v / 1e12 for v in pct(res)], sov=float(np.mean(sov)),
               capex_total=float(np.median(capex_tot)) / 1e12,
               lag_open=pct(lo) if lo else None, lag_front=pct(lf) if lf else None,
               P_lagF_le1=float(np.mean(np.array(lf) <= 1)) if lf else None)
    if keep_params:
        out["params"] = mlist; out["per_path"] = per
    if keep_paths:
        A = np.array(A); F = np.array(F); Q = np.array(Q)
        out["paths"] = dict(A=A, F=F, Q=Q, capex=np.array(capex), opex=np.array(opex), rev=np.array(rev), success=np.array(succ),
                            cum_gap=np.array(gaps), lag_f=np.array(lf) if lf else None)
        out["traj"] = dict(years=YEARS, A_p10=np.percentile(A, 10, 0).tolist(), A_p50=np.percentile(A, 50, 0).tolist(),
                           A_p90=np.percentile(A, 90, 0).tolist(), F_p50=np.percentile(F, 50, 0).tolist(), Q_p50=np.percentile(Q, 50, 0).tolist(),
                           lag_f=[float((F[:, i] - A[:, i]).mean() / GF) for i in range(6)] if sc["own_frontier"] else None)
    return out


def summary_row(key, sc, r):
    return {"시나리오": f"{key} {sc['name']}", "2029 도메인 게이트 확률": r["G2029_dom"], "2029 프런티어 게이트 확률": r["G2029_front"],
            "2027 게이트": r["G2027"], "2028 게이트": r["G2028"], "조기 중단 확률": r["P_stop"],
            "누적 재무 갭 중앙(조원)": r["gap"][1], "누적 재무 갭 10%": r["gap"][0], "누적 재무 갭 90%": r["gap"][2],
            "주권 커버리지": r["sov"], "2029 오픈웨이트 격차(년)": (r["lag_open"] or [None, None, None])[1],
            "2029 프런티어 격차(년)": (r["lag_front"] or [None, None, None])[1], "프런티어 격차 1년 이하 확률": r["P_lagF_le1"],
            "2028 전력 구속 확률": r["P_ebind2028"], "총투입(조원)": r["capex_total"]}


LEVERS = {
    "자본 2배": ("capex", 2.0),
    "출발점 153(최상위 팀)": ("A_kr0", 153.0),
    "출발점 147.3(결선 평균)": ("A_kr0", 147.3),
    "결집·검증 흡수 가속 제거": ("no_bonus", None),
    "흡수율 추가 가속 +0.05/년": ("g_zeta_add", 0.05),
}


def run_levers(sc, P, N=3000, seed=SEED):
    base = run_scenario(sc, P, N, seed, keep_paths=False)["G2029_dom"]
    out = {"기준": base}
    for name, (kind, v) in LEVERS.items():
        s2 = dict(sc); tw = None
        if kind == "capex": s2 = dict(sc, capex_krw=[x * v for x in sc["capex_krw"]])
        elif kind == "A_kr0": tw = lambda m, v=v: m.update(A_kr0=v)
        elif kind == "no_bonus":
            s2 = dict(sc, verif=False); tw = lambda m: m.update(g_zeta=m["g_zeta"] - (P["bonus_consolidation"] if sc["n_teams"] == 1 else 0.0))
        elif kind == "g_zeta_add": tw = lambda m, v=v: m.update(g_zeta=m["g_zeta"] + v)
        out[name] = run_scenario(s2, P, N, seed, tweak=tw, keep_paths=False)["G2029_dom"]
    out["감속 체제"] = run_scenario(sc, P, N, seed, pacing=0.5, keep_paths=False)["G2029_dom"]
    return out


def absorption_closure(sc, P, N=3000, seed=SEED):
    a = run_scenario(sc, P, N, seed, keep_paths=False)["lag_front"]
    b = run_scenario(sc, P, N, seed, tweak=lambda m: m.update(diff_rate=0.0), keep_paths=False)["lag_front"]
    return dict(open=a, closed=b)
