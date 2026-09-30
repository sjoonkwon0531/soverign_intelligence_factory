"""실패 시 잔존가치, 임차 경로 종속비용, SPC 순편익(국가 관점).

본보고서 r1과 같은 수식·난수 순서(기본 시드 20260922)를 쓴다. 모든 분포는 CB_PARAMS에서 바꿀 수 있다.
"""
import numpy as np
from copy import deepcopy

CB_PARAMS = {
    "discount": 0.05,
    "krw_per_usd": 1350.0,
    "delta": (0.25, 0.33), "g_p": (0.15, 0.25), "p0_usd": (15e3, 25e3),
    "eq_share": 0.70,
    "resale_disc": (0.40, 0.65),        # 중고 GPU 할인
    "dc_remain": (0.70, 0.85), "dc_reuse": (0.60, 0.90),
    "people_data_annual": (0.20e12, 0.35e12),
    "cap_retain_nation": (0.30, 0.60), "cap_retain_spc": (0.0, 0.15),
    "spend0": (1.0e12, 2.5e12),         # 2026 문턱 위 AI 지출
    "g_spend": (0.25, 0.45), "foreign_share": (0.70, 0.90), "premium": (0.10, 0.30),
    "surplus_mult": (2.0, 4.0), "pi_us": (0.5, 0.9), "pi_open": (0.6, 0.95),
    "T_block": (0.5, 2.0), "switch_frac": (0.15, 0.35),
    "capture": (0.20, 0.40),            # 성공 시 국산 전환율
    "va_margin": (0.30, 0.50), "cov_gain": (0.40, 0.70),
    "ext_growth": 0.15, "spc_annual_ext": (0.3e12, 1.2e12),
}
CB_LABELS = {
    "discount": "할인율", "resale_disc": "중고 GPU 할인", "dc_remain": "인프라 잔존율", "dc_reuse": "인프라 타용도 회수율",
    "people_data_annual": "연 인력·데이터 지출(원)", "cap_retain_nation": "역량 잔존율(국가)", "spend0": "2026 문턱 위 AI 지출(원)",
    "g_spend": "지출 성장률", "foreign_share": "외산 비중", "premium": "경쟁 결여 프리미엄", "surplus_mult": "이용자 잉여 승수",
    "pi_us": "미국 접근 유지 확률", "T_block": "차단 지속(년)", "switch_frac": "전환비용 비율", "capture": "국산 전환율(성공 시)",
    "va_margin": "국내 잔류 부가가치", "cov_gain": "차단 손실 축소율", "spc_annual_ext": "2032~35 SPC 연 순비용(원)",
}


def default_cb():
    return deepcopy(CB_PARAMS)


def run_cb(cum_gap_tri, p_success, capex_path, C=None, N=20000, seed=20260922, ps_grid=(0.2, 0.31, 0.4, 0.5, 0.6, 0.7),
           cap_grid=(0.1, 0.2, 0.3, 0.4, 0.5), cap_grid35=(0.2, 0.3, 0.4, 0.5), keep=False):
    """cum_gap_tri: 누적 갭 (10, 50, 90 백분위) 원. capex_path: 2026~2029 설비투자 4개 값(원)."""
    C = C or CB_PARAMS
    rng = np.random.default_rng(seed)
    DISC = C["discount"]; KRW = C["krw_per_usd"]; YN = 6
    U = lambda k: rng.uniform(*C[k], N) if isinstance(C[k], (tuple, list)) else np.full(N, float(C[k])) + 0 * rng.uniform(0, 1, N)
    pv = lambda arr: sum(v / (1 + DISC) ** i for i, v in enumerate(arr))
    # A. 잔존가치
    delta = U("delta"); g_p = U("g_p"); p0 = U("p0_usd") * KRW
    eq_share = C["eq_share"]
    book_eq = np.zeros(N); repl_eq = np.zeros(N); salv_eq = np.zeros(N)
    resale_disc = U("resale_disc")
    for i, cx in enumerate(capex_path):
        p_t = p0 * np.exp(-g_p * i); units = cx * eq_share / p_t
        age = 3 - i
        book_eq += cx * eq_share * np.clip(1 - (age + 1) / 4.0, 0, 1)
        remain = units * (1 - delta) ** (age + 1)
        p29 = p0 * np.exp(-g_p * 3)
        repl_eq += remain * p29; salv_eq += remain * p29 * (1 - resale_disc)
    dc_capex = sum(capex_path) * (1 - eq_share)
    dc_remain = U("dc_remain"); dc_reuse = U("dc_reuse")
    resid_dc = dc_capex * dc_remain * dc_reuse
    people = U("people_data_annual") * 4
    ret_n = U("cap_retain_nation"); ret_s = U("cap_retain_spc")
    resid_nation = salv_eq + resid_dc + people * ret_n
    resid_spc = salv_eq + resid_dc + people * ret_s
    # B. 임차 경로 종속비용
    spend0 = U("spend0"); g_spend = U("g_spend"); foreign = U("foreign_share"); premium = U("premium")
    surplus = U("surplus_mult"); pi_us = U("pi_us"); pi_open = U("pi_open"); open_sub = pi_open * 0.7
    if C.get("pi_shift"): pi_us = np.minimum(0.99, pi_us + C["pi_shift"])     # 국내 호스팅에 따른 접근 안정성 개선(가정)
    T_block = U("T_block"); switch_frac = U("switch_frac")
    haz = 1 - pi_us ** (1 / 4)
    spend = np.array([spend0 * (1 + g_spend) ** i for i in range(YN)])
    M = spend * foreign; rent = M * premium
    block_year = np.full(N, -1); alive = np.ones(N, bool)
    for i in range(YN):
        hit = alive & (rng.uniform(size=N) < haz); block_year[hit] = i; alive &= ~hit
    block_loss = np.zeros(N); switch_cost = np.zeros(N)
    for i in range(YN):
        sel = block_year == i
        yl = M[i] * surplus * (1 - open_sub)
        block_loss[sel] += (yl[sel] * T_block[sel]) / (1 + DISC) ** i
        switch_cost[sel] += (M[i][sel] * switch_frac[sel]) / (1 + DISC) ** i
    rent_pv = pv(rent); M_pv = pv(M)
    lockin = rent_pv + block_loss + switch_cost
    # C. SPC 회피분과 순편익
    capture = U("capture") * C.get("capture_scope", 1.0); va_margin = U("va_margin"); cov_gain = U("cov_gain")   # capture_scope: 전환 대상 범위(B2G 전용이면 공공 몫)
    success = rng.uniform(size=N) < p_success
    ramp = np.array([0, 0, 0.3, 0.6, 1.0, 1.0])
    avoided_rent = pv(M * ramp[:, None] * capture * (premium + va_margin))
    avoided_block = np.zeros(N)
    for i in range(YN):
        sel = (block_year == i) & (i >= 2)
        avoided_block[sel] += (M[i][sel] * surplus[sel] * (1 - open_sub[sel]) * T_block[sel] * cov_gain[sel] * ramp[i]) / (1 + DISC) ** i
        avoided_block[sel] += (M[i][sel] * switch_frac[sel] * capture[sel] * ramp[i]) / (1 + DISC) ** i
    avoided = np.where(success, avoided_rent + avoided_block, 0.0)
    cum_gap = rng.triangular(*cum_gap_tri, N)
    net_cost = cum_gap - resid_nation
    nb = avoided - net_cost
    q = lambda x: [float(np.percentile(x, p)) / 1e12 for p in (10, 50, 90)]
    out = dict(
        residual=dict(equipment_book=q(book_eq), equipment_salvage=q(salv_eq), dc_power=q(resid_dc), capability_nation=q(people * ret_n),
                      total_nation=q(resid_nation), total_spc=q(resid_spc), capex_2026_29=sum(capex_path) / 1e12),
        lockin=dict(gross_outflow=q(M_pv), rent=q(rent_pv), block_loss=q(block_loss), switch=q(switch_cost), total=q(lockin),
                    P_block_2031=float(np.mean(block_year >= 0)), block_loss_if_block=q(block_loss[block_year >= 0]) if (block_year >= 0).any() else None),
        spc=dict(avoided_if_success=q(avoided[success]) if success.any() else None, net_cost=q(net_cost), net_benefit=q(nb),
                 P_nb_pos=float(np.mean(nb > 0)), nb_success=q(nb[success]) if success.any() else None, nb_fail=q(nb[~success]) if (~success).any() else None,
                 E_nb=float(np.mean(nb)) / 1e12, p_success=p_success))
    grid = []
    for ps in ps_grid:
        for cap in cap_grid:
            sc = cap / np.mean(capture)
            grid.append(dict(P_success=ps, capture=cap, E_nb=float(np.mean(ps * (avoided_rent + avoided_block) * sc - net_cost)) / 1e12))
    out["grid2031"] = grid
    pis = []
    for lo, hi in ((0.85, 0.95), (0.7, 0.85), (0.5, 0.7), (0.3, 0.5)):
        mk = (pi_us >= lo) & (pi_us < hi)
        if mk.sum() == 0: continue
        pis.append({"접근 유지 확률": f"{lo}~{hi}", "표본": int(mk.sum()), "종속비용 중앙(조원)": float(np.median(lockin[mk])) / 1e12,
                    "차단 손실 중앙(조원)": float(np.median(block_loss[mk])) / 1e12, "순편익 양일 확률": float(np.mean(nb[mk] > 0))})
    out["pi_sensitivity"] = pis
    # D. 2035 지평
    g_ext = C["ext_growth"]; spc_ext = U("spc_annual_ext")
    M_last = M[-1]; ext_M = [M_last * (1 + g_ext) ** (k + 1) for k in range(4)]
    ext_rent = sum(m_ * premium / (1 + DISC) ** (6 + k) for k, m_ in enumerate(ext_M))
    ext_block = np.zeros(N); ext_avoid = np.zeros(N); alive2 = block_year < 0
    for k, m_ in enumerate(ext_M):
        hit = alive2 & (rng.uniform(size=N) < haz); alive2 &= ~hit
        loss = m_ * surplus * (1 - open_sub) * T_block / (1 + DISC) ** (6 + k)
        ext_block[hit] += loss[hit] + (m_ * switch_frac / (1 + DISC) ** (6 + k))[hit]
        ext_avoid[hit] += (loss * cov_gain + m_ * switch_frac * capture / (1 + DISC) ** (6 + k))[hit]
    ext_avoid_rent = sum(m_ * capture * (premium + va_margin) / (1 + DISC) ** (6 + k) for k, m_ in enumerate(ext_M))
    ext_cost = sum(spc_ext / (1 + DISC) ** (6 + k) for k in range(4))
    av_ext = np.where(success, avoided + ext_avoid_rent + ext_avoid, 0.0)
    nb35 = av_ext - (net_cost + np.where(success, ext_cost, 0.0))
    out["h2035"] = dict(lockin=q(lockin + ext_rent + ext_block), net_benefit=q(nb35), P_nb_pos=float(np.mean(nb35 > 0)),
                        nb_success=q(nb35[success]) if success.any() else None, E_nb=float(np.mean(nb35)) / 1e12, P_block_2035=float(np.mean(~alive2)))
    g35 = []
    for ps in ps_grid:
        for cap in cap_grid35:
            sc = cap / np.mean(capture)
            av = (avoided_rent + avoided_block + ext_avoid_rent + ext_avoid) * sc
            g35.append(dict(P_success=ps, capture=cap, E_nb=float(np.mean(ps * av - (net_cost + ps * ext_cost))) / 1e12))
    out["grid2035"] = g35
    if keep:
        out["arrays"] = dict(nb=nb, nb35=nb35, success=success, net_cost=net_cost, avoided=avoided, lockin=lockin, resid=resid_nation)
    return out


def breakeven_capture(grid, p_success):
    """주어진 성공 확률에서 기대 순편익이 0이 되는 국산 전환율(선형 보간). 격자 범위 밖이면 None."""
    rows = sorted([g for g in grid if abs(g["P_success"] - p_success) < 1e-9], key=lambda g: g["capture"])
    if not rows:
        ps = sorted(set(g["P_success"] for g in grid))
        near = min(ps, key=lambda v: abs(v - p_success)); rows = sorted([g for g in grid if g["P_success"] == near], key=lambda g: g["capture"])
    for a, b in zip(rows, rows[1:]):
        if a["E_nb"] <= 0 <= b["E_nb"]:
            return dict(status="ok", value=a["capture"] + (b["capture"] - a["capture"]) * (-a["E_nb"]) / (b["E_nb"] - a["E_nb"]))
    if rows and rows[0]["E_nb"] > 0: return dict(status="below", value=rows[0]["capture"])
    return dict(status="above", value=rows[-1]["capture"] if rows else None)


def fmt_breakeven(be):
    if be["status"] == "ok": return f"약 {be['value']*100:.0f}%"
    if be["status"] == "below": return f"{be['value']*100:.0f}% 미만"
    return f"{be['value']*100:.0f}% 초과"


def tornado(cum_gap_tri, p_success, capex_path, C=None, N=6000, keys=None, horizon="2031"):
    """각 모수를 분포 하한·상한에 고정했을 때 기대 순편익 변화."""
    C = C or CB_PARAMS
    base = run_cb(cum_gap_tri, p_success, capex_path, C, N=N)
    b0 = base["spc"]["E_nb"] if horizon == "2031" else base["h2035"]["E_nb"]
    keys = keys or ["capture", "premium", "va_margin", "spend0", "g_spend", "foreign_share", "pi_us", "surplus_mult", "resale_disc", "dc_reuse", "cap_retain_nation", "discount"]
    rows = []
    for k in keys:
        v = C[k]
        lo, hi = (v * 0.6, v * 1.4) if not isinstance(v, (tuple, list)) else (v[0], v[1])
        res = []
        for val in (lo, hi):
            C2 = dict(C); C2[k] = (val, val) if isinstance(v, (tuple, list)) else val
            r = run_cb(cum_gap_tri, p_success, capex_path, C2, N=N)
            res.append(r["spc"]["E_nb"] if horizon == "2031" else r["h2035"]["E_nb"])
        rows.append(dict(key=k, label=CB_LABELS.get(k, k), low_val=lo, high_val=hi, E_low=res[0], E_high=res[1], base=b0))
    # 성공 확률
    res = []
    for ps in (max(0.0, p_success - 0.1), min(1.0, p_success + 0.1)):
        r = run_cb(cum_gap_tri, ps, capex_path, C, N=N); res.append(r["spc"]["E_nb"] if horizon == "2031" else r["h2035"]["E_nb"])
    rows.append(dict(key="p_success", label="성공 확률 ±0.1", low_val=p_success - 0.1, high_val=p_success + 0.1, E_low=res[0], E_high=res[1], base=b0))
    rows.sort(key=lambda r: abs(r["E_high"] - r["E_low"]), reverse=True)
    return rows
