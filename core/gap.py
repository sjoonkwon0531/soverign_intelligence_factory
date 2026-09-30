"""격차 다층 구조 산출과 시뮬레이션 모수 자동 보정.

입력: ECI, 주목 모델, AA–ECI 대응표, 한국 모델 AA 점수, 컴퓨팅 보유 요약. 산출식은 본보고서 부록 G와 같다.
"""
import numpy as np
import pandas as pd


def yr(d):
    return d.dt.year + (d.dt.dayofyear - 1) / 365.25


def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X)); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; n, k = X.shape
    s2 = res @ res / max(n - k, 1); cov = s2 * np.linalg.pinv(X.T @ X)
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())) if n > 1 else float("nan")
    return b, np.sqrt(np.abs(np.diag(cov))), r2, n


def envelope(df):
    """시점별 기록 경신 모델만 추출(원자료 순서 기준 날짜 정렬, 본보고서와 같은 처리)."""
    df = df.sort_values("date"); best = -1e9; rows = []
    for _, r in df.iterrows():
        if r["eci"] > best:
            best = r["eci"]; rows.append(r)
    return pd.DataFrame(rows)


def analyze(data, asof=None, window="2024-01-01", price_decline=(21.6, 40.0), kr_growth=(1.36, 2.0)):
    eci = data["eci"].copy()
    asof = pd.Timestamp(asof) if asof else eci["date"].max()
    eci = eci[eci["date"] <= asof].copy(); eci["t"] = yr(eci["date"])
    W0 = pd.Timestamp(window)
    T = []
    def tr(layer, item, val, unit="", how=""):
        T.append(dict(층위=layer, 항목=item, 값=val, 단위=unit, 산출=how)); return val
    R = dict(asof=str(asof.date()), window=window)
    closed = eci[eci["Accessibility group"] == "Closed weights"]; opn = eci[eci["Accessibility group"] == "Open weights"]
    env_c = envelope(closed); env_o = envelope(opn)
    R["env_closed"] = env_c; R["env_open"] = env_o
    ec = env_c[env_c["date"] >= W0]; eo = env_o[env_o["date"] >= W0]
    bc, sc, r2c, nc = ols([ec["t"].values], ec["eci"].values)
    bo, so, r2o, no = ols([eo["t"].values], eo["eci"].values)
    bq, sq, _, _ = ols([ec["t"].values - 2025, (ec["t"].values - 2025) ** 2], ec["eci"].values)
    R.update(g_fr=float(bc[1]), g_fr_se=float(sc[1]), g_fr_n=int(nc), g_fr_r2=float(r2c), g_fr_a=float(bc[0]),
             g_op=float(bo[1]), g_op_se=float(so[1]), g_op_n=int(no), g_op_a=float(bo[0]), quad=float(bq[2]), quad_t=float(abs(bq[2] / sq[2])) if sq[2] > 0 else float("nan"))
    tr("표면 격차", "폐쇄형 프런티어 상승 속도", round(R["g_fr"], 2), "ECI/년", f"{window[:7]} 이후 기록 경신 {nc}개 OLS, 표준오차 {sc[1]:.2f}, 결정계수 {r2c:.2f}")
    tr("표면 격차", "오픈웨이트 최상위 상승 속도", round(R["g_op"], 2), "ECI/년", f"기록 경신 {no}개 OLS, 표준오차 {so[1]:.2f}")
    tr("표면 격차", "2차항(가속 검정)", round(R["quad"], 2), "ECI/년²", f"|t|={R['quad_t']:.2f} (2 미만이면 가속 증거 없음)")
    R["fr_now"] = float(closed["eci"].max()); R["op_now"] = float(opn["eci"].max())
    R["fr_model"] = str(closed.loc[closed["eci"].idxmax(), "Model"]); R["op_model"] = str(opn.loc[opn["eci"].idxmax(), "Model"])
    tr("표면 격차", "프런티어 최고점", R["fr_now"], "ECI", R["fr_model"]); tr("표면 격차", "오픈웨이트 최고점", R["op_now"], "ECI", R["op_model"])
    # 기간 민감도
    sens = []
    for grp, envg, lab in (("Closed weights", env_c, "폐쇄형"), ("Open weights", env_o, "오픈웨이트")):
        for w in ("2023-01-01", "2024-01-01", "2025-01-01"):
            x = envg[envg["date"] >= w]
            if len(x) >= 3:
                b_, s_, _, n_ = ols([x["t"].values], x["eci"].values); sens.append(dict(구분=lab, 기간=f"{w[:7]} 이후", 기울기=b_[1], 표준오차=s_[1], 기록수=n_))
        now = eci[eci["Accessibility group"] == grp]["eci"].max()
        for mo in (12, 24, 36):
            prev = eci[(eci["Accessibility group"] == grp) & (eci["date"] <= asof - pd.DateOffset(months=mo))]["eci"].max()
            if pd.notna(prev): sens.append(dict(구분=lab, 기간=f"최근 {mo}개월 평균", 기울기=(now - prev) / (mo / 12), 표준오차=np.nan, 기록수=np.nan))
    R["slope_sens"] = pd.DataFrame(sens)
    # 진보 생산 주체
    env_all = envelope(eci); env_all = env_all[env_all["date"] >= W0]
    R["records_all"] = env_all["Country (of organization)"].fillna("미상").value_counts().to_dict()
    R["records_open"] = eo["Country (of organization)"].fillna("미상").value_counts().to_dict()
    R["kr_in_eci"] = int(eci["Country (of organization)"].fillna("").str.contains("Korea").sum())
    tr("진보 생산 주체", "프런티어 기록 경신 국적", str(R["records_all"]), "개", f"{window[:7]} 이후")
    tr("진보 생산 주체", "오픈웨이트 기록 경신 국적", str(R["records_open"]), "개", f"{window[:7]} 이후")
    tr("진보 생산 주체", "ECI 수록 한국 모델 수", R["kr_in_eci"], "개")
    # AA 환산
    pairs = data["aa_pairs"]
    bm, sm_, r2m, nm = ols([pairs["aa"].values], pairs["eci"].values)
    res_sd = float(np.std(pairs["eci"] - (bm[0] + bm[1] * pairs["aa"]), ddof=2))
    R.update(map_a=float(bm[0]), map_b=float(bm[1]), map_r2=float(r2m), map_n=int(nm), map_sd=res_sd, map_pi=1.96 * res_sd)
    tr("표면 격차", "AA→ECI 환산식", f"ECI = {bm[1]:.3f}·AA + {bm[0]:.2f}", "", f"n={nm}, 결정계수 {r2m:.3f}, 95% 예측구간 ±{1.96*res_sd:.1f}")
    kr = data["korea"].copy()
    kr["eci"] = bm[0] + bm[1] * kr["aa"]; kr["eci_low"] = bm[0] + bm[1] * kr["aa_low"]
    kr["프런티어 격차(년)"] = (R["fr_now"] - kr["eci"]) / R["g_fr"]; kr["프런티어 격차 상한(년)"] = (R["fr_now"] - kr["eci_low"]) / R["g_fr"]
    kr["오픈웨이트 격차(년)"] = (R["op_now"] - kr["eci"]) / R["g_op"]
    R["korea"] = kr
    best = kr.loc[kr["eci"].idxmax()]
    R.update(kr_best=str(best["model"]), kr_best_eci=float(best["eci"]), lag_f=float((R["fr_now"] - best["eci"]) / R["g_fr"]),
             lag_f_hi=float((R["fr_now"] - best["eci_low"]) / R["g_fr"]), lag_o=float((R["op_now"] - best["eci"]) / R["g_op"]),
             lag_band=float(1.96 * res_sd / R["g_fr"]))
    tr("표면 격차", "한국 최상위 모델 프런티어 격차", f"{R['lag_f']:.2f}~{R['lag_f_hi']:.2f}", "년", f"{R['kr_best']}, 환산 오차 ±{R['lag_band']:.2f}년")
    tr("표면 격차", "한국 최상위 모델 오픈웨이트 격차", round(R["lag_o"], 2), "년")
    fin = kr[kr["finalist"]]
    if len(fin):
        R["fin_lag"] = (float((R["fr_now"] - fin["eci"].max()) / R["g_fr"]), float((R["fr_now"] - fin["eci"].min()) / R["g_fr"]))
        tr("정부 사업 내부 격차", "결선 팀 모델 프런티어 격차", f"{R['fin_lag'][0]:.2f}~{R['fin_lag'][1]:.2f}", "년")
    # 컴퓨팅 환산
    nt = data["notable"]
    j = eci.merge(nt[["Model", "flop"]], on="Model", how="inner").dropna(subset=["flop"]); j = j[j["flop"] > 0]
    jw = j[j["date"] >= W0]
    if len(jw) >= 8:
        bf, sf, r2f, nf = ols([np.log10(jw["flop"].values), jw["t"].values], jw["eci"].values)
        R.update(a1=float(bf[1]), a1_se=float(sf[1]), c_time=float(bf[2]), c_time_se=float(sf[2]), flop_n=int(nf), flop_r2=float(r2f))
        R["ten_x_years"] = R["a1"] / R["g_fr"]; R["one_year_decades"] = R["g_fr"] / R["a1"]
        tr("컴퓨팅 환산", "연산량 10배당 ECI", round(R["a1"], 2), "ECI", f"n={nf}, 표준오차 {sf[1]:.2f}, 결정계수 {r2f:.2f}")
        tr("컴퓨팅 환산", "시간항(연산량 통제 후)", round(R["c_time"], 2), "ECI/년", f"표준오차 {sf[2]:.2f}")
        tr("컴퓨팅 환산", "연산 10배의 효과", round(R["ten_x_years"], 2), "년")
        tr("컴퓨팅 환산", "1년치 진보의 연산 환산", f"10^{R['one_year_decades']:.1f} ≈ {10**R['one_year_decades']:,.0f}배", "")
        cs = []
        for lab, x in (("2023. 2. 이후 전체", j), ("2025. 1. 이후", j[j["date"] >= "2025-01-01"]), ("오픈웨이트만", j[j["Accessibility group"] == "Open weights"]), (f"{window[:7]} 이후(기준)", jw)):
            if len(x) >= 8:
                b_, s_, _, n_ = ols([np.log10(x["flop"].values), x["t"].values], x["eci"].values)
                cs.append(dict(표본=lab, 모델수=n_, 연산계수=b_[1], 표준오차=s_[1], 시간항=b_[2], 환산배수=f"10^{R['g_fr']/b_[1]:.1f}"))
        R["compute_sens"] = pd.DataFrame(cs)
    # 생산 능력
    S = data.get("stock") or {}
    if S:
        k = S["KR_position_2025"]; g_w = S["global_stock_growth"]["2025"]
        R.update(kr_share=k["share_of_global_pct"], us_over_kr=k["US_owned_over_KR"], cluster_gap=S["single_cluster_gap_US_over_KR"],
                 kr_flop_sum=S["KR_2026_sum_flop_log10"], world_growth=g_w,
                 input_gap_growth=(g_w / kr_growth[1], g_w / kr_growth[0]))
        tr("생산 능력", "AI 칩 보유 점유율", R["kr_share"], "%"); tr("생산 능력", "미국 소유 대비", f"1/{R['us_over_kr']}")
        tr("생산 능력", "최대 단일 클러스터 대비", f"1/{R['cluster_gap']}")
        tr("생산 능력", "투입 격차 확대 배수", f"{R['input_gap_growth'][0]:.2f}~{R['input_gap_growth'][1]:.2f}", "배/년", f"세계 증가율 {g_w} ÷ 한국 증가율 {kr_growth}")
    # 가격 환산
    lo, hi = price_decline
    R["price_x"] = (lo ** R["lag_f"], hi ** R["lag_f_hi"])
    tr("가격 환산", "표면 격차의 가격 환산", f"{R['price_x'][0]:.0f}~{R['price_x'][1]:.0f}", "배", f"연간 가격 하락 {lo}~{hi}배를 격차 연수만큼 거듭제곱")
    R["trace"] = pd.DataFrame(T)
    return R


def suggest_params(R, P):
    """격차 산출 결과로 시뮬레이션 모수를 보정한 사본을 돌려준다. 규칙은 본보고서 r1 보정과 같다."""
    Q = dict(P)
    Q["fr_drift"] = (round(R["g_fr"] - 1.25, 2), round(R["g_fr"] + 1.25, 2))
    if "c_time" in R:
        Q["diff_rate"] = (round(R["c_time"] - R["c_time_se"], 2), round(R["c_time"] + R["c_time_se"], 2))
    if "compute_sens" in R and len(R["compute_sens"]):
        v = R["compute_sens"]["연산계수"]; Q["a1"] = (round(max(1.5, float(v.min())) - 0.2, 1), round(float(v.max()) - 0.2, 1))
    kr = R["korea"]; fin = kr[kr["finalist"]]
    lo = float(fin["eci"].min()) if len(fin) else float(kr["eci"].min())
    Q["A_kr0"] = (round(lo, 0), round(float(kr["eci"].max()), 0))
    Q["Afor0"] = round(R["fr_now"], 1)
    gap_now = R["fr_now"] - R["op_now"]
    Q["gap_open"] = (round(max(3.0, gap_now - 2.7), 0), round(gap_now + 2.3, 0))
    return Q


def forecast(R, years=range(2026, 2032), band=None):
    """프런티어와 오픈웨이트 최고점의 선형 전망(기준일 이후). band: 기울기 (하한, 상한)."""
    t0 = pd.Timestamp(R["asof"]); t0 = t0.year + (t0.dayofyear - 1) / 365.25
    sens = R["slope_sens"]
    if band is None:
        v = sens[sens["구분"] == "폐쇄형"]["기울기"]; band = (float(v.min()), float(v.max()))
    rows = []
    for y in years:
        dt = (y + 0.75) - t0   # 각 연도 3분기 말 기준
        rows.append(dict(연도=y, 프런티어=R["fr_now"] + R["g_fr"] * dt, 프런티어_하한=R["fr_now"] + band[0] * dt, 프런티어_상한=R["fr_now"] + band[1] * dt,
                         오픈웨이트=R["op_now"] + R["g_op"] * dt))
    return pd.DataFrame(rows)
