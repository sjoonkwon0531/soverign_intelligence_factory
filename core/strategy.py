"""전략 분석: 전역 민감도, 시나리오 발견(강건 의사결정), 단계 투자의 실물옵션 가치.

전역 민감도는 몬테카를로 표본에서 결과를 표준화한 모수에 회귀한 표준화 회귀계수와 순위상관을 쓴다(Saltelli et al. 2008).
시나리오 발견은 같은 난수 표본에서 두 대안의 성과 차이(후회)를 모수 공간에서 설명하는 얕은 결정나무로 찾는다
(Lempert, Popper and Bankes 2003의 강건 의사결정 방식을 단순화). 실물옵션 가치는 중간 시점의 합리적 중단 규칙이 있을 때와
없을 때의 기대 성과 차이로 계산한다(Dixit and Pindyck 1994).
"""
import numpy as np
import pandas as pd
from .sim import run_scenario
from .params import PARAM_LABELS, SEED

DRAWN = ["delta", "g_p", "p0_usd", "dc_capex_per_mw_usd", "eps_kw", "power_krw_kwh", "maint", "L", "eta_g", "zeta", "a1", "fr_drift",
         "diff_rate", "gap_open", "A_kr0", "zeta0", "g_zeta", "pi_us", "pi_open", "b2g_real", "market_above_thr_krw", "rent_per_point"]
LAB = dict(PARAM_LABELS, L="계통 접속 리드타임")


def sensitivity(sc, P, N=3000, seed=SEED, outcomes=("success", "lag_f", "cum_gap")):
    r = run_scenario(sc, P, N, seed, keep_paths=False, keep_params=True)
    X = pd.DataFrame(r["params"])[[k for k in DRAWN if k in r["params"][0]]].astype(float)
    Y = pd.DataFrame(r["per_path"])
    keep = [c for c in X.columns if X[c].std() > 0]
    Xs = (X[keep] - X[keep].mean()) / X[keep].std()
    rows = []
    for o in outcomes:
        y = Y[o].astype(float)
        if y.std() == 0 or y.isna().all(): continue
        ys = (y - y.mean()) / y.std()
        b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ys)), Xs.values]), ys.values, rcond=None)
        r2 = 1 - np.sum((ys.values - np.column_stack([np.ones(len(ys)), Xs.values]) @ b) ** 2) / np.sum(ys.values ** 2)
        for j, c in enumerate(keep):
            rows.append(dict(결과=o, 모수=LAB.get(c, c), 키=c, 표준화회귀계수=float(b[j + 1]), 순위상관=float(pd.Series(X[c]).rank().corr(y.rank())), 설명력=float(r2)))
    return pd.DataFrame(rows)


HIGHER_BETTER = {"success": True, "sov": True, "residual": True, "lag_f": False, "cum_gap": False}


def discover(sc_a, sc_b, P, N=3000, seed=SEED, metric="lag_f", max_depth=2, tol=0.0):
    """같은 난수 표본에서 대안 A와 B의 성과를 경로별로 비교하고, A가 B보다 나쁜(후회가 생기는) 모수 영역을 결정나무로 찾는다.

    A가 어떤 경로에서도 나쁘지 않으면(지배) 대신 A가 더 나은 영역을 설명한다. tol은 차이로 인정할 최소 폭.
    """
    ra = run_scenario(sc_a, P, N, seed, keep_paths=False, keep_params=True)
    rb = run_scenario(sc_b, P, N, seed, keep_paths=False, keep_params=True)
    X = pd.DataFrame(ra["params"])[[k for k in DRAWN if k in ra["params"][0]]].astype(float)
    X = X[[c for c in X.columns if X[c].std() > 0]]
    ya = pd.DataFrame(ra["per_path"])[metric].astype(float).fillna(np.inf if not HIGHER_BETTER[metric] else -np.inf)
    yb = pd.DataFrame(rb["per_path"])[metric].astype(float).fillna(np.inf if not HIGHER_BETTER[metric] else -np.inf)
    d = (ya - yb) if HIGHER_BETTER[metric] else (yb - ya)       # 양수: A가 더 나음
    d = d.replace([np.inf, -np.inf], np.nan)
    worse = (d < -tol).values; better = (d > tol).values
    target, label = (worse, "A가 B보다 나쁜 경로") if worse.any() else (better, "A가 B보다 나은 경로")
    rules, imp = "해당 경로 없음(두 대안의 성과가 모든 경로에서 같음)", {}
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        if target.any() and (~target).any():
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(20, N // 50), random_state=0, class_weight="balanced").fit(X.values, target)
            rules = export_text(clf, feature_names=[LAB.get(c, c) for c in X.columns], decimals=2, class_names=["그 밖", "해당"])
            imp = {LAB.get(c, c): float(v) for c, v in zip(X.columns, clf.feature_importances_) if v > 0.01}
    except Exception as e:
        rules = f"결정나무 생략({e})"
    return dict(metric=metric, share_worse=float(worse.mean()), share_better=float(better.mean()), explained=label,
                mean_a=float(np.nanmean(ya.replace([np.inf, -np.inf], np.nan))), mean_b=float(np.nanmean(yb.replace([np.inf, -np.inf], np.nan))),
                mean_diff=float(np.nanmean(d)), regret_mean=float(np.nanmean(np.clip(-d, 0, None))), rules=rules, importance=imp)


def option_value(sc, P, rule=(2028, 3.0), N=3000, seed=SEED):
    """합리적 중단 규칙(해당 연도에 오픈웨이트 문턱 대비 여유보다 뒤지면 트랙 A 중단)의 가치.

    같은 표본에서 규칙이 없을 때와 비교해 누적 재무 갭 절감, 게이트 확률 손실, 오판 중단 비율(중단했으나 계속했다면 통과했을 경로),
    적중 중단 비율(중단했고 계속했어도 미달이었을 경로)을 계산한다. 게이트 손실이 0.1%p 미만이면 교환비는 정의하지 않는다.
    """
    base = run_scenario(sc, P, N, seed, keep_paths=False, keep_params=True)
    alt = run_scenario(dict(sc, stop_rule=tuple(rule)), P, N, seed, keep_paths=False, keep_params=True)
    pb = pd.DataFrame(base["per_path"]); pa = pd.DataFrame(alt["per_path"])
    stopped = pa["stopped"].values > 0.5; succ_b = pb["success"].values > 0.5
    gate_loss = float(pb["success"].mean() - pa["success"].mean()); saved = float(pb["cum_gap"].mean() - pa["cum_gap"].mean())
    return dict(rule=tuple(rule), stop_share=float(stopped.mean()), gap_base=float(pb["cum_gap"].mean()), gap_rule=float(pa["cum_gap"].mean()),
                gap_saved=saved, gate_base=float(pb["success"].mean()), gate_rule=float(pa["success"].mean()), gate_loss=gate_loss,
                false_stop=float((stopped & succ_b).mean()), true_stop=float((stopped & ~succ_b).mean()),
                saved_per_gate_point=(saved / (gate_loss * 100)) if gate_loss > 0.001 else None)


def option_table(sc, P, rules=((2027, 3.0), (2027, 6.0), (2028, 0.0), (2028, 3.0), (2028, 6.0), (2029, 5.0)), N=2000, seed=SEED):
    return [option_value(sc, P, r, N, seed) for r in rules]
