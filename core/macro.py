"""국가 거시경제 파급 모듈.

이론 틀
  (1) 과업 기반 성장회계(Hulten 1978; Acemoglu 2025): AI의 GDP 효과는 AI가 수행하는 과업의 비중과 과업 단위 비용 절감의 곱으로 근사.
      한국 보정은 한국은행(2025) 추정치(장기 GDP +4.2~12.6%)를 장기 수준 효과로 두고, 도입은 로지스틱 확산 경로를 따른다고 가정.
  (2) 범용기술과 무형 보완투자(Bresnahan and Trajtenberg 1995; Brynjolfsson, Rock and Syverson 2021): 초기에는 보완투자가
      측정 성과를 가리는 J-곡선이 나타남. 도입 속도에 비례한 보완투자 비용으로 반영.
  (3) GDP와 GNI의 구분: AI 효과 자체는 공급원과 무관하게 발생하나(대부분 GDP), AI 서비스 지출 중 해외 공급분은 수입이며
      그 부가가치는 국외로 귀속됨. 국내 공급 비중(전환율)이 국내 귀속 부가가치와 서비스수지를 결정.
  (4) 접근 차단 충격: 프런티어급 모델에 의존하는 AI 효과(f_front)는 차단 기간 동안 상실되며, 주권 커버리지만큼 완화됨.

모든 계수는 범위 가정이며 결론은 시나리오 간 차이로만 해석한다. GDP 기준값은 사용자가 최신 국민계정으로 바꾼다.
"""
import numpy as np

HORIZON = list(range(2026, 2036))

MACRO_PARAMS = {
    "gdp0_krw": 2600e12,           # 2025 명목 GDP(원). 한국은행 국민계정 확정치로 교체할 것
    "g_nominal": 0.035,
    "G_LR": (0.042, 0.126),        # AI의 장기 GDP 수준 효과(한국은행 2025, 성공적 도입 시)
    "adopt_mid": (2029.0, 2033.0), # 확산 경로 중간 시점
    "adopt_k": (0.5, 1.0),         # 확산 속도
    "jcurve": (0.0, 0.3),          # 보완 무형투자 비용(도입 증가분 대비 GDP 효과의 비율)
    "f_front": (0.15, 0.40),       # 프런티어급 모델이 필요한 AI 효과의 비중(문턱 위)
    "spend_ratio": (0.15, 0.35),   # AI 서비스 지출 / AI의 GDP 효과
    "dom_share0": (0.10, 0.25),    # 현재 AI 서비스의 국내 공급 비중
    "capture": (0.20, 0.40),       # SPC 성공 시 문턱 위 지출의 국산 전환율
    "trackB_gain": (0.05, 0.15),   # 오픈웨이트 배치 트랙의 문턱 아래 국내 공급 증가
    "v_dom": (0.60, 0.85),         # 국내 공급 AI 서비스의 국내 부가가치 비율
    "v_loc": (0.10, 0.25),         # 해외 공급 AI 서비스의 국내 부가가치 비율(통합·유통)
    "pi_us": (0.5, 0.9), "T_block": (0.5, 2.0),
    "discount": 0.045,             # 사회적 할인율(기획재정부 예타 지침 4.5%, 2017. 9. 이후)
}


def logistic(t, mid, k):
    return 1.0 / (1.0 + np.exp(-k * (t - mid)))


def run_macro(scen_stats, M=None, N=5000, seed=31):
    """scen_stats: {키: dict(p_success, coverage, trackB(bool))}. 같은 난수로 모든 시나리오를 평가해 차이를 줄인다.

    반환: 시나리오별 현재가치(조원) 분포 요약과 연도별 경로(중앙값).
    """
    M = M or MACRO_PARAMS
    rng = np.random.default_rng(seed)
    T = len(HORIZON); t = np.array(HORIZON, dtype=float)
    u = lambda k: rng.uniform(*M[k], N) if isinstance(M[k], (tuple, list)) else np.full(N, float(M[k]))
    G = u("G_LR"); mid = u("adopt_mid"); k = u("adopt_k"); J = u("jcurve"); ff = u("f_front"); sr = u("spend_ratio")
    d0 = u("dom_share0"); cap = u("capture"); tb = u("trackB_gain"); vd = u("v_dom"); vl = u("v_loc")
    pi = u("pi_us"); Tb = u("T_block")
    haz = 1 - pi ** (1 / 4)
    block_year = np.full(N, -1); alive = np.ones(N, bool)
    for i in range(T):
        hit = alive & (rng.uniform(size=N) < haz); block_year[hit] = i; alive &= ~hit
    succ_u = rng.uniform(size=N); cov_u = rng.uniform(size=N)
    gdp = M["gdp0_krw"] * (1 + M["g_nominal"]) ** (t - 2025)                 # (T,)
    adopt = logistic(t[None, :], mid[:, None], k[:, None])                  # (N,T)
    dadopt = np.diff(np.concatenate([logistic(2025.0, mid, k)[:, None], adopt], axis=1), axis=1)
    gain = gdp[None, :] * G[:, None] * adopt - gdp[None, :] * G[:, None] * J[:, None] * dadopt   # 측정 GDP 효과(J-곡선 반영)
    X = sr[:, None] * gdp[None, :] * G[:, None] * adopt                     # AI 서비스 지출
    disc = np.array([(1 + M["discount"]) ** -(y - 2026) for y in HORIZON])
    ramp = np.clip((t - 2027) / 3, 0, 1)                                    # 2028년부터 3년에 걸쳐 전환
    out = {}
    for key, st in scen_stats.items():
        succ = succ_u < st["p_success"]
        dshare = d0[:, None] + (succ[:, None] * cap[:, None] * ff[:, None] * ramp[None, :]) + (tb[:, None] * (1 - ff[:, None]) * ramp[None, :] * (1.0 if st.get("trackB") else 0.0))
        dshare = np.clip(dshare, 0, 1)
        retained = X * (dshare * vd[:, None] + (1 - dshare) * vl[:, None])  # 국내 귀속 부가가치
        imports = X * (1 - dshare)
        covered = cov_u < st["coverage"]
        loss = np.zeros((N, T))
        for i in range(T):
            sel = block_year == i
            loss[sel, i] = gain[sel, i] * ff[sel] * np.where(covered[sel], 0.3, 1.0) * np.minimum(Tb[sel], 1.0)
            if i + 1 < T:
                extra = np.clip(Tb[sel] - 1.0, 0, 1)
                loss[sel, i + 1] += gain[sel, i + 1] * ff[sel] * np.where(covered[sel], 0.3, 1.0) * extra
        pv = lambda A: (A * disc[None, :]).sum(1) / 1e12
        q = lambda x: [float(np.percentile(x, p)) for p in (10, 50, 90)]
        out[key] = dict(gdp_gain=q(pv(gain)), retained=q(pv(retained)), imports=q(pv(imports)), block_loss=q(pv(loss)),
                        block_loss_mean=float(pv(loss).mean()), block_loss_p95=float(np.percentile(pv(loss), 95)),
                        retained_mean=float(pv(retained).mean()), imports_mean=float(pv(imports).mean()),
                        path_gain=np.median(gain, 0) / 1e12, path_retained=np.median(retained, 0) / 1e12, path_dshare=np.median(dshare, 0),
                        _retained=pv(retained), _loss=pv(loss), _imports=pv(imports))
    return out


def compare(macro_out, base_key):
    """기준 대비 차이(같은 난수 표본의 경로별 차이): 국내 귀속 부가가치, 서비스 수입, 차단 손실."""
    b = macro_out[base_key]; rows = []
    for k, v in macro_out.items():
        dr = v["_retained"] - b["_retained"]; dl = b["_loss"] - v["_loss"]; di = b["_imports"] - v["_imports"]
        q = lambda x: [float(np.percentile(x, p)) for p in (10, 50, 90)]
        rows.append({"시나리오": k, "국내 귀속 부가가치 증가(조원)": q(dr), "회피한 차단 손실(조원)": q(dl), "서비스 수입 감소(조원)": q(di),
                     "주권 배당 합계 평균(조원)": float((dr + dl).mean()), "차단 손실 95% 꼬리 감소(조원)": float(np.percentile(b["_loss"], 95) - np.percentile(v["_loss"], 95))})
    return rows
