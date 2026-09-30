"""민간 인텔리전스 팩토리(국내 대기업과 프런티어 랩의 공동 AI 데이터센터) 모듈.

정부 출자 없이 민간이 짓는 대규모 AI 데이터센터가 (1) 국내 보유 연산 비중, (2) 국가 할당 용량(sovereign slice)을 통한
자체 역량, (3) 국내 부가가치와 전력 부담, (4) 국내 상업 시장 잠식에 주는 효과를 계산한다. 모든 계수는 범위 가정이다.
"""
import numpy as np
from .params import YEARS

# IT 용량 경로(GW, 해당 연도 말 가동). 실현율을 곱해 쓴다.
FACTORY_PATHS = {
    "F0": dict(name="민간 팩토리 없음", gw=[0, 0, 0, 0, 0, 0]),
    "F1": dict(name="발표 일부 실현(약 2GW, 2031)", gw=[0, 0.1, 0.4, 0.9, 1.5, 2.2]),
    "F2": dict(name="SK 15GW 계획 궤도(약 5GW, 2031)", gw=[0, 0.2, 0.8, 2.0, 3.5, 5.0]),
}

FACTORY_PARAMS = {
    "realization": (0.5, 1.0),        # 계통·부지·앵커 계약 지연을 반영한 실현율(스타게이트 코리아 지연 사례)
    "kw_per_h100e_2025": 0.936,       # IT kW/H100e (Epoch AIDC 허브)
    "eff_gain": 0.19,                 # H100e당 전력의 연 감소율(Epoch ML Hardware −19%/년)
    "slice_frac": (0.02, 0.08),       # 공공 지원(전력·부지·특구)의 대가로 확보하는 국가 할당 용량 비율
    "slice_cost": (0.25, 0.35),       # 할당 용량의 연 이용료(구매가 대비, 원가 수준)
    "Tcap0": (4e4, 8e4), "gT": (1.4, 1.9),   # 국내 팀의 흡수 역량 상한(격자 모형과 같음)
    "spill": (0.0, 0.02),             # 프런티어 랩 국내 거점의 인재·기술 파급(흡수율 연 가산)
    "crowd": (0.10, 0.40),            # 국내 문턱 위 상업 시장 잠식(프런티어 랩 국내 공급 확대)
    "shell_capex_krw_per_gw": (13e12, 19e12),   # 건물·전력·냉각(GPU 제외), 1,350원/달러
    "full_capex_krw_per_gw": 60e12,   # SK텔레콤 추정(GPU 포함)
    "colo_yield": (0.10, 0.14),       # 임대(코로케이션) 수입/누적 건설비
    "va_colo": (0.50, 0.70),          # 임대 수입의 국내 부가가치 비율
    "va_dc": (0.55, 0.75),            # 건설의 국내 부가가치 비율
    "pue": (1.2, 1.35),
    "power_gap_krw_kwh": (0.0, 30.0), # 장기 한계비용과 산업용 요금 차(전력 보조 규모의 근사)
    "world_2025_h100e": 20.93e6, "world_growth": (1.8, 2.5),
    "discount": 0.05,
}


def h100e_per_mw(i, F=FACTORY_PARAMS):
    """연도 인덱스 i(2026=0)의 IT MW당 H100e."""
    return 1000.0 / F["kw_per_h100e_2025"] / (1 - F["eff_gain"]) ** (i + 1)


def draw_factory(rng, path_key, F=None, with_slice=True):
    F = F or FACTORY_PARAMS
    u = lambda k: float(rng.uniform(*F[k]))
    real = u("realization"); gw = np.array(FACTORY_PATHS[path_key]["gw"]) * real
    cap = np.array([gw[i] * 1000 * h100e_per_mw(i, F) for i in range(len(YEARS))])
    sf = u("slice_frac") if (with_slice and path_key != "F0") else 0.0
    x = dict(gw=gw, cap_h100e=cap, slice=cap * sf, slice_frac=sf, slice_cost=u("slice_cost"), slice_train=0.8,
             Tcap0=u("Tcap0"), gT=u("gT"), spill=u("spill") if path_key != "F0" else 0.0,
             crowd=u("crowd") if path_key != "F0" else 0.0, world_growth=u("world_growth"))
    return x


def factory_economics(path_key, N=5000, seed=11, F=None):
    """팩토리 자체의 국내 부가가치, 전력 부담, 국내 보유 연산 비중(정부 재정과 무관)."""
    F = F or FACTORY_PARAMS
    rng = np.random.default_rng(seed)
    beta = 1 / (1 + F["discount"]); disc = np.array([beta ** i for i in range(len(YEARS))])
    out = {k: [] for k in ("gw31", "h100e29", "share29", "capex_shell", "capex_full", "gva", "power_twh31", "power_subsidy")}
    for _ in range(N):
        u = lambda k: float(rng.uniform(*F[k]))
        real = u("realization"); gw = np.array(FACTORY_PATHS[path_key]["gw"]) * real
        add = np.diff(np.concatenate([[0], gw]))
        shell = add * u("shell_capex_krw_per_gw"); full = add * F["full_capex_krw_per_gw"]
        cum_shell = np.cumsum(shell)
        colo = cum_shell * u("colo_yield")
        gva = float(np.sum(disc * (shell * u("va_dc") + colo * u("va_colo"))))
        pue = u("pue"); twh = gw * pue * 8.76
        subsidy = float(np.sum(disc * twh * 1e9 * u("power_gap_krw_kwh")))
        world29 = F["world_2025_h100e"] * u("world_growth") ** 4
        h29 = gw[3] * 1000 * h100e_per_mw(3, F)
        for k, v in (("gw31", gw[-1]), ("h100e29", h29), ("share29", h29 / world29), ("capex_shell", shell.sum()), ("capex_full", full.sum()),
                     ("gva", gva), ("power_twh31", twh[-1]), ("power_subsidy", subsidy)):
            out[k].append(v)
    q = lambda v: [float(np.percentile(v, p)) for p in (10, 50, 90)]
    return {k: q(v) for k, v in out.items()}
