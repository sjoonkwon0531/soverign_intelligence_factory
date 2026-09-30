"""모수와 시나리오 정의 (기본값 = 본보고서 초안 r1, 2026-09-30 재보정).

모든 확률 모수는 (하한, 상한) 균등분포로 표기한다. 앱의 '모수 설정'에서 바꾸면 이 사전의 사본이 쓰인다.
"""
from copy import deepcopy

YEARS = list(range(2026, 2032))
FLOP_PER_H100E_YEAR = 4.0e14 * 3.15e7   # 약 1e15 FLOP/s 피크, MFU 40%, 1년
K0 = 1.1e25                              # 한국 2026 누적 학습 연산량(주요 모델 5개 합)
SEED = 20260930

# (하한, 상한) = 균등분포, 단일 값 = 고정
DEFAULT_PARAMS = {
    "krw_per_usd": 1350.0,
    "delta": (0.25, 0.33),            # 장비 물리 감가/년
    "g_p": (0.15, 0.25),              # 성능당 가격 하락/년
    "p0_usd": (15e3, 25e3),           # 2026 H100e당 서버 비용(USD)
    "dc_capex_per_mw_usd": (8e6, 14e6),
    "eps_kw": (1.2, 1.5),             # 운영비 계산용 all-in kW/H100e
    "eps_it": 0.936,                  # IT kW/H100e (Epoch AIDC 허브)
    "power_krw_kwh": (120, 160),
    "maint": (0.04, 0.07),
    "L_choices": ([2, 3, 4], [0.30, 0.45, 0.25]),   # 계통 접속 리드타임(년)과 확률
    "eta_g": (1.8, 3.0),              # 알고리즘 효율 배수/년
    "zeta": (0.5, 1.0),               # 효율의 국내 확산계수
    "a1": (2.5, 4.5),                 # 연산량 10배당 ECI
    "fr_drift": (14.5, 17.0),         # 프런티어 상승 ECI/년
    "diff_rate": (11.7, 13.6),        # 글로벌 진보 시간항 ECI/년
    "Afor0": 166.3,                   # 기준일 프런티어 ECI
    "gap_open": (6.0, 11.0),          # 폐쇄형-오픈웨이트 간격
    "A_kr0": (146.0, 153.0),          # 한국 출발 ECI
    "zeta0": (0.55, 0.85),            # 흡수율 초기값
    "g_zeta": (0.03, 0.10),           # 흡수율 연 개선
    "pi_us": (0.5, 0.9),              # 미국 접근 유지 확률(2029 누적)
    "pi_open": (0.6, 0.95),           # 오픈웨이트 공급 지속 확률
    "b2g_real": (0.6, 1.0),           # B2G 실현율
    "market_above_thr_krw": (2.0e12, 5.0e12),
    "rent_per_point": (0.05, 0.12),
    # 게이트·흡수 가산(구조 가정)
    "bonus_consolidation": 0.05,      # 결집 시 흡수율 개선 가산
    "bonus_verification": 0.03,       # 검증 루프 가산
    "gate_2027_margin": 3.0, "gate_2029_domain_margin": 5.0, "gate_2029_frontier_margin": 5.0,
    "verif_revenue_krw": 0.15e12,
}

PARAM_LABELS = {
    "delta": "장비 물리 감가(연)", "g_p": "성능당 가격 하락(연)", "p0_usd": "H100e당 서버 비용(달러)", "dc_capex_per_mw_usd": "데이터센터 건설비(달러/MW)",
    "eps_kw": "운영 전력 원단위(kW/H100e)", "power_krw_kwh": "전기요금(원/kWh)", "maint": "유지보수비(설비투자 대비)",
    "eta_g": "알고리즘 효율 배수(연)", "zeta": "효율 확산계수", "a1": "연산량 10배당 역량지수", "fr_drift": "프런티어 상승(역량지수/년)",
    "diff_rate": "글로벌 진보 시간항(역량지수/년)", "gap_open": "폐쇄형과 오픈웨이트 간격", "A_kr0": "한국 출발 역량지수", "zeta0": "흡수율 초기값",
    "g_zeta": "흡수율 연 개선", "pi_us": "미국 접근 유지 확률", "pi_open": "오픈웨이트 공급 지속 확률", "b2g_real": "B2G 실현율",
    "market_above_thr_krw": "문턱 위 시장(원/년)", "rent_per_point": "초과 1점당 시장 점유",
}

_E = [50, 80, 110, 140, 170, 200]
SCENARIOS = {
    "S1": dict(name="분산 지속", desc="부처별 분산, 3팀 균등, 컴퓨팅 임차 중심. 연 1.5조원 6년(권고안과 같은 총투입)",
               capex_krw=[1.5e12] * 6, n_teams=3, train_share=0.30, B2G=[0.3e12, 0.4e12, 0.5e12, 0.6e12, 0.7e12, 0.8e12],
               E_pipeline_mw=_E, fast_track=False, own_frontier=True, verif=False, gate_stop=False, open_fallback=False),
    "S1B": dict(name="분산 지속(예산 수준)", desc="S1 구조에 실제 예산 수준 적용: 2026 2.61조, 2027 이후 연 5.59조",
               capex_krw=[2.61e12] + [5.59e12] * 5, n_teams=3, train_share=0.30, B2G=[0.3e12, 0.4e12, 0.5e12, 0.6e12, 0.7e12, 0.8e12],
               E_pipeline_mw=_E, fast_track=False, own_frontier=True, verif=False, gate_stop=False, open_fallback=False),
    "S2": dict(name="SPC 자동 중단", desc="10조/5년 단일 주체, 패스트트랙, 앵커 구매. 게이트 미달 시 자동 중단",
               capex_krw=[2.5e12, 2.5e12, 2.0e12, 1.5e12, 1.5e12, 0.0], n_teams=1, train_share=0.50, B2G=[0.5e12, 1.0e12, 1.5e12, 1.8e12, 2.0e12, 2.0e12],
               E_pipeline_mw=_E, fast_track=True, own_frontier=True, verif=True, gate_stop=True, open_fallback=False),
    "S2b": dict(name="SPC 에스크로", desc="S2와 같으나 미달 시 중단 없이 재편",
               capex_krw=[2.5e12, 2.5e12, 2.0e12, 1.5e12, 1.5e12, 0.0], n_teams=1, train_share=0.50, B2G=[0.5e12, 1.0e12, 1.5e12, 1.8e12, 2.0e12, 2.0e12],
               E_pipeline_mw=_E, fast_track=True, own_frontier=True, verif=True, gate_stop=False, open_fallback=False),
    "S3": dict(name="오픈웨이트 배치와 검증", desc="자체 프런티어 없음. 오픈웨이트 격리 배치와 국가 검증 인프라, 4조원",
               capex_krw=[0.8e12, 0.8e12, 0.6e12, 0.6e12, 0.6e12, 0.6e12], n_teams=0, train_share=0.0, B2G=[0.5e12, 0.8e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12],
               E_pipeline_mw=_E, fast_track=False, own_frontier=False, verif=True, gate_stop=False, open_fallback=False),
    "S4": dict(name="이중 트랙 자동 중단", desc="트랙 A 자체 추격과 트랙 B 오픈웨이트 배치. 미달 시 트랙 A 중단",
               capex_krw=[2.2e12, 2.2e12, 1.8e12, 1.4e12, 1.4e12, 0.0], n_teams=1, train_share=0.45, B2G=[0.5e12, 1.0e12, 1.5e12, 1.8e12, 2.0e12, 2.0e12],
               E_pipeline_mw=_E, fast_track=True, own_frontier=True, verif=True, gate_stop=True, open_fallback=True),
    "S4b": dict(name="이중 트랙과 에스크로(권고안)", desc="S4와 같으나 미달 시 중단 없이 재편",
               capex_krw=[2.2e12, 2.2e12, 1.8e12, 1.4e12, 1.4e12, 0.0], n_teams=1, train_share=0.45, B2G=[0.5e12, 1.0e12, 1.5e12, 1.8e12, 2.0e12, 2.0e12],
               E_pipeline_mw=_E, fast_track=True, own_frontier=True, verif=True, gate_stop=False, open_fallback=True),
}

# 격자 모형 추가 모수
GRID_EXTRA = {
    "Cp": (2.0e12, 5.0e12), "Cg": (3.0e12, 6.0e12), "Tcap0": (4e4, 8e4), "gT": (1.4, 1.9),
    "Ecap_mw": (1500, 3000), "Gcap": (1.5e5, 4.0e5), "lease_rate": (0.38, 0.50),
}
GRID_SHAPE = [0.25, 0.25, 0.20, 0.15, 0.15, 0.0]

# 격차 환산용 상승 속도(시뮬레이션 결과를 년으로 바꿀 때)
LAG_DIVISOR = {"frontier": 15.76, "open": 14.21}


def default_params():
    return deepcopy(DEFAULT_PARAMS)


def default_scenarios():
    return deepcopy(SCENARIOS)
