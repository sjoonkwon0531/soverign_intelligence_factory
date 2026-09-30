"""예비타당성조사식 종합 평가.

경제성 분석: 사회적 할인율(기본 4.5%)로 비용과 편익을 현재가치화. 비용은 설비투자와 운영비(경제적 비용), 재정 조달분에는
공공자금의 한계비용(Dahlby 2008)을 곱할 수 있음. 매출 중 B2G는 정부와 SPC 사이의 이전이므로 편익에서 제외.
편익은 (1) 회피한 임차 경로 종속비용(비용편익 모듈), (2) 국내 귀속 부가가치 증가와 회피한 차단 손실(거시 모듈), (3) 기간 말 잔존가치.
투입의 생산·부가가치 유발(산업연관 효과)은 파급효과로 따로 보고하고 편익에 더하지 않음(이중 계산 방지).

종합 평가: 경제성·정책성·기술성 기준의 가중치를 사용자가 정하고, 대안별 점수를 정규화해 AHP 형식의 우선순위를 산출.
"""
import numpy as np


def pv(arr, r):
    return float(sum(v / (1 + r) ** i for i, v in enumerate(arr)))


def economic_bc(sim_paths_capex, sim_paths_opex, residual_2031, benefits_pv, r=0.045, gov_share=0.3, mcpf=1.0):
    """경로 중앙값 기준 B/C. sim_paths_*: (N,6) 원. residual_2031: 원. benefits_pv: 원(현재가치)."""
    capex = np.median(sim_paths_capex, 0); opex = np.median(sim_paths_opex, 0)
    cost = pv(capex + opex, r) * (1 + (mcpf - 1) * gov_share)
    ben = benefits_pv + residual_2031 / (1 + r) ** 5
    return dict(cost=cost / 1e12, benefit=ben / 1e12, bc=ben / cost if cost > 0 else np.nan, npv=(ben - cost) / 1e12)


def ahp(scores, weights):
    """scores: {대안: {기준: 원점수(클수록 좋음)}}, weights: {기준: 가중치}. 기준별로 대안 점수를 합 1로 정규화 후 가중합."""
    alts = list(scores.keys()); crit = list(weights.keys())
    wsum = sum(weights.values()); w = {c: weights[c] / wsum for c in crit}
    local = {}
    for c in crit:
        vals = np.array([max(scores[a][c], 0.0) for a in alts], dtype=float)
        s = vals.sum(); local[c] = vals / s if s > 0 else np.full(len(alts), 1 / len(alts))
    total = {a: float(sum(w[c] * local[c][i] for c in crit)) for i, a in enumerate(alts)}
    return dict(weights=w, local={c: dict(zip(alts, map(float, local[c]))) for c in crit}, total=total)
