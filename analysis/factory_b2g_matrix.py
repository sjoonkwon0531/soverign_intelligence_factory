"""민간 인텔리전스 팩토리 × 소버린 AI 추진 방식(B2G 전용 포함) 시나리오 행렬.

실행: python -m analysis.factory_b2g_matrix  (저장소 최상위에서)
모든 셀에 흡수 역량 상한(격자 모형과 같은 가정)을 적용하므로 본보고서 기본 시나리오 값과 다를 수 있다.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.params import default_params, default_scenarios, SEED
from core.sim import run_scenario
from core.factory import draw_factory, factory_economics, FACTORY_PATHS
from core.costbenefit import run_cb, default_cb, breakeven_capture, fmt_breakeven

P = default_params(); S = default_scenarios()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
FLOOR = 0.4   # 품질 연동 조달에서 성능과 무관하게 국산으로 가는 보호 물량 비율(가정)
VAR = {
    "A 분산 지속": dict(S["S1"]),
    "B SPC B2G 전용(조달 보장)": dict(S["S4b"], market_mult=0.0),
    "C SPC B2G 전용(품질 연동)": dict(S["S4b"], market_mult=0.0, b2g_quality_floor=FLOOR),
    "D SPC B2G 전용(품질 연동, 실현 60%)": dict(S["S4b"], market_mult=0.0, b2g_quality_floor=FLOOR, b2g_mult=0.6),
    "E 권고안(B2G와 민간·수출, 품질 연동)": dict(S["S4b"], b2g_quality_floor=FLOOR),
    "F 권고안 경량(설비 절반)과 할당 용량": dict(S["S4b"], b2g_quality_floor=FLOOR, capex_krw=[v * 0.5 for v in S["S4b"]["capex_krw"]]),
}
SCOPE = {k: 0.22 for k in VAR if "B2G 전용" in k}   # 문턱 위 지출 중 공공 몫(가정 15~30%의 중앙)
PI_SHIFT = {"F0": 0.0, "F1": 0.03, "F2": 0.06}                                  # 국내 호스팅의 접근 안정성 효과(가정, 수출통제 위험은 불변)
rows = []
for fk in FACTORY_PATHS:
    for vk, sc in VAR.items():
        r = run_scenario(sc, P, N, SEED, keep_paths=False, xdraw=lambda rng, fk=fk: draw_factory(rng, fk), xseed=7)
        row = {"민간 팩토리": f"{fk} {FACTORY_PATHS[fk]['name']}", "추진 방식": vk, "2029 도메인 게이트": r["G2029_dom"],
               "2029 프런티어 격차(년)": r["lag_front"][1], "격차 1년 이하 확률": r["P_lagF_le1"], "누적 재무 갭(조원)": r["gap"][1],
               "갭 10%": r["gap"][0], "갭 90%": r["gap"][2], "주권 커버리지": r["sov"], "공공 총투입(조원)": r["capex_total"], "2028 게이트": r["G2028"]}
        if not vk.startswith("A"):
            C = default_cb(); C["capture_scope"] = SCOPE.get(vk, 1.0); C["pi_shift"] = PI_SHIFT[fk]
            cb = run_cb(tuple(v * 1e12 for v in r["gap"]), r["G2029_dom"], sc["capex_krw"][:4], C, N=10000,
                        ps_grid=(round(r["G2029_dom"], 3), 0.3, 0.4, 0.5), cap_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8))
            row.update({"국가 순편익 2031(조원)": cb["spc"]["net_benefit"][1], "순편익 양일 확률": cb["spc"]["P_nb_pos"],
                        "국가 순편익 2035(조원)": cb["h2035"]["net_benefit"][1], "SPC 순비용(조원)": cb["spc"]["net_cost"][1],
                        "손익분기 전환율 2031": fmt_breakeven(breakeven_capture(cb["grid2031"], round(r["G2029_dom"], 3)))})
        rows.append(row); print(fk, vk, round(r["G2029_dom"], 3), round(r["lag_front"][1], 2), round(r["gap"][1], 1), row.get("국가 순편익 2031(조원)"))
df = pd.DataFrame(rows)
fe = {fk: factory_economics(fk) for fk in FACTORY_PATHS}
out = Path(__file__).resolve().parent / "out"; out.mkdir(exist_ok=True)
df.to_csv(out / "factory_b2g_matrix.csv", index=False, encoding="utf-8-sig")
json.dump(dict(matrix=rows, factory_economics=fe, N=N, note="흡수 역량 상한 적용, 시드 20260930/7"), open(out / "factory_b2g_matrix.json", "w"), ensure_ascii=False, indent=1, default=float)
for k, v in fe.items(): print(k, {kk: [round(x, 3) if abs(x) < 100 else round(x / 1e12, 2) for x in vv] for kk, vv in v.items()})
