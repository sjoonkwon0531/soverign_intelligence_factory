import streamlit as st

st.title("방법과 출처")
t1, t2, t3, t4 = st.tabs(["모형 구성과 식", "경제 이론의 위치", "해석 원칙", "참고문헌과 자료"])
with t1:
    st.markdown(r"""
### 모듈
| 모듈 | 내용 | 코드 |
|---|---|---|
| 격차 진단 | 역량지수 최고 기록의 선형 추세, AA 환산, 연산량 회귀, 생산 능력·가격 환산 | `core/gap.py` |
| 시나리오 | 연 단위 몬테카를로. 물리층(장비·전력), 역량층(연산·흡수), 재무, 게이트, 중단 규칙 | `core/sim.py` |
| 규모·지분 격자 | 출자 여력, 흡수 역량, 계통, GPU 공급 상한, 임대 방식 | `core/grid.py` |
| 민간 팩토리 | 가동 GW 경로, 국가 할당 용량, 인재 파급, 시장 잠식, 전력 부담 | `core/factory.py` |
| 비용편익 | 실패 시 잔존가치, 임차 경로 종속비용, 순편익, 손익분기, 2035 지평 | `core/costbenefit.py` |
| 재무 지표 | NPV, IRR, 회수기간, 편익비용비, ROI, 비용 효과성 | `core/econ.py` |
| 부가가치·공공수요 | 산업연관표 계수, 상향식 적산 | `core/gva_b2g.py` |
| 거시 파급 | 과업 기반 성장회계, 로지스틱 확산, J-곡선, GDP와 GNI 구분, 차단 충격 | `core/macro.py` |
| 종합 평가 | 사회적 할인율 편익비용비, 공공자금 한계비용, AHP | `core/appraisal.py` |
| 전략 분석 | 표준화 회귀계수 민감도, 결정나무 시나리오 발견, 중단 규칙의 옵션 가치 | `core/strategy.py` |

### 핵심 식
- 물리층: $S_t=\min(C_t,\,E_t/\varepsilon)$, $C_t=\sum_v(1-\delta)^{t-v} I_v/p_v$
- 역량층: $A_t=A_0+a_1[\log_{10}(K_t+K_0)-\log_{10}K_0]+\zeta_t\,g\,t$, $\zeta_t=\min(1,\zeta_0+g_\zeta t)$
- 프런티어와 문턱: $A^F_t=A^F_0+d\,\tau_t$, $\tau_t=\min(t,1)+\kappa\max(0,t-1)$ ($\kappa$: 진보 체제 배수, 감속 0.5·기준 1·가속 1.3), $q_{c,t}=A^F_t-\Delta_{open}$, 격차(년) $=(A^F_t-A_t)/d$
- 게이트: 2027 $A\ge q_c-3$, 2028 $A\ge q_c$ 및 상업 매출, 2029 도메인 $A\ge q_c-5$, 프런티어 $A\ge A^F-5$
- 품질 연동 조달: $A_t<q_{c,t}-5$이면 공공 조달 중 보호 물량 비율만 국산 발주
- 민간 팩토리 할당 용량: 팀별 사용량 $=\min(\text{할당},\ \text{흡수 역량 상한}-\text{자체 보유})$, 사용분에만 이용료 부과
- 순편익(미시): 성공 시 회피한 종속비용 − (누적 재무 갭 − 잔존가치)
- 거시 효과: $Y^{AI}_t=Y_t\,G\,\Lambda_t-Y_t\,G\,J\,\Delta\Lambda_t$ ($\Lambda_t$ 로지스틱 확산, $J$ 보완투자 비율)
- 국내 귀속: $R_t=X_t[s_t v_d+(1-s_t)v_l]$, $X_t=\sigma Y^{AI}_t$ (AI 서비스 지출), $s_t$ 국내 공급 비중
- 차단 손실: 차단 연도에 $Y^{AI}_t f_{front}\min(T,1)$, 주권 커버리지가 있으면 30%만 발생
- 편익비용비: $B/C=\dfrac{\sum_t \Delta R_t(1+r)^{-t}+\sum_t \Delta L_t(1+r)^{-t}}{\sum_t (I_t+O_t)(1+r)^{-t}[1+(\text{MCPF}-1)\theta]}$ ($\theta$ 재정 조달 비중)

### 재현
기본 모수(본보고서 r1, 2026. 9. 30.)와 시드 20260930, 3,000회로 실행하면 본보고서 표의 값과 같습니다. `pytest tests/` 로 확인합니다.
""")
with t2:
    st.markdown(r"""
### 이 시뮬레이터가 주류 이론과 만나는 지점
**1. AI의 거시 효과: 과업 기반 접근.** Hulten(1978)의 정리에 따르면 한 부문의 생산성 향상이 총생산성에 주는 효과는 그 부문의 GDP 비중과 비용 절감의 곱으로 근사됩니다. Acemoglu(2025)는 이를 AI에 적용해 10년 총요소생산성 효과를 0.7% 안팎으로 낮게 추정했고, 한국은행(2025)은 한국의 장기 GDP 효과를 4.2～12.6%로 추정했습니다. 이 도구는 효과의 크기를 범위로 두고, 추진 방식에 따라 달라지지 않는다고 봅니다. **소버린 AI의 편익은 효과의 크기가 아니라 그 효과의 국내 귀속과 위험 분포에서 나온다**는 것이 이 모듈의 핵심 설계입니다.

**2. 범용기술과 생산성 J-곡선.** Bresnahan and Trajtenberg(1995)의 범용기술 이론은 AI 같은 기술의 효과가 보완적 혁신과 투자를 거쳐 늦게 나타난다고 봅니다. Brynjolfsson, Rock and Syverson(2021)은 무형 보완투자가 초기 측정 생산성을 낮추는 J-곡선을 제시했습니다. 거시 모듈의 보완투자 항이 이를 반영합니다.

**3. 자동화와 성장.** Aghion, Jones and Jones(2019)는 AI가 자동화하는 과업의 범위와 병목 과업(보몰 효과)이 성장률을 결정한다고 보았고, Korinek and Suh(2024)는 AGI 시나리오에서 임금과 산출의 경로를 분석했습니다. Epoch AI의 GATE 모형(2025)은 연산 투자와 자동화를 결합한 통합 평가 모형입니다. 이 도구의 가속 체제(진보 속도 1.3배)는 이러한 자동화 가속 시나리오를 단순화한 것입니다.

**4. 학습효과와 전략적 무역정책.** Arrow(1962)의 실행에 의한 학습과 Irwin and Klenow(1994)의 반도체 학습곡선 실증은 초기 생산 경험이 이후 비용을 낮추는 경로를 보여 줍니다. 역량층의 흡수율 개선($g_\zeta$)과 결집 가산이 이 경로입니다. Brander and Spencer(1985)는 불완전경쟁 산업에서 정부 지원이 지대를 자국으로 옮길 수 있음을 보였으나, 그 조건(과점 구조, 신뢰할 만한 약속)이 까다롭다는 점도 함께 알려져 있습니다. Juhász, Lane and Rodrik(2024)는 새로운 산업정책이 성과 조건부 지원과 폐기 규칙을 갖출 때 효과적이라고 정리했습니다. 게이트와 에스크로 설계가 이 원칙을 따릅니다.

**5. 공공투자 평가.** 편익비용비는 기획재정부 예비타당성조사 지침의 사회적 할인율 4.5%를 기본으로 합니다. 재정 조달의 초과부담은 공공자금의 한계비용(Dahlby 2008)으로 선택 반영합니다. 종합 평가는 예타의 AHP 틀을 대안 비교용으로 단순화했습니다.

**6. 깊은 불확실성 아래의 의사결정.** 확률 분포를 신뢰하기 어려운 문제에서는 최적안보다 강건한 안을 찾는 것이 합리적입니다(Lempert, Popper and Bankes 2003). 시나리오 발견은 권고안이 불리해지는 모수 영역을 찾아 조기 경보 지표로 바꿉니다. 전역 민감도는 Saltelli et al.(2008)의 표준화 회귀계수 방법을 따릅니다. 중간 점검과 중단 규칙은 Dixit and Pindyck(1994)의 단계 투자 옵션 논리입니다.
""")
with t3:
    st.markdown("""
- 확률과 금액은 명시된 가정에 따른 조건부 값입니다. 결론은 시나리오 간 순위와 감도에 한정합니다.
- 흡수 경로 차단 결과는 확률을 부여하지 않은 조건부 추정입니다.
- 한국 모델은 역량지수에 없으므로 AA 지수 환산(95% 예측구간 약 ±3.6점, 시간으로 ±0.23년)에 의존합니다.
- 거시 효과의 절대 수준은 외부 추정 범위에 의존하므로 시나리오 간 차이로만 해석합니다. 국내 귀속 부가가치와 산업연관 파급효과는 더하지 않습니다.
- 미시(비용편익)와 거시 편익은 같은 편익을 다른 경로로 잰 것이므로 더하지 않고 방향이 같은지 교차 확인합니다.
- 민간 팩토리 모듈의 할당 용량, 시장 잠식, 인재 파급 계수는 사례 기반 가정이며 협상 결과에 따라 크게 달라집니다.
- 공공자금 한계비용과 AHP 가중치·판정 기준은 국내 공식 값을 확인하지 못했으므로 사용자 입력으로 둡니다.
- 2025 명목 GDP 기본값은 잠정치이므로 한국은행 국민계정 확정치로 바꿔 쓰십시오.
""")
with t4:
    st.markdown("""
### 경제 이론과 방법
- Acemoglu, D. (2025). The simple macroeconomics of AI. *Economic Policy*, 40(121), 13–58. doi:10.1093/epolic/eiae042
- Aghion, P., Jones, B. F., & Jones, C. I. (2019). Artificial intelligence and economic growth. In A. Agrawal, J. Gans, & A. Goldfarb (Eds.), *The Economics of Artificial Intelligence: An Agenda* (ch. 9). University of Chicago Press.
- Arrow, K. J. (1962). The economic implications of learning by doing. *Review of Economic Studies*, 29(3), 155–173.
- Brander, J. A., & Spencer, B. J. (1985). Export subsidies and international market share rivalry. *Journal of International Economics*, 18. (NBER Working Paper 1464)
- Bresnahan, T. F., & Trajtenberg, M. (1995). General purpose technologies 'Engines of growth'? *Journal of Econometrics*, 65(1), 83–108.
- Brynjolfsson, E., Rock, D., & Syverson, C. (2021). The productivity J-curve: How intangibles complement general purpose technologies. *American Economic Journal: Macroeconomics*, 13(1), 333–372. doi:10.1257/mac.20180386
- Dahlby, B. (2008). *The Marginal Cost of Public Funds: Theory and Applications*. MIT Press.
- Dixit, A. K., & Pindyck, R. S. (1994). *Investment under Uncertainty*. Princeton University Press.
- Epoch AI (2025). GATE: An integrated assessment model for AI automation. arXiv:2503.04941
- Hulten, C. R. (1978). Growth accounting with intermediate inputs. *Review of Economic Studies*, 45(3), 511–518.
- Irwin, D. A., & Klenow, P. J. (1994). Learning-by-doing spillovers in the semiconductor industry. *Journal of Political Economy*, 102(6), 1200–1227.
- Juhász, R., Lane, N., & Rodrik, D. (2024). The new economics of industrial policy. *Annual Review of Economics*, 16, 213–242.
- Korinek, A., & Suh, D. (2024). Scenarios for the transition to AGI. NBER Working Paper 32255.
- Lempert, R. J., Popper, S. W., & Bankes, S. C. (2003). *Shaping the Next One Hundred Years: New Methods for Quantitative, Long-Term Policy Analysis* (MR-1626). RAND.
- Saltelli, A., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- 한국은행 (2025). AI의 경제적 효과와 시사점(BOK 이슈노트 2025-2).
- 기획재정부. 예비타당성조사 수행 총괄지침(사회적 할인율 4.5%, 2017. 9. 이후; 2019 평가 가중치 개편).

### 자료
- Epoch AI, Epoch Capabilities Index, Notable AI Models, AI Chip Owners, GPU Clusters, AI Data Centers Hub, ML Hardware (CC BY 4.0, 조회일 2026. 9. 21.～22.)
- Artificial Analysis, Intelligence Index v4.3.2 (2026. 9. 22. 조회)
- Epoch AI, LLM inference prices have fallen rapidly but unequally across tasks (2025. 3. 12.); Stanford HAI, AI Index Report 2025
- 한국은행 2020년 기준년 산업연관표; 과학기술정보통신부 2027년도 예산안(2026. 8. 31.); 공공부문 SW·ICT 수요예보 2026
""")
