import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
st.set_page_config(page_title="소버린 AI 추진체계 시뮬레이터", page_icon="📊", layout="wide")
from app_pages._ui import init_state
init_state()

pages = {
    "시작": [st.Page("app_pages/home.py", title="개요", default=True)],
    "데이터와 진단": [st.Page("app_pages/data.py", title="데이터 관리"),
                  st.Page("app_pages/gap.py", title="격차 진단과 모수 보정"),
                  st.Page("app_pages/forecast.py", title="격차 전망")],
    "시나리오 평가": [st.Page("app_pages/scenarios.py", title="시나리오 시뮬레이션"),
                  st.Page("app_pages/grid.py", title="규모·지분 격자"),
                  st.Page("app_pages/factory.py", title="민간 팩토리와 B2G 전용")],
    "경제성": [st.Page("app_pages/economics.py", title="비용편익과 ROI"),
            st.Page("app_pages/b2g.py", title="공공수요 적산"),
            st.Page("app_pages/macro.py", title="국가 거시경제 파급"),
            st.Page("app_pages/appraisal.py", title="예타식 종합 평가")],
    "전략": [st.Page("app_pages/strategy.py", title="민감도·시나리오 발견·중단 규칙")],
    "정리": [st.Page("app_pages/compare.py", title="결과 비교와 내보내기"),
           st.Page("app_pages/method.py", title="방법과 출처")],
}
st.navigation(pages).run()
