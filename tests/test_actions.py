"""버튼으로 실행되는 계산 경로 점검."""
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _open(page):
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=900)
    at.run(); at.switch_page(f"app_pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _click(at, label):
    b = [x for x in at.button if x.label.startswith(label)][0]
    b.click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_appraisal_grid():
    at = _open("appraisal"); _click(at, "전환율")


def test_strategy_options():
    at = _open("strategy"); _click(at, "중단 규칙 평가")


def test_factory_matrix():
    at = _open("factory")
    at.multiselect[0].set_value(["F0", "F2"]).run()
    at.multiselect[1].set_value(["B B2G 전용(조달 보장)", "E 권고안(B2G와 민간·수출, 품질 연동)"]).run()
    at.selectbox[1].set_value(500).run()
    _click(at, "행렬 계산")


def test_accel_regime():
    at = _open("scenarios")
    sb = [s for s in at.selectbox if s.label == "프런티어 진보 체제"][0]
    sb.set_value("가속(2027년부터 속도 1.3배)").run()
    assert not at.exception
