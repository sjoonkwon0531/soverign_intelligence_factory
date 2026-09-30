"""모든 화면이 예외 없이 그려지는지 확인(Streamlit AppTest)."""
import sys
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PAGES = ["home", "data", "gap", "forecast", "scenarios", "grid", "factory", "economics", "b2g", "macro", "appraisal", "strategy", "compare", "method"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=600)
    at.run()
    at.switch_page(f"app_pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
