"""pytest hooks: keep test output out of the real reports/, and skip live
tests (real LLM/API calls) when no API key is set.

Live tests are marked with @pytest.mark.live. With DEEPSEEK_API_KEY in .env
they run normally; without it they are skipped so CI can run the pure-logic
suite without credentials.
"""
import os
import sys

import pytest
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

import history  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_report_output(tmp_path, monkeypatch):
    """Point report exports and interview history at a temp directory.

    Every test that drives the graph to its wrap node exports a report and
    appends a history entry. Without this the suite quietly fills the owner's
    own reports/ with fixture data — which is exactly what happened before
    this fixture existed.
    """
    reports = tmp_path / "reports"
    monkeypatch.setattr(history, "REPORTS_DIR", str(reports))
    monkeypatch.setattr(history, "HISTORY_FILE", str(reports / "history.json"))
    # agent is only patched when something already imported it: importing it
    # here would drag torch into the pure-logic tests.
    agent = sys.modules.get("agent")
    if agent is not None:
        monkeypatch.setattr(agent, "REPORTS_DIR", str(reports))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="DEEPSEEK_API_KEY not set — live tests skipped")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
