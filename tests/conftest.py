"""pytest hooks: skip live tests (real LLM/API calls) when no API key is set.

Live tests are marked with @pytest.mark.live. With DEEPSEEK_API_KEY in .env
they run normally; without it they are skipped so CI can run the pure-logic
suite without credentials.
"""
import os

import pytest
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="DEEPSEEK_API_KEY not set — live tests skipped")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
