"""Streamlit UI tests: the page must load fast and stay healthy.

Regression guard for the "page won't open" bug: importing `agent` at ui.py
module level pulls in torch + sentence-transformers (~1.6 GB RSS, 1-2 min on
first load), which blocks the entire Streamlit script — the browser just spins
until it finishes. The import must stay inside the interview functions.
"""
import ast
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

UI = os.path.join(BASE_DIR, "ui.py")


def _top_level_imports(tree):
    """Roots of every module imported at file top level."""
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_ui_has_no_heavy_module_level_imports():
    """`agent` must not be imported at module level (it would block page load)."""
    tree = ast.parse(open(UI, encoding="utf-8").read())
    assert "agent" not in _top_level_imports(tree)


def test_agent_is_imported_lazily():
    """...but it must still be imported somewhere, or the buttons raise NameError."""
    tree = ast.parse(open(UI, encoding="utf-8").read())
    lazy = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(a.name == "agent" for a in node.names)
    ]
    assert lazy, "agent should be imported inside the interview functions"


def test_ui_page_renders_without_exception():
    """The page itself must render clean — no model loading, no exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(UI, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
