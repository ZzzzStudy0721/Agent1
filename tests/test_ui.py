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


def test_uploaded_resume_shows_extracted_preview():
    """Uploading a Word resume parses it and shows the text for confirmation.

    Drives the real page: if the session_state / widget ordering breaks, this
    fails with a StreamlitAPIException instead of a blank upload area.
    """
    import io

    import docx
    from streamlit.testing.v1 import AppTest

    document = docx.Document()
    document.add_paragraph("上传的简历：精通 Python")
    buf = io.BytesIO()
    document.save(buf)

    at = AppTest.from_file(UI, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("🎤 模拟面试").run()
    at.file_uploader[0].set_value(
        (
            "我的简历.docx",
            buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    ).run()

    assert not at.exception, [e.value for e in at.exception]
    preview = at.text_area(key="resume_preview")
    assert "精通 Python" in preview.value


def test_interview_page_renders_upload_area():
    """Interview tab shows the resume/JD upload area, still without loading models."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(UI, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("🎤 模拟面试").run()

    assert not at.exception, [e.value for e in at.exception]
    labels = [e.label for e in at.expander]
    assert any("上传简历" in label for label in labels), labels
    assert len(at.file_uploader) == 2, [u.label for u in at.file_uploader]


def test_history_module_stays_standard_library_only():
    """ui.py imports history at module level, so history must not grow a heavy
    dependency (torch / langchain / streamlit) — that would put the 1-2 minute
    page-load block right back."""
    tree = ast.parse(open(os.path.join(BASE_DIR, "history.py"), encoding="utf-8").read())
    assert _top_level_imports(tree) <= {"json", "logging", "os"}


def test_history_area_invites_the_first_interview(monkeypatch, tmp_path):
    """With no history file the area explains itself instead of rendering an
    empty table."""
    import history
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "missing.json"))

    at = AppTest.from_file(UI, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("🎤 模拟面试").run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("还没有记录" in c.value for c in at.caption), [c.value for c in at.caption]


def test_history_area_shows_a_recorded_interview(monkeypatch, tmp_path):
    """A recorded interview renders as a table row plus its detail view."""
    import json

    import history
    from streamlit.testing.v1 import AppTest

    entry = {
        "time": "2026-09-29 20:15",
        "job": "默认",
        "rounds": 6,
        "topics": ["毕设深挖"],
        "report": "interview_report_20260929_2015.md",
        "technical_depth": 7,
        "communication": 6,
        "project_experience": 8,
        "job_fit": 7,
        "star_completeness": 5,
        "summary": "整体表现稳健。",
        "strength": "项目经历描述具体。",
        "improvement": "缺少量化数据。",
    }
    path = tmp_path / "history.json"
    path.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(history, "HISTORY_FILE", str(path))
    monkeypatch.setattr(history, "REPORTS_DIR", str(tmp_path))

    at = AppTest.from_file(UI, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("🎤 模拟面试").run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.dataframe, "the score table should render"
    assert any("查看某一场的详情" in s.label for s in at.selectbox), [
        s.label for s in at.selectbox
    ]
    scores = " ".join(m.value for m in at.markdown)
    assert "整体表现稳健。" in scores, "the detail view should show the review summary"
    # the verdicts render as callouts, not markdown
    assert any("缺少量化数据。" in w.value for w in at.warning), [w.value for w in at.warning]
    assert any("项目经历描述具体。" in s.value for s in at.success), [s.value for s in at.success]
