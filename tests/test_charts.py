"""Ability radar chart: trace layout, the closing point, and the baseline.

plotly needs no API key and renders nothing here — the figure is inspected as
data, which is why the chart lives in its own module instead of inline in
ui.py (where the only way to check it would be the slow Streamlit AppTest).
"""
import ast
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import charts  # noqa: E402
from history import SCORE_DIMENSIONS  # noqa: E402


def test_plotly_is_imported_lazily():
    """ui.py imports charts on every page load, so plotly must stay out of the
    module level: most renders have no history to chart and should not pay for
    it (the same reason `agent` is imported inside functions in ui.py)."""
    tree = ast.parse(open(os.path.join(BASE_DIR, "charts.py"), encoding="utf-8").read())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    assert "plotly" not in roots, "plotly must be imported inside ability_radar()"


def _entry(**overrides):
    """One interview record with every dimension at 5."""
    entry = {key: 5 for key, _ in SCORE_DIMENSIONS}
    entry.update(overrides)
    return entry


def test_single_interview_has_no_average_trace():
    fig = charts.ability_radar([_entry()], 0)

    assert len(fig.data) == 1


def test_radar_closes_the_polygon():
    """The outline must repeat its first point, or the shape ends up with a gap."""
    fig = charts.ability_radar([_entry()], 0)
    trace = fig.data[0]

    assert len(trace.r) == len(SCORE_DIMENSIONS) + 1
    assert trace.r[0] == trace.r[-1]
    assert trace.theta[0] == trace.theta[-1]


def test_average_trace_appears_when_other_interviews_exist():
    fig = charts.ability_radar([_entry(), _entry()], 0)

    assert len(fig.data) == 2
    assert "历史平均" in fig.data[1].name


def test_average_excludes_the_selected_interview():
    """The baseline means "what I usually score" — counting the run under
    review would let a strong run raise its own baseline."""
    entries = [
        _entry(technical_depth=2),  # the interview being reviewed
        _entry(technical_depth=8),
        _entry(technical_depth=10),
    ]

    fig = charts.ability_radar(entries, 0)

    assert fig.data[1].r[0] == 9.0  # technical_depth is the first dimension


def test_dimensions_follow_score_dimensions_order():
    entries = [
        _entry(
            technical_depth=1,
            communication=2,
            project_experience=3,
            job_fit=4,
            star_completeness=5,
        )
    ]

    fig = charts.ability_radar(entries, 0)
    trace = fig.data[0]

    assert list(trace.r[:5]) == [1, 2, 3, 4, 5]
    assert list(trace.theta[:5]) == [label for _, label in SCORE_DIMENSIONS]


def test_missing_scores_do_not_crash_the_chart():
    """A hand-edited or truncated entry must still render something."""
    fig = charts.ability_radar([{"time": "2026-09-29 20:00"}], 0)

    assert list(fig.data[0].r[:5]) == [0, 0, 0, 0, 0]
