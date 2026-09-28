"""Interview history (reports/history.json) — recording, reading back, and
tolerance for a missing/corrupt file.

Everything runs against tmp_path: the real reports/ directory holds the
owner's own interview records and must never be touched by a test.
"""
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import history  # noqa: E402


@pytest.fixture
def hist(tmp_path, monkeypatch):
    """Redirect the history module at a temp directory."""
    monkeypatch.setattr(history, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "history.json"))
    return tmp_path


def test_load_missing_file_is_empty(hist):
    assert history.load_history() == []


def test_record_then_load_roundtrip(hist):
    entry = {"time": "2026-09-29 20:15", "job": "默认", "rounds": 6}

    history.record_history(entry)

    assert history.load_history() == [entry]


def test_record_appends_in_order(hist):
    history.record_history({"rounds": 1})
    history.record_history({"rounds": 2})

    assert [e["rounds"] for e in history.load_history()] == [1, 2]


def test_record_creates_reports_dir(hist):
    nested = hist / "reports"
    history.REPORTS_DIR = str(nested)
    history.HISTORY_FILE = str(nested / "history.json")

    history.record_history({"rounds": 1})

    assert os.path.isfile(nested / "history.json")


def test_corrupt_file_reads_as_empty(hist):
    (hist / "history.json").write_text("{not json", encoding="utf-8")

    assert history.load_history() == []


def test_non_list_json_reads_as_empty(hist):
    (hist / "history.json").write_text('{"a": 1}', encoding="utf-8")

    assert history.load_history() == []


def test_non_dict_entries_are_dropped(hist):
    (hist / "history.json").write_text('[{"rounds": 3}, "junk", 7]', encoding="utf-8")

    assert history.load_history() == [{"rounds": 3}]


def test_write_failure_is_swallowed(hist):
    """Scoring is done and the interview is over by the time history is
    written — an unwritable path must not turn that into a crash."""
    (hist / "history.json").mkdir()  # a directory can never be opened for write

    history.record_history({"rounds": 1})  # must not raise


def test_score_dimensions_match_the_stored_fields(hist):
    """The labels the UI draws its table from must cover exactly the score
    fields a recorded entry carries."""
    history.record_history(
        {field: 5 for field, _ in history.SCORE_DIMENSIONS} | {"rounds": 1}
    )

    entry = history.load_history()[0]

    assert set(entry) >= {field for field, _ in history.SCORE_DIMENSIONS}
