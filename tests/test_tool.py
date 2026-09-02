"""Tool Calling test (WBS 4.6): export_report tool writes a markdown report.

Usage: python tests/test_tool.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent


def test_export_report_tool():
    result = agent.export_report.invoke(
        {"filename": "test_report.md", "content": "# 测试报告\n\n这是测试内容。"}
    )
    print(f"[tool result] {result}")
    path = os.path.join(agent.REPORTS_DIR, "test_report.md")
    assert os.path.isfile(path), "report file was not created"
    with open(path, encoding="utf-8") as f:
        assert "测试报告" in f.read(), "report content mismatch"


if __name__ == "__main__":
    test_export_report_tool()
    print("\nTool Calling test passed")
