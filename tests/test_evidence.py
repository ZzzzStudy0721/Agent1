"""Evidence verification test (WBS 4.5): fabricated facts flagged, true facts pass.

Usage: python tests/test_evidence.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent

FABRICATED = (
    "我毕设的检测模型 mAP@0.5 达到了 99.9%，精确率 98.5%，"
    "车辆计数误差控制在 0.1% 以内，车速估算误差只有 0.5km/h。"
)
TRUTHFUL = (
    "毕设中车辆计数采用虚拟检测线算法，配合四重过滤机制，"
    "计数误差控制在 3.5% 以内，车速估算误差小于 4.1km/h。"
)


def test_fabricated_facts_flagged():
    result = agent.verify_answer(FABRICATED)
    print(f"[fabricated]\n{result}\n")
    assert "矛盾" in result or "⚠️" in result, "fabricated facts should be flagged"


def test_truthful_facts_pass():
    result = agent.verify_answer(TRUTHFUL)
    print(f"[truthful]\n{result}\n")
    assert "无冲突" in result, "truthful answer should pass verification"


if __name__ == "__main__":
    test_fabricated_facts_flagged()
    test_truthful_facts_pass()
    print("Evidence verification test passed")
