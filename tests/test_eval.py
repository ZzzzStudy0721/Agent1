"""Evaluation module unit tests (non-live): rubric template + summary math.

Usage: python tests/test_eval.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "evaluation"))

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

from pydantic import ValidationError

import judge


def test_judge_prompt_anchors():
    prompt = judge.JUDGE_PROMPT.format(topic="话题A：说明", history="无", question="测试问题")
    for key in ("relevance", "specificity", "depth", "1=", "5="):
        assert key in prompt, f"missing rubric anchor {key!r}"


def test_judge_verdict_bounds():
    v = judge.JudgeVerdict(relevance=5, specificity=3, depth=1, reason="ok")
    assert v.relevance == 5
    for bad in (0, 6):
        try:
            judge.JudgeVerdict(relevance=bad, specificity=3, depth=3, reason="x")
            raise AssertionError("score out of range accepted")
        except ValidationError:
            pass


def test_summarize_pass_rates():
    vs = [
        judge.JudgeVerdict(relevance=5, specificity=4, depth=3, reason="a"),
        judge.JudgeVerdict(relevance=2, specificity=3, depth=5, reason="b"),
    ]
    s = judge.summarize(vs)
    assert s["n"] == 2
    assert s["relevance_mean"] == 3.5
    assert s["relevance_pass"] == 0.5
    assert s["depth_pass"] == 0.5
    assert judge.summarize([]) == {"n": 0}


if __name__ == "__main__":
    test_judge_prompt_anchors()
    test_judge_verdict_bounds()
    test_summarize_pass_rates()
    print("\nEvaluation unit tests passed")
