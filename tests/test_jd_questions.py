"""JD question generation test (WBS 4.2): >=10 questions, JD-flavored.

Usage: python tests/test_jd_questions.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent


def test_generate_from_jd():
    jd_text = agent.load_jd()
    assert jd_text, "no JD file found under data/"
    questions = agent.generate_questions(jd_text)
    print(f"[i] generated {len(questions)} questions")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    assert len(questions) >= 8, f"expected >=8 questions from JD, got {len(questions)}"


if __name__ == "__main__":
    test_generate_from_jd()
    print("\nJD question generation test passed")
