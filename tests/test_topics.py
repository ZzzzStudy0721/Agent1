"""Topic anchor extraction test (WBS 4.2, v2): parse format + live extraction.

Usage: python tests/test_topics.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent
import pytest


def test_parse_topics():
    text = (
        "1. RAG 管线：混合检索与重排的理解\n"
        "2. 毕设深挖：车辆检测技术细节\n"
        "垃圾行\n"
        "3. 模型选型\n"
        "**4. 工程落地：** 计数误差归因分析\n"
    )
    topics = agent.parse_topics(text)
    assert len(topics) == 3, f"expected 3 valid topics, got {topics}"
    assert topics[0].startswith("RAG 管线")
    assert topics[1].startswith("毕设深挖")
    assert topics[2].startswith("工程落地")
    assert "*" not in topics[2], f"markdown bold not stripped: {topics[2]!r}"


@pytest.mark.live
def test_generate_from_jd_and_resume():
    jd_text = agent.load_jd()
    assert jd_text, "no JD file found under data/"
    resume_text = agent.load_resume()
    assert resume_text, "no resume files found under data/"
    topics = agent.generate_topics(jd_text, resume_text)
    print(f"[i] generated {len(topics)} topics")
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t}")
    assert len(topics) == 8, f"expected 8 topics, got {len(topics)}"


if __name__ == "__main__":
    test_parse_topics()
    test_generate_from_jd_and_resume()
    print("\nTopic extraction test passed")
