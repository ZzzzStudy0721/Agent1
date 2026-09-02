"""M3 skeleton test: run a full 3-question interview with scripted answers.

Usage: python tests/test_agent.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent

ANSWERS = [
    "我对比了 Faster R-CNN、YOLOv5s/n 和 YOLOv8n 四种模型，最终选 YOLOv8n，因为它帧率 35FPS 且精度最高。",
    "采用虚拟检测线算法，配合四重过滤机制，计数误差控制在 3.5% 以内。",
    "我按结构化需求描述加模块接口契约的方式驱动 AI 生成代码，自己负责架构设计和集成联调。",
]


def test_full_interview():
    graph = agent.build_graph()
    history = []
    state = {"messages": history, "question_index": 0, "output": ""}
    rounds = []
    for ans in ANSWERS:
        result = graph.invoke(state)
        rounds.append(result["output"])
        assert result["output"], "empty graph output"
        history.append(("user", ans))
        state = {
            "messages": history,
            "question_index": result["question_index"],
            "output": "",
        }
    result = graph.invoke(state)  # final round: review + wrap
    rounds.append(result["output"])
    assert result["question_index"] >= len(agent.DEMO_QUESTIONS), "graph did not reach wrap"
    for i, out in enumerate(rounds):
        print(f"\n--- round {i + 1} output ---\n{out[:300]}")
    print(f"\n{len(rounds)} rounds completed, final state index = {result['question_index']}")


if __name__ == "__main__":
    test_full_interview()
    print("\nAgent state machine test passed")
