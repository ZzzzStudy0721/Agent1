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

ANSWER_POOL = [
    "我对比了 Faster R-CNN、YOLOv5s/n 和 YOLOv8n 四种模型，最终选 YOLOv8n，因为它帧率 35FPS 且精度最高。",
    "采用虚拟检测线算法，配合四重过滤机制，计数误差控制在 3.5% 以内。",
    "我按结构化需求描述加模块接口契约的方式驱动 AI 生成代码，自己负责架构设计和集成联调。",
    "补充：对比实验中 YOLOv8n 的帧率达到 35FPS，远超 Faster R-CNN 的 0.42FPS，精度只差一个百分点。",
    "补充：四重过滤指最小尺寸过滤、有效区域掩码、Track ID 去重和分车道分车型统计。",
    "补充：我会先写模块接口契约和 JSON 格式的需求描述，AI 生成后我逐模块审查并跑集成测试。",
    "补充：最终 mAP@0.5 达到 93.2%，精确率 92.7%，ID 切换较 SORT 降低 46.7%。",
    "补充：数据集按 8:2 划分训练验证集，用了直方图均衡化和组合滤波预处理。",
    "补充：项目从零到可运行原型用了两周，交付了四个模块的完整系统。",
    "补充：遇到报错时我会把错误信息回贴给 AI 并加约束重新生成，复杂问题自己读代码定位。",
]


def test_full_interview():
    graph = agent.build_graph()
    history = []
    pool = iter(ANSWER_POOL)
    state = {
        "messages": history,
        "question_index": 0,
        "followup_count": 0,
        "followup_verdict": "",
        "output": "",
    }
    rounds = 0
    while True:
        result = graph.invoke(state)
        rounds += 1
        assert result["output"], "empty graph output"
        if result["question_index"] >= len(agent.DEMO_QUESTIONS):
            break
        ans = next(pool)  # raises StopIteration if follow-ups loop forever
        history.append(("user", ans))
        state = {
            "messages": history,
            "question_index": result["question_index"],
            "followup_count": result.get("followup_count", 0),
            "followup_verdict": "",
            "output": "",
        }
        assert rounds < 15, "graph did not converge"
    assert result["question_index"] >= len(agent.DEMO_QUESTIONS), "graph did not reach wrap"
    print(f"\n{rounds} rounds completed, final state index = {result['question_index']}")
    print(f"\n--- wrap output ---\n{result['output'][-400:]}")


if __name__ == "__main__":
    test_full_interview()
    print("\nAgent state machine test passed")
