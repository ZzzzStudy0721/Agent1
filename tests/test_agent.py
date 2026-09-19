"""M3 v2 end-to-end test: free-form interview driven to wrap with scripted answers.

Usage: python tests/test_agent.py
"""
import itertools
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent
from langchain_core.messages import HumanMessage
import pytest

pytestmark = pytest.mark.live  # real LLM calls end to end

ANSWER_POOL = [
    "我是物联网工程专业的应届生，毕设做的是基于 YOLOv8 和 DeepSORT 的交通视频车辆检测与计数系统。",
    "我对比了 Faster R-CNN、YOLOv5s/n 和 YOLOv8n，最终选 YOLOv8n，帧率 35FPS 且精度最高。",
    "车辆计数采用虚拟检测线算法，配合最小尺寸过滤、有效区域掩码、Track ID 去重和分车道分车型统计四重过滤，计数误差控制在 3.5% 以内。",
    "系统用 PyQt5 做界面，Flask 提供接口，模型用 ONNX 导出加速推理。",
    "我会先写模块接口契约和 JSON 格式的需求描述，AI 生成代码后我逐模块审查并跑集成测试。",
    "最大的挑战是车辆遮挡导致 ID 切换，最后用 DeepSORT 替代 SORT，ID 切换降低 46.7%。",
    "我现在在学 LangChain 和 LangGraph，用 Claude API 做过 RAG 项目。",
    "我想做 AI Agent 应用开发方向，正在冲刺求职。",
    "数据按 8:2 划分训练验证集，用了直方图均衡化和组合滤波预处理，最终 mAP@0.5 达到 93.2%。",
    "车速估算用检测框位移除以帧间隔，误差小于 4.1km/h，夜间用高斯滤波降噪。",
    "模型部署阶段用 TensorRT 做过一次优化尝试，但最后为了稳定选择了 ONNX Runtime。",
    "RAG 项目里我用 BM25 和向量检索做混合召回，再用 CrossEncoder 重排，Recall@5 提到 0.986。",
    "遇到报错时我会把错误信息回贴给 AI 并加约束重新生成，复杂问题自己读代码定位。",
    "面试官模拟项目的设计思路：用状态机保证流程可控，同时让 LLM 做自由对话的决策。",
]


def test_full_interview():
    graph = agent.build_turn_graph()
    state = agent.init_state(agent.DEFAULT_TOPICS)
    pool = itertools.cycle(ANSWER_POOL)  # guards against a runaway interview
    rounds = 0
    while not state["finished"]:
        ans = next(pool)  # raises StopIteration if the interview runs away
        state["messages"] = state["messages"] + [HumanMessage(content=ans)]
        state["output"] = ""
        state["decision"] = {}
        result = graph.invoke(state)
        state = dict(result)
        rounds += 1
        assert result["output"], "empty graph output"
        assert result["round_count"] <= agent.MAX_ROUNDS, "round cap guardrail broken"
        # the last invoke is the wrap round, so rounds may reach MAX_ROUNDS + 1
        assert rounds <= agent.MAX_ROUNDS + 1, "round cap guardrail broken"
    print(f"\n{rounds} rounds completed, topics covered = {state['covered_topics']}")
    print(f"\n--- wrap output ---\n{state['output'][-600:]}")


if __name__ == "__main__":
    test_full_interview()
    print("\nAgent state machine test passed")
