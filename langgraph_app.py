"""LangGraph Harness 入口（`langgraph dev`）。

langgraph.json 会把 Harness 服务端指向下面的 `graph` 属性。服务端按 thread
对面试状态做 checkpoint，因此调试 UI 可以跨多次运行驱动整场面试：
问候 -> 候选人回答 -> 面试官提问 ->
收尾（复盘 + 证据核实 + 报告导出）。

在项目根目录执行：

    langgraph dev

然后打开调试 UI（启动时会自动弹出一个浏览器标签页）。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agent import build_interview_graph  # noqa: E402

graph = build_interview_graph()
