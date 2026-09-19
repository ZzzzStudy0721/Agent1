"""LangGraph Harness entry point (`langgraph dev`).

langgraph.json points the Harness server at the `graph` attribute below. The
server checkpoints interview state per thread, so the debug UI drives the whole
interview across runs: greeting -> candidate answers -> interviewer turns ->
wrap with debrief + evidence verification + report export.

Run from the project root:

    langgraph dev

and open the debug UI (a browser tab opens automatically on startup).
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agent import build_interview_graph  # noqa: E402

graph = build_interview_graph()
