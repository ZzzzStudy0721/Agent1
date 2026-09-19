@echo off
REM Start the LangGraph Harness (debug UI) for the interviewer agent.
REM PYTHONUTF8=1 is required on Chinese Windows: without it langgraph-api
REM reads its config with the GBK codec and crashes on startup.
set PYTHONUTF8=1
langgraph dev
