@echo off
REM Start the Streamlit interview UI (chat-style web page).
REM Keeps the console open so errors are visible if it fails.
set PYTHONUTF8=1
cd /d "%~dp0"
streamlit run ui.py
