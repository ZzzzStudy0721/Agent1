"""HTTP smoke test against a running uvicorn instance (start it first).

Usage: python tests/http_smoke.py
"""
import requests

r = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "车辆计数误差控制在多少？"},
    timeout=300,
)
print("status:", r.status_code)
d = r.json()
print("refused:", d["refused"])
print("answer:", d["answer"][:200])

r2 = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "我养了几只猫？"},
    timeout=300,
)
print("\nrefused check:", r2.json()["refused"], "|", r2.json()["answer"])
