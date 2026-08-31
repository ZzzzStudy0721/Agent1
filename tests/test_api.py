"""Smoke test: verify DeepSeek API key works (WBS task 1.3)."""
import json
import urllib.request

# Load key from .env without extra dependencies (MVP stage)
key = None
with open(".env", encoding="utf-8") as f:
    for line in f:
        if line.startswith("DEEPSEEK_API_KEY="):
            key = line.split("=", 1)[1].strip()

assert key, "DEEPSEEK_API_KEY not found in .env"

req = urllib.request.Request(
    "https://api.deepseek.com/chat/completions",
    data=json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 10,
        }
    ).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())

content = data["choices"][0]["message"]["content"]
print(f"API OK, reply: {content}")
assert "OK" in content, f"Unexpected reply: {content}"
