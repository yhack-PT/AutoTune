"""Quick test script for the Modal vLLM endpoint."""

import requests

BASE_URL = "https://andrew-qian64--vllm-lora-serve-serve-dev.modal.run"

# Check available models
print("=== Models ===")
r = requests.get(f"{BASE_URL}/v1/models")
if r.headers.get("content-type", "").startswith("application/json"):
    print(r.json())
else:
    print(f"Status {r.status_code}: {r.text[:500]}")

# Run a chat completion
print("\n=== Chat Completion ===")
r = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "qwen3.5-9b-sft",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?",
            }
        ],
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "max_tokens": 64,
    },
)
if r.headers.get("content-type", "").startswith("application/json"):
    resp = r.json()
    if "choices" in resp:
        print(resp["choices"][0]["message"]["content"])
    else:
        print(resp)
else:
    print(f"Status {r.status_code}: {r.text[:500]}")
