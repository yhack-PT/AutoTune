from __future__ import annotations

"""
Modal vLLM serving script for LoRA adapters produced by modal_trl_posttrain.py.

Deploys a base model with one or more LoRA adapters (or a merged model) via
vLLM's OpenAI-compatible API on Modal.

Configure via environment variables:

```bash
# Serve a LoRA adapter on top of the base model
ADAPTER_PATH=experiments/qwen3-8b-sft/final_adapter \
  modal serve backend/modal_vllm_serve.py

# Serve a merged model directly (no LoRA)
MERGED=1 ADAPTER_PATH=experiments/qwen3-8b-sft/merged \
  modal serve backend/modal_vllm_serve.py

# Override base model, GPU, and context length
BASE_MODEL=meta-llama/Llama-3-8B \
  ADAPTER_PATH=experiments/llama3-sft/final_adapter \
  GPU_TYPE=A100 MAX_MODEL_LEN=4096 \
  modal serve backend/modal_vllm_serve.py
```

Test the endpoint:

```bash
curl http://<modal-url>/v1/models
curl http://<modal-url>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "my-adapter", "prompt": "Hello", "max_tokens": 32}'
```

Notes:
- Reuses the same volumes and secrets as modal_trl_posttrain.py.
- Create a Modal secret named `huggingface-secret` before running.
- The ADAPTER_PATH is relative to the checkpoints volume root (/checkpoints).
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config via environment variables (works with `modal serve` and `modal deploy`)
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
ADAPTER_NAME = os.environ.get("ADAPTER_NAME", "")
GPU_TYPE = os.environ.get("GPU_TYPE", "A10G")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
MERGED = os.environ.get("MERGED", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = os.environ.get("APP_NAME", "vllm-lora-serve")
VLLM_PORT = 8000

MODEL_CACHE_DIR = Path("/model_cache")
CHECKPOINTS_DIR = Path("/checkpoints")

HF_SECRET = modal.Secret.from_name("huggingface-secret")
MODEL_CACHE_VOLUME = modal.Volume.from_name("trl-model-cache", create_if_missing=True)
CHECKPOINTS_VOLUME = modal.Volume.from_name("trl-checkpoints", create_if_missing=True)

serve_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("vllm>=0.11.1", "huggingface_hub")
    .env(
        {
            "HF_HOME": str(MODEL_CACHE_DIR / "huggingface"),
            "TRANSFORMERS_CACHE": str(MODEL_CACHE_DIR / "transformers"),
            "BASE_MODEL": BASE_MODEL,
            "ADAPTER_PATH": ADAPTER_PATH,
            "ADAPTER_NAME": ADAPTER_NAME,
            "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "MERGED": "1" if MERGED else "0",
        }
    )
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_adapter_name(adapter_path: str) -> str:
    """Derive a short adapter name from the path, e.g. 'experiments/qwen3-8b-sft/final_adapter' -> 'qwen3-8b-sft'."""
    parts = Path(adapter_path).parts
    # If the path looks like experiments/<name>/final_adapter, use <name>
    if len(parts) >= 2 and parts[-1] in ("final_adapter", "merged"):
        return parts[-2]
    # Otherwise use the last path component
    return parts[-1] if parts else "adapter"


def _get_runtime_config() -> dict[str, str | int | bool]:
    """Read serving config from the container environment."""
    return {
        "base_model": os.environ.get("BASE_MODEL", BASE_MODEL),
        "adapter_path": os.environ.get("ADAPTER_PATH", ADAPTER_PATH),
        "adapter_name": os.environ.get("ADAPTER_NAME", ADAPTER_NAME),
        "max_model_len": int(os.environ.get("MAX_MODEL_LEN", str(MAX_MODEL_LEN))),
        "merged": os.environ.get("MERGED", "1" if MERGED else "0").lower() in ("1", "true", "yes"),
    }


def _build_vllm_cmd() -> list[str]:
    """Build the vLLM launch command from the active runtime config."""
    config = _get_runtime_config()
    adapter_path = str(config["adapter_path"])
    adapter_name = str(config["adapter_name"])
    base_model = str(config["base_model"])
    max_model_len = int(config["max_model_len"])
    merged = bool(config["merged"])

    if merged:
        if not adapter_path:
            raise ValueError("ADAPTER_PATH is required when MERGED is set (path to the merged model directory).")
        resolved = CHECKPOINTS_DIR / adapter_path
        if not resolved.exists():
            raise FileNotFoundError(f"Merged model directory not found: {resolved}")
        model = str(resolved)
    else:
        model = base_model

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", str(max_model_len),
        "--dtype", "auto",
        "--trust-remote-code",
    ]

    # Add LoRA flags for non-merged adapter serving
    if not merged and adapter_path:
        resolved = CHECKPOINTS_DIR / adapter_path
        if not resolved.exists():
            raise FileNotFoundError(
                f"Adapter not found: {resolved}. "
                "Check that training has completed and the ADAPTER_PATH is correct."
            )
        name = adapter_name or _derive_adapter_name(adapter_path)
        cmd += ["--enable-lora", "--lora-modules", f"{name}={resolved}"]

    return cmd


# ---------------------------------------------------------------------------
# Serve function
# ---------------------------------------------------------------------------

@app.function(
    image=serve_image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    min_containers=1,
    scaledown_window=300,
    volumes={
        str(MODEL_CACHE_DIR): MODEL_CACHE_VOLUME,
        str(CHECKPOINTS_DIR): CHECKPOINTS_VOLUME,
    },
    secrets=[HF_SECRET],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve():
    MODEL_CACHE_VOLUME.reload()
    CHECKPOINTS_VOLUME.reload()

    config = _get_runtime_config()
    print(f"Serving config: {config}", flush=True)
    cmd = _build_vllm_cmd()
    print(f"Starting vLLM: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


# ---------------------------------------------------------------------------
# Local entrypoint (for `modal run` — just prints config)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Current config (from environment variables):")
    print(f"  APP_NAME:      {APP_NAME}")
    print(f"  BASE_MODEL:    {BASE_MODEL}")
    print(f"  ADAPTER_PATH:  {ADAPTER_PATH or '(none)'}")
    print(f"  ADAPTER_NAME:  {ADAPTER_NAME or '(auto)'}")
    print(f"  GPU_TYPE:      {GPU_TYPE}")
    print(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
    print(f"  MERGED:        {MERGED}")
    print()
    print("Use `modal serve` to start the vLLM server:")
    print(f"  ADAPTER_PATH=experiments/your-run/final_adapter modal serve backend/modal_vllm_serve.py")
