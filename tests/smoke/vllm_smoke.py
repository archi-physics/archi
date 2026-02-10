#!/usr/bin/env python3
"""vLLM provider smoke checks.

Validates that a vLLM server is reachable and serving at least one model,
then sends a minimal completion request to verify inference works end-to-end.

Expected env vars:
  VLLM_BASE_URL  – vLLM OpenAI-compatible API base (default: http://localhost:8000/v1)
  VLLM_MODEL     – (optional) specific model id to validate is loaded
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _fail(message: str) -> None:
    print(f"[vllm-smoke] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _info(message: str) -> None:
    print(f"[vllm-smoke] {message}")


def _check_vllm_health(base_url: str, timeout: int = 120) -> None:
    """Wait for /v1/models to return at least one model."""
    models_url = f"{base_url}/models"
    _info(f"Waiting for vLLM at {models_url} (timeout {timeout}s) ...")
    deadline = time.time() + timeout
    last_err = None
    while True:
        try:
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    models = data.get("data", [])
                    if models:
                        model_ids = [m.get("id") for m in models]
                        _info(f"vLLM serving {len(models)} model(s): {model_ids}")
                        return
                    last_err = "No models loaded yet"
        except Exception as exc:
            last_err = str(exc)

        if time.time() >= deadline:
            _fail(f"vLLM not ready: {last_err}")
        time.sleep(3)


def _check_model_loaded(base_url: str, expected_model: str) -> None:
    """Verify a specific model is loaded on the server."""
    models_url = f"{base_url}/models"
    _info(f"Checking model '{expected_model}' is loaded ...")
    try:
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        _fail(f"Failed to query models: {exc}")

    model_ids = [m.get("id") for m in data.get("data", [])]
    if expected_model not in model_ids:
        _fail(f"Model '{expected_model}' not found in {model_ids}")
    _info(f"Model '{expected_model}' OK")


def _check_inference(base_url: str, model: str) -> None:
    """Send a minimal chat completion to verify inference works."""
    completions_url = f"{base_url}/chat/completions"
    _info(f"Testing inference on '{model}' ...")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        completions_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                _fail(f"Inference request failed: HTTP {resp.status}")
            data = json.loads(resp.read().decode())
    except Exception as exc:
        _fail(f"Inference request failed: {exc}")

    choices = data.get("choices", [])
    if not choices:
        _fail("Inference returned no choices")
    content = choices[0].get("message", {}).get("content", "")
    _info(f"Inference OK — response: {content!r}")


def main() -> None:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    expected_model = os.getenv("VLLM_MODEL", "")
    timeout = int(os.getenv("VLLM_HEALTH_TIMEOUT", "120"))

    _check_vllm_health(base_url, timeout=timeout)

    if expected_model:
        _check_model_loaded(base_url, expected_model)
    else:
        # Use the first available model for the inference check
        models_url = f"{base_url}/models"
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        expected_model = data["data"][0]["id"]

    _check_inference(base_url, expected_model)
    _info("vLLM smoke checks passed")


if __name__ == "__main__":
    main()
