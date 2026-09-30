"""
test_api.py
-----------
Basic smoke tests for the backend API. Run from the backend/ directory:

    pytest ../tests/test_api.py -v

The first test run will download/load the model, so it can take a little
longer than usual — that's expected, not a failure.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from app.main import app
from app import model_service

client = TestClient(app)


def setup_module(module):
    # Ensure the model is loaded before any test runs, same as startup event.
    model_service.load_model()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["mode"] == "cpu_demo"


def test_metrics_shape():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "request_count" in body
    assert "average_tokens_per_second" in body


def test_generate_basic():
    resp = client.post("/generate", json={"prompt": "The weather today is", "max_new_tokens": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["generated_text"] != ""
    assert body["new_tokens"] > 0
    assert body["latency_seconds"] > 0


def test_generate_rejects_empty_prompt():
    resp = client.post("/generate", json={"prompt": "", "max_new_tokens": 10})
    assert resp.status_code == 422  # pydantic validation error


def test_chat_completions():
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] != ""


def test_batch_demo_batched_mode():
    resp = client.post(
        "/batch/demo",
        json={"prompts": ["Hi there", "Tell me a fact"], "mode": "batched", "max_new_tokens": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["requests_in_batch"] == 2
    assert len(body["timeline"]) == 2


def test_kv_cache_demo():
    resp = client.post("/demo/kv-cache", json={"prompt": "Once upon a time", "max_new_tokens": 15})
    assert resp.status_code == 200
    body = resp.json()
    assert "with_cache" in body and "without_cache" in body
