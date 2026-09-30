"""
main.py
-------
FastAPI entrypoint for the TGI LLM Serving Dashboard backend (CPU/demo mode).

Run with:  uvicorn app.main:app --reload --port 8000   (from backend/)

Routes
------
GET  /                      basic info
GET  /health                model + server readiness
GET  /metrics                request count, avg latency, tokens/sec
POST /generate               plain or streaming (SSE) text generation
POST /v1/chat/completions    OpenAI-compatible chat endpoint (streaming supported)
POST /batch/demo             sequential vs batched comparison
POST /demo/kv-cache          use_cache True vs False comparison
POST /demo/quantization      fp32 vs INT8-dynamic-quantized comparison
"""

import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import config, metrics, model_service, batching
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    ChatCompletionRequest,
    BatchDemoRequest,
    KvCacheDemoRequest,
    QuantizationDemoRequest,
)

app = FastAPI(
    title="TGI LLM Serving Dashboard — Backend (CPU/Demo Mode)",
    description=(
        "FastAPI backend that demonstrates TGI-style LLM serving concepts "
        "(continuous batching, KV cache, streaming, quantization) on CPU, "
        "using a small Hugging Face model. This is a CONCEPT/DEMO MODE, "
        "not actual Hugging Face TGI. See tgi/ for real TGI via Docker+GPU."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Loading at startup means the first user request isn't slowed down
    # by a cold model load.
    await run_in_threadpool(model_service.load_model)


@app.get("/")
def root():
    return {
        "project": "TGI LLM Serving Dashboard",
        "mode": "CPU/DEMO MODE (not actual Hugging Face TGI)",
        "model": model_service.get_model_name(),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model_service.is_loaded() else "loading",
        "model_loaded": model_service.is_loaded(),
        "model_name": model_service.get_model_name(),
        "device": config.DEVICE,
        "mode": "cpu_demo",
    }


@app.get("/metrics")
def get_metrics():
    return metrics.snapshot()


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is still loading")

    if req.stream:
        def event_stream():
            start = time.perf_counter()
            token_count = 0
            for piece in model_service.stream_generate(req.prompt, req.max_new_tokens):
                token_count += 1
                yield f"data: {json.dumps({'token': piece})}\n\n"
            latency = time.perf_counter() - start
            metrics.record_request(token_count, latency)
            yield f"data: {json.dumps({'done': True, 'latency_seconds': round(latency, 4)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await run_in_threadpool(
        model_service.generate, req.prompt, req.max_new_tokens, req.use_cache
    )
    metrics.record_request(result.new_tokens, result.latency_seconds)
    return GenerateResponse(
        prompt=req.prompt,
        generated_text=result.generated_text,
        new_tokens=result.new_tokens,
        latency_seconds=round(result.latency_seconds, 4),
        tokens_per_second=round(result.tokens_per_second, 2),
        use_cache=req.use_cache,
        model=model_service.get_model_name(),
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible endpoint, mirroring the shape TGI itself exposes
    at /v1/chat/completions so existing OpenAI-SDK client code can point
    at this server with minimal changes."""
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is still loading")

    # Keep a plain-text fallback for models that do not define a chat template.
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant:"
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        def event_stream():
            created = int(time.time())
            start = time.perf_counter()
            token_count = 0
            for piece in model_service.stream_generate(prompt, req.max_tokens, messages):
                token_count += 1
                chunk = {
                    "id": "chatcmpl-demo",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            latency = time.perf_counter() - start
            metrics.record_request(token_count, latency)
            final_chunk = {
                "id": "chatcmpl-demo",
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await run_in_threadpool(
        model_service.generate, prompt, req.max_tokens, True, messages
    )
    metrics.record_request(result.new_tokens, result.latency_seconds)
    return {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.generated_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "completion_tokens": result.new_tokens,
            "prompt_tokens": None,
            "total_tokens": None,
        },
    }


@app.post("/batch/demo")
async def batch_demo(req: BatchDemoRequest):
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is still loading")

    if req.mode == "sequential":
        results, overall_latency = await run_in_threadpool(
            model_service.sequential_batch, req.prompts, req.max_new_tokens
        )
        return {
            "mode": "sequential",
            "explanation": (
                "Each prompt is generated one at a time in its own forward pass — "
                "this is how a naive, non-batched server would handle concurrent requests."
            ),
            "requests": results,
            "total_latency_seconds": round(overall_latency, 4),
        }
    else:
        result = await batching.simulate_continuous_batching(req.prompts, req.max_new_tokens)
        result["mode"] = "batched"
        result["explanation"] = (
            "Requests arrive with staggered timing (simulating independent concurrent "
            "clients), are collected within a short scheduler window, and are then sent "
            "through ONE real batched forward pass — the same idea TGI's continuous "
            "batching scheduler uses to keep the GPU busy across many requests at once."
        )
        return result


@app.post("/demo/kv-cache")
async def kv_cache_demo(req: KvCacheDemoRequest):
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is still loading")
    result = await run_in_threadpool(
        model_service.kv_cache_comparison, req.prompt, req.max_new_tokens
    )
    result["explanation"] = (
        "use_cache=True reuses previously computed key/value attention tensors instead "
        "of recomputing them for every new token — this is the KV cache TGI relies on "
        "during decoding. use_cache=False recomputes attention over the whole sequence "
        "at every step, which is slower and is shown here only for comparison."
    )
    return result


@app.post("/demo/quantization")
async def quantization_demo(req: QuantizationDemoRequest):
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is still loading")
    result = await run_in_threadpool(
        model_service.quantization_comparison, req.prompt, req.max_new_tokens
    )
    return result
