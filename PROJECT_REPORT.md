# PROJECT REPORT
## TGI LLM Serving Dashboard — A Practical Implementation of Text Generation Inference Concepts

**Name:** Shreyash Damdar
**Roll No:** 45
**Section:** B
**Semester:** VII
**Branch:** Artificial Intelligence
**Subject:** Large Language Model (UNAIL411)
**Subject Teacher:** Dr. Pranay Saraf
**College:** G H Raisoni College of Engineering
**Academic Session:** Winter-2026

---

## 1. Introduction

Text Generation Inference (TGI) is Hugging Face's production-grade
toolkit for serving large language models: it accepts many concurrent
requests, schedules them efficiently onto a GPU, and streams generated
text back to clients with low latency and high throughput. This project
turns the theoretical study of TGI (covered in the accompanying TAE-I
report on LangSmith-style evaluation concepts and TGI's own
architecture) into a **working, runnable system**: a small FastAPI
backend that reproduces TGI's core serving ideas on an ordinary CPU, a
web dashboard to interact with and observe it, and a documented path to
run **actual** TGI on a GPU machine via Docker.

The guiding principle throughout this implementation is honesty about
what is and isn't real TGI: every place the CPU/demo mode stands in for
a GPU-only TGI feature, that substitution is explicitly labelled, both
in the code's docstrings and in the API's own JSON responses.

---

## 2. Project Objective

To build a demonstrable system that shows, with real measured behaviour
rather than only diagrams, **why** each TGI serving mechanism matters:

- Continuous batching vs. one-request-at-a-time serving
- KV caching vs. recomputing attention every step
- Quantization's memory/latency trade-off
- Token streaming vs. waiting for a full response
- An OpenAI-compatible API surface
- (Conceptually) tensor parallelism and multi-GPU serving, which needs
  hardware this project does not assume the user has

---

## 3. System Architecture

### 3.1 Real TGI Architecture (for reference)

```
User → Router (Rust: queueing, tokenisation, continuous-batching
scheduler, OpenAI-compatible HTTP API) → Launcher-managed Model
Server(s) (Python/PyTorch, gRPC) → LLM weights on GPU(s), NCCL-
synchronised when tensor-parallel → streamed tokens back to User
```

### 3.2 This Project's CPU/Demo Architecture

```
User (browser)
   │  HTTP / fetch (SSE for streaming)
   ▼
Frontend (index.html + app.js) — dashboard UI, no build step
   │  HTTP requests to API_BASE
   ▼
FastAPI app (backend/app/main.py) — plays the role of TGI's Router + Launcher
   │  function calls (in-process, not gRPC — no separate process needed for a demo)
   ▼
model_service.py — plays the role of TGI's Python Model Server
   │
   ▼
Hugging Face Transformers model (Qwen2.5-0.5B-Instruct) running on CPU
```

The router/launcher/model-server separation that real TGI uses across
processes (for scaling and fault isolation) is collapsed into a single
Python process here, since the demo's goal is to show the *request
lifecycle and scheduling behaviour*, not to reproduce TGI's process
topology at small scale.

### 3.3 Component Responsibilities

| Component | File(s) | Responsibility |
|---|---|---|
| API / Router | `backend/app/main.py` | Request validation, routing, SSE streaming, response shaping |
| Scheduler (simplified) | `backend/app/batching.py` | Simulates staggered request arrival + batch-window collection |
| Model Server | `backend/app/model_service.py` | Model loading, generation, KV-cache toggle, quantization comparison |
| Metrics | `backend/app/metrics.py` | Thread-safe in-memory counters, mirrors what Prometheus tracks in real TGI |
| Config | `backend/app/config.py` | Environment-driven settings, no hard-coded secrets |
| Frontend | `frontend/*` | Dashboard, generation UI, batching UI, concept reference, API demo, architecture diagram |

---

## 4. Implementation Details

### 4.1 Model Choice

`Qwen/Qwen2.5-0.5B-Instruct` was chosen as the default demo model because
it is compact enough for CPU inference while being instruction-tuned for
useful responses. The service applies the tokenizer's chat template to
prompts and conversations. The model remains configurable via the
`MODEL_NAME` environment variable.

### 4.2 Streaming

Streaming is implemented with `transformers.TextIteratorStreamer`
running the model's `.generate()` call on a background thread, while the
main thread yields each decoded text piece as it becomes available. The
FastAPI route wraps this generator in a `StreamingResponse` with
`text/event-stream` media type, and the frontend reads the response body
as a `ReadableStream`, parsing each `data: {...}` line as it arrives.
This is genuine token-by-token streaming, not a simulated typing effect.

### 4.3 Continuous Batching Simulation

Reproducing TGI's exact iteration-level scheduler (which re-evaluates
batch membership after every single decoding step) is impractical
around a single high-level `model.generate()` call. Instead,
`batching.py` simulates realistic **staggered request arrival** using
`asyncio.sleep`, collects whatever arrives within a short
`BATCH_WINDOW_SECONDS` window, and then performs one **real** padded
batched forward pass over that group via `model_service.batched_generate()`.
The arrival timing is simulated; the batched generation and its latency
are measured and real. This distinction is stated explicitly in the
`/batch/demo` response's `explanation` field.

### 4.4 KV Cache Demonstration

`model_service.kv_cache_comparison()` runs the identical prompt twice —
once with `use_cache=True` and once with `use_cache=False` — and reports
both real, measured latencies. This lets a user see the actual speed
contribution of KV caching rather than only reading about it.

### 4.5 Quantization Demonstration

TGI's own quantization backends (bitsandbytes, GPTQ, AWQ, FP8, Marlin)
require an NVIDIA GPU. Since this project must run on a CPU-only student
laptop, `model_service.quantization_comparison()` instead uses PyTorch's
CPU-native `torch.quantization.quantize_dynamic`, producing a real INT8
copy of the model's linear layers. It reports real measured latency and
approximate model size (in MB) for both the full-precision and quantized
models. The API response includes an explicit `note` field stating this
is a CPU concept-equivalent, not TGI's own quantization path.

### 4.6 OpenAI-Compatible API

`/v1/chat/completions` accepts the same request shape (messages array,
model, max_tokens, stream) that OpenAI's API and real TGI's compatible
endpoint accept, and returns a response in the same
`choices[0].message.content` shape, including a streaming
delta-chunk variant terminated with `data: [DONE]`.

### 4.7 Metrics

`metrics.py` keeps thread-safe, in-memory counters (request count, total
tokens generated, total and average latency, average tokens/sec, last
request's figures) updated after every real generation call. This
mirrors, in spirit, what Prometheus scraping does for real TGI, without
requiring a Prometheus server for a small demo.

---

## 5. Testing

`tests/test_api.py` uses FastAPI's `TestClient` to verify, against the
real backend (not mocks):

- `/health` reports `model_loaded: true` after startup
- `/metrics` returns the expected fields
- `/generate` returns non-empty text with positive latency for a valid
  prompt, and rejects an empty prompt with a 422 validation error
- `/v1/chat/completions` returns a correctly shaped OpenAI-style response
- `/batch/demo` in batched mode returns one timeline entry per submitted
  prompt
- `/demo/kv-cache` returns both `with_cache` and `without_cache` timing
  blocks

Run with `pytest ../tests/test_api.py -v` from the `backend/` directory.

---

## 6. Measured Results

*(This section intentionally does not pre-fill numbers. Run the steps
below on your own machine and paste your own real output — every number
you get will differ by hardware, so no fabricated figures are given
here.)*

To fill this section for submission:

1. Start the backend (`uvicorn app.main:app --port 8000`).
2. Call each demo endpoint with `curl` or the dashboard UI and record the
   JSON response, e.g.:
   ```bash
   curl -X POST http://localhost:8000/demo/kv-cache \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_new_tokens": 60}'
   ```
3. Paste the real `with_cache` / `without_cache` latency figures here.
4. Repeat for `/demo/quantization` and `/batch/demo` (sequential vs
   batched), and paste those real figures here too.
5. After a short demo session, call `/metrics` and paste the aggregate
   request count / average latency / average tokens-per-second.

**[ Insert Screenshot: Dashboard tab with live metrics ]**
**[ Insert Screenshot: Text Generation tab, streaming mid-response ]**
**[ Insert Screenshot: Batching Demo — sequential vs batched results ]**
**[ Insert Screenshot: API Demo tab with a live response ]**

---

## 7. Discussion

Two points consistently came up while building this project, and they
match the theoretical TAE-I report's conclusions about TGI:

1. **The scheduling/serving layer is a separate concern from model
   quality.** Every demo in this project uses the same small model;
   what changes across "sequential vs. batched" or "cache vs. no cache"
   is purely how the same model is *served*, not what it knows.
2. **CPU hardware does not showcase GPU-oriented techniques well.** The
   batched-forward-pass demo shows a real but modest improvement on CPU,
   because CPUs have far less parallel throughput headroom than GPUs —
   which is exactly why TGI's continuous batching was designed for GPU
   serving in the first place. This is an honest limitation of the demo,
   not a flaw hidden from the reader.

---

## 8. Limitations of This Implementation

- CPU/demo mode cannot demonstrate tensor parallelism (no multi-GPU
  hardware assumed); it is explained and configured, but not run.
- The continuous-batching demo simulates request arrival timing rather
  than implementing TGI's true per-token, iteration-level admission
  scheduler.
- Quantization uses a CPU-native equivalent (dynamic INT8), not TGI's
  actual bitsandbytes/GPTQ/AWQ/FP8 GPU kernels.
- The demo model (`Qwen/Qwen2.5-0.5B-Instruct`) is far smaller than the models TGI is
  typically used to serve in production, so absolute latency/throughput
  numbers from this project are not representative of production-scale
  serving — only the *relative* comparisons (cached vs. uncached,
  sequential vs. batched, fp32 vs. quantized) are meaningful.
- Metrics are in-memory and reset when the backend restarts; real TGI's
  Prometheus metrics can be scraped and retained externally.

---

## 9. Conclusion

This project converts the theoretical study of Hugging Face TGI into a
runnable system that demonstrates, with genuinely measured behaviour,
why each of TGI's core serving mechanisms — continuous batching, KV
caching, quantization, streaming, and an OpenAI-compatible API — exists
and what it changes about serving performance. Where a mechanism cannot
be honestly reproduced on CPU-only student hardware (tensor parallelism,
GPU quantization kernels), the project says so explicitly rather than
faking the result, and provides a documented, ready-to-use path
(`tgi/`) to run real TGI on GPU hardware when available.

---

## 10. References

Hugging Face. *Text Generation Inference Documentation*.
huggingface.co/docs/text-generation-inference, 2025–2026.

Hugging Face. *Text Generation Inference — Architecture*.
huggingface.co/docs/text-generation-inference/architecture, 2025–2026.

Hugging Face. *Text Generation Inference — Quantization*.
huggingface.co/docs/text-generation-inference/conceptual/quantization,
2025–2026.

Hugging Face. *transformers* library documentation —
`TextIteratorStreamer`, `generate()`, `quantize_dynamic`.
huggingface.co/docs/transformers, 2025–2026.

FastAPI documentation. fastapi.tiangolo.com, 2025–2026.

PyTorch documentation — Dynamic Quantization.
pytorch.org/docs/stable/quantization.html, 2025–2026.
