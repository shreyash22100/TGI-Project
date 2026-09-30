# VIVA — Questions & Answers
## TGI LLM Serving Dashboard

**1. What is Text Generation Inference (TGI)?**
TGI is Hugging Face's open-source, production-grade toolkit for serving
large language models efficiently — handling concurrent requests,
batching them, managing GPU memory, and streaming responses.

**2. Is this project real TGI?**
No. The main dashboard runs in **CPU/Demo Mode**, a FastAPI +
Transformers backend that reproduces TGI's *concepts* without a GPU.
The `tgi/` folder documents how to run **actual** TGI via Docker on a
GPU machine.

**3. Why can't real TGI run on a normal laptop?**
Real TGI's fastest paths — FlashAttention kernels, GPU quantization
backends (bitsandbytes/GPTQ/AWQ/FP8), and tensor parallelism — require
an NVIDIA CUDA-capable GPU. Most student laptops don't have one.

**4. What is continuous batching?**
A scheduling technique where new requests are merged into the active
batch as soon as capacity allows, rather than waiting for the whole
batch to finish, keeping the GPU busy under mixed request lengths.

**5. How does this project demonstrate continuous batching?**
`batching.py` simulates staggered request arrival, collects requests
within a short time window, and then runs one real batched forward pass
over them — showing the scheduling idea, with the batched generation
itself being real and measured.

**6. What is the KV cache?**
Cached key/value attention tensors from previous tokens, reused instead
of being recomputed at every new decoding step — this is what makes
long text generation fast.

**7. How is the KV cache demonstrated here?**
The `/demo/kv-cache` endpoint runs the same prompt with `use_cache=True`
and `use_cache=False` and reports both real, measured latencies side by
side.

**8. What is quantization?**
Reducing the numerical precision of model weights (e.g., from 32-bit
floats to 8-bit integers) to shrink memory usage, usually with some
trade-off in speed or accuracy.

**9. Which quantization does this project actually run?**
PyTorch's CPU-native dynamic INT8 quantization
(`torch.quantization.quantize_dynamic`), used as a concept-equivalent
stand-in for TGI's GPU-only backends.

**10. Why not use bitsandbytes or GPTQ directly?**
Those quantization backends require an NVIDIA GPU with CUDA kernels;
they cannot run on CPU-only hardware.

**11. What is tensor parallelism?**
Splitting a single model's weight matrices across multiple GPUs so a
model too large for one GPU's memory can still be served, with GPUs
synchronising via NCCL during each layer's computation.

**12. Why isn't tensor parallelism actually demonstrated in this project?**
It requires multiple GPUs, which this project does not assume the user
has. It's explained conceptually and configured (but not executed) in
`tgi/docker-compose.yml`.

**13. What is token streaming, and how is it implemented here?**
Sending generated tokens to the client as soon as they're produced
rather than waiting for the full response. This project uses
`TextIteratorStreamer` running generation on a background thread, with
tokens yielded to the client over Server-Sent Events (SSE).

**14. What does "OpenAI-compatible API" mean?**
An endpoint whose request and response JSON shapes match OpenAI's own
API (e.g., `/v1/chat/completions` with a `messages` array and a
`choices[0].message.content` response), so existing client code written
for OpenAI can point at this server with minimal changes.

**15. What backend framework is used, and why?**
FastAPI — chosen for its async support (needed for streaming), automatic
request validation via Pydantic, and built-in interactive API docs.

**16. What model library is used for generation?**
Hugging Face `transformers`, with PyTorch as the underlying tensor
library.

**17. Which model does the demo use by default, and why?**
`Qwen/Qwen2.5-0.5B-Instruct` — a compact instruction-tuned model that
can run on CPU and follows prompts more usefully than a tiny base model.

**18. Can the model be changed?**
Yes, via the `MODEL_NAME` environment variable in `.env`.

**19. How does the frontend talk to the backend?**
Plain `fetch()` calls to the FastAPI backend's REST endpoints, including
reading a streamed response body for token-by-token output.

**20. Why is the frontend plain HTML/CSS/JS instead of React?**
To keep the project runnable without a build step (no npm install/build
required) while still being a complete, functional dashboard — simpler
and more reliable for a student demo.

**21. What does the `/metrics` endpoint report?**
Request count, total tokens generated, average and last-request latency,
and average/last tokens-per-second — all computed from real requests
handled since the server started.

**22. Is `/metrics` the same as TGI's real metrics?**
No. Real TGI exposes Prometheus-format metrics; this project uses a
simpler in-memory JSON snapshot that captures the same *idea* (request
count, latency, throughput) without needing a Prometheus server.

**23. What is the difference between sequential and batched processing in the Batching Demo?**
Sequential processes each prompt one at a time in its own forward pass.
Batched pads all prompts together and runs them through a single forward
pass with a batch dimension — closer to how a serving engine handles
concurrent requests.

**24. Why might the batched mode not show a huge speed-up on CPU?**
CPUs have much less parallel compute headroom than GPUs, so the benefit
of batching multiple sequences together is smaller than it would be on a
GPU — which is exactly why continuous batching matters most for GPU
serving.

**25. What HTTP status code does the API return for an invalid request (e.g., empty prompt)?**
422 Unprocessable Entity, raised automatically by Pydantic's request
validation in FastAPI.

**26. How does the project avoid hard-coding secrets?**
All configurable values (model name, device, CORS origins, ports) are
read from environment variables via `config.py` and a `.env` file (which
is git-ignored); `.env.example` documents the expected variables without
containing real secrets.

**27. What testing does the project include?**
`tests/test_api.py` uses FastAPI's `TestClient` to test `/health`,
`/metrics`, `/generate`, `/v1/chat/completions`, `/batch/demo`, and
`/demo/kv-cache` against the real backend.

**28. What are the router, launcher, and model server in real TGI's architecture?**
The **router** (Rust) receives and queues client requests and runs the
batching scheduler; the **launcher** starts the router and one or more
model-server processes with matching arguments; the **model server**
(Python) loads the model and performs the actual forward passes,
communicating with the router over gRPC.

**29. How does this project's architecture map onto that?**
The single FastAPI process plays the combined role of router + launcher,
and `model_service.py` plays the role of the model server — collapsed
into one process since the demo's purpose is to show the request
lifecycle, not to reproduce TGI's multi-process topology.

**30. What would need to change to run this project against real GPU-based TGI instead of the CPU demo?**
Point the frontend's `API_BASE` at the real TGI server's address (e.g.,
`http://localhost:8080` from `tgi/docker-compose.yml`), use
`/v1/chat/completions` (the endpoint whose shape is intentionally
identical in both modes), and adapt the metrics display to scrape real
TGI's Prometheus metrics instead of this project's simplified
`/metrics` JSON — details are in `tgi/README.md`.
