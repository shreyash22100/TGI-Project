# TGI LLM Serving Dashboard

A working demonstration of how a large language model is served in
production using ideas from **Hugging Face Text Generation Inference
(TGI)** — continuous batching, KV caching, quantization, tensor
parallelism, token streaming, and an OpenAI-compatible API.

This project ships in **two modes**:

| Mode | What it is | Requirements |
|---|---|---|
| **CPU/Demo Mode** | A FastAPI + Transformers backend that reproduces TGI's *concepts* on an ordinary laptop, with a web dashboard | Python, CPU only |
| **Real TGI Mode** | Actual Hugging Face TGI, run via Docker | NVIDIA GPU + Docker |

**Only CPU/Demo Mode is expected to run on a normal student laptop
without a GPU.** Real TGI genuinely requires an NVIDIA GPU for its fast
paths (FlashAttention, GPU quantization kernels, tensor parallelism) —
see [`tgi/README.md`](tgi/README.md) for that path, meant for a GPU
machine or cloud GPU instance if/when one is available.

All numbers shown in the dashboard (latency, tokens/sec, request counts)
are measured live from real requests handled by the backend — nothing is
hard-coded or fabricated.

---

## Project Structure

```
tgi-llm-serving-dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app + all routes
│   │   ├── model_service.py Model loading, generation, streaming, KV-cache & quantization demos
│   │   ├── batching.py      Continuous-batching scheduler simulation
│   │   ├── metrics.py       In-memory request metrics
│   │   ├── schemas.py       Pydantic request/response models
│   │   └── config.py        Environment-driven settings
│   └── requirements.txt
├── frontend/
│   ├── index.html           Dashboard shell (sidebar + 6 sections)
│   ├── style.css
│   └── app.js                All frontend logic (no build step needed)
├── tgi/
│   ├── docker-compose.yml    Real TGI (GPU) service definition
│   └── README.md             GPU / quantization / multi-GPU instructions
├── tests/
│   └── test_api.py           Basic API smoke tests (pytest)
├── README.md                 (this file)
├── PROJECT_REPORT.md
├── VIVA.md
├── .env.example
└── .gitignore
```

---

## 1. Requirements

- Python 3.9–3.11
- Several GB of free disk space for PyTorch and the instruction-tuned model
- A modern browser
- No GPU required for CPU/Demo Mode

---

## 2. Installation & Setup (CPU/Demo Mode)

### Step 1 — Get the project onto your machine

If you received this as a zip, extract it. Then open a terminal inside
the `tgi-llm-serving-dashboard/` folder.

### Step 2 — Create a virtual environment

```bash
cd backend
python -m venv venv
```

Activate it:

- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source venv/bin/activate`

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

(This installs FastAPI, Uvicorn, Transformers, PyTorch (CPU build),
pydantic, python-dotenv, and pytest. It can take a few minutes the
first time.)

### Step 4 — Configure environment variables

```bash
cd ..
cp .env.example .env
```

The defaults (`Qwen/Qwen2.5-0.5B-Instruct` model, CPU device) work on a
normal laptop. The model follows instructions and produces much more
useful responses than a tiny base language model. You can change
`MODEL_NAME` in `.env` if you want to try another Transformers model.

### Step 5 — Start the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The first startup downloads `Qwen/Qwen2.5-0.5B-Instruct` from the Hugging
Face Hub (roughly 1 GB) — this only happens once; it's then cached
locally. Allow extra time for this download, then check:

```
http://localhost:8000/health
```

You should see `"model_loaded": true` once loading finishes.

### Step 6 — Start the frontend

The frontend is plain HTML/CSS/JS — no npm/build step required. From the
`frontend/` folder, serve it with Python's built-in server (this avoids
browser file:// restrictions on fetch calls):

```bash
cd ../frontend
python -m http.server 5500
```

Then open:

```
http://localhost:5500
```

The dashboard will connect to the backend at `http://localhost:8000`
(configured at the top of `frontend/app.js` as `API_BASE`).

---

## 3. Using the Dashboard

- **Dashboard** — live server status, model name, request count, average
  latency, average tokens/sec, uptime.
- **Text Generation** — type a prompt, toggle streaming on/off, and
  generate. With streaming on, tokens appear as the model produces them.
- **Batching Demo** — enter several prompts and compare *Sequential*
  (one request at a time) against *Batched* (grouped, single forward
  pass) processing, with real measured timings.
- **TGI Concepts** — short reference cards on continuous batching, KV
  cache, quantization, tensor parallelism, streaming, and the
  OpenAI-compatible API.
- **API Demo** — see the exact request/response shape for
  `/v1/chat/completions`, and call it live from the browser.
- **Architecture** — a diagram mapping this project's components to real
  TGI's router / launcher / model-server / multi-GPU architecture.

---

## 4. API Usage (curl examples)

```bash
# Health check
curl http://localhost:8000/health

# Plain (non-streaming) generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_new_tokens": 30, "stream": false}'

# Streaming generation (server-sent events)
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 40, "stream": true}'

# OpenAI-compatible chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 30}'

# Batching demo — sequential vs batched
curl -X POST http://localhost:8000/batch/demo \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hi there", "Tell me a fact"], "mode": "batched", "max_new_tokens": 20}'

# KV cache comparison
curl -X POST http://localhost:8000/demo/kv-cache \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 40}'

# Quantization comparison
curl -X POST http://localhost:8000/demo/quantization \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The weather today", "max_new_tokens": 30}'

# Live metrics
curl http://localhost:8000/metrics
```

Interactive API docs (Swagger UI) are also available automatically at:

```
http://localhost:8000/docs
```

---

## 5. Running Tests

```bash
cd backend
pytest ../tests/test_api.py -v
```

The first test run loads the model, so it will be a little slower than
subsequent runs.

---

## 6. Running Real TGI (GPU Mode)

See [`tgi/README.md`](tgi/README.md) for the full walkthrough, including
Docker Compose setup, quantization flags, and multi-GPU tensor
parallelism. In short (on a machine with an NVIDIA GPU and Docker + the
NVIDIA Container Toolkit installed):

```bash
cd tgi
docker compose up
```

TGI will then be reachable at `http://localhost:8080`.

---

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ModuleNotFoundError` on startup | Virtual environment not activated, or `pip install -r requirements.txt` not run |
| First request hangs for a long time | Model is still downloading — check the backend terminal log |
| Frontend shows "🔴 Offline" | Backend not running, or `API_BASE` in `app.js` doesn't match the backend's actual port |
| CORS error in browser console | Make sure you're serving the frontend via `python -m http.server`, not opening `index.html` directly with `file://` |
| Streaming output doesn't appear incrementally | Some browsers/extensions buffer `fetch` streams — try a different browser or disable ad-block/dev-tools throttling |
| `docker compose up` fails to see the GPU | NVIDIA Container Toolkit not installed, or Docker not configured with the `nvidia` runtime |
| Out-of-memory on CPU | Lower `max_new_tokens`, or send fewer prompts at once in the Batching Demo |
| `pip install` fails on `torch==...` or hangs "Preparing metadata" for a long time | You're likely on a very new Python version (e.g. 3.13/3.14) that doesn't yet have prebuilt wheels for every pinned package. `requirements.txt` in this project intentionally has **no version pins** for this reason — if you edited it back to pinned versions, remove the pins again. If a package still fails to build from source, the most reliable fix is installing Python 3.11 or 3.12 side-by-side (they have the widest wheel support for ML packages) and creating the virtual environment with that version instead, e.g. `py -3.12 -m venv venv` on Windows, then re-activate and reinstall |

---

## 8. Taking Screenshots for Submission

1. Start backend and frontend as above.
2. Open the dashboard and, in order, screenshot: Dashboard tab (with a
   few requests already made so the stats aren't all zero), Text
   Generation tab mid-stream, Batching Demo results, TGI Concepts grid,
   API Demo with a live response shown, and the Architecture diagram.
3. Save screenshots into a `screenshots/` folder (not included by
   default) and reference them in `PROJECT_REPORT.md`.

---

## 9. Honesty Notes

- CPU/Demo Mode is clearly labelled everywhere (API responses, dashboard
  badge, docs) as a **concept demonstration**, not actual TGI.
- All latency/throughput numbers are measured with `time.perf_counter()`
  around real model calls at request time — none are hard-coded.
- The Batching Demo's "continuous batching" is a simplified simulation of
  TGI's scheduler behaviour (staggered arrival + batch window), combined
  with a **real** batched forward pass — this distinction is stated in
  the API response itself (`explanation` field) and in
  `backend/app/batching.py`'s docstring.
- The Quantization demo uses PyTorch's CPU-native dynamic INT8
  quantization as a stand-in for TGI's GPU-only backends
  (bitsandbytes/GPTQ/AWQ/FP8), and says so explicitly in its response.
- Tensor parallelism cannot be demonstrated without multiple GPUs; it is
  explained conceptually and configured (but not run) in
  `tgi/docker-compose.yml`.
