# Running Real Hugging Face TGI (GPU Mode)

The rest of this project (`backend/`, `frontend/`) is a **CPU/demo mode**
that reproduces TGI's *concepts* on a laptop without a GPU. This folder
is for actually running **real TGI** when you have access to an
NVIDIA GPU machine (a cloud GPU instance is fine — most student laptops
do not have a CUDA-capable GPU).

## Requirements

- An NVIDIA GPU with recent drivers
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so Docker containers can see the GPU
- (Optional) a Hugging Face account/token if the chosen model is gated

## 1. Basic run (single GPU, no quantization)

```bash
cd tgi
docker compose up
```

This uses the default model set in `docker-compose.yml`
(`meta-llama/Llama-3.2-1B-Instruct` — change `MODEL_ID` to any
TGI-supported model on the Hub). TGI will be reachable at:

```
http://localhost:8080
http://localhost:8080/v1/chat/completions   (OpenAI-compatible)
http://localhost:8080/generate               (native TGI API)
http://localhost:8080/health
```

## 2. Quantization

Edit `docker-compose.yml` and uncomment the `QUANTIZE` environment
variable / `--quantize` command flag, e.g.:

```yaml
command:
  - "--model-id"
  - "${MODEL_ID}"
  - "--quantize"
  - "bitsandbytes"     # or: gptq / awq / eetq / fp8 / marlin
```

- `bitsandbytes` quantizes on load — no pre-quantized checkpoint needed.
- `gptq` / `awq` / `exl2` / `marlin` require a model that has **already**
  been quantized in that format (point `MODEL_ID` at such a checkpoint on
  the Hub).

## 3. Multi-GPU / Tensor Parallelism

If the host has more than one GPU, shard the model across them:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2              # number of GPUs to expose to the container
          capabilities: [gpu]
command:
  - "--model-id"
  - "${MODEL_ID}"
  - "--num-shard"
  - "2"                         # split the model across 2 GPUs
```

## 4. Testing the real server

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tgi","messages":[{"role":"user","content":"Hello"}],"max_tokens":40}'
```

## 5. Pointing the dashboard frontend at real TGI

The CPU/demo backend's `/generate` and `/v1/chat/completions` schemas
were written to match TGI's own shapes closely, but they are not
byte-for-byte identical (TGI's native `/generate` endpoint, for example,
uses a slightly different JSON structure). To point `frontend/app.js` at
real TGI instead of the demo backend:

1. Change `API_BASE` in `frontend/app.js` to `http://localhost:8080`.
2. Use `/v1/chat/completions` (OpenAI-compatible) rather than `/generate`,
   since that endpoint's shape is intentionally the same in both modes.
3. Metrics: real TGI exposes Prometheus-format metrics rather than this
   project's simple `/metrics` JSON — the Dashboard tab's stat cards would
   need a small adapter to scrape and parse Prometheus text format instead.

This project intentionally keeps CPU/demo mode and real-TGI mode as two
separate, clearly labelled paths rather than silently mixing them, so
it's always clear which one produced a given number.
