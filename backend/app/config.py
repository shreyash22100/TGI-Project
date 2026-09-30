"""
config.py
---------
Central configuration, loaded from environment variables (with sensible
defaults) so no secrets or machine-specific paths are hard-coded in source.

Copy .env.example to .env and edit values there; python-dotenv loads it
automatically when the app starts.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the working directory, if present

# Which Hugging Face model to load in CPU/demo mode.
# Qwen2.5-0.5B-Instruct is small enough for CPU use and follows instructions.
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# Device: "cpu" by default. If a CUDA GPU is available and torch detects it,
# model_service.py will still respect this override.
DEVICE = os.getenv("DEVICE", "cpu")

# Default generation limits (kept small so CPU demo responses stay fast)
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "40"))
MAX_ALLOWED_NEW_TOKENS = int(os.getenv("MAX_ALLOWED_NEW_TOKENS", "200"))

# Continuous-batching demo window (seconds) — how long the simulated
# scheduler waits to collect concurrent requests into one batch.
BATCH_WINDOW_SECONDS = float(os.getenv("BATCH_WINDOW_SECONDS", "0.15"))

# CORS — wide open for a local student demo. Tighten this before deploying
# anywhere public.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
