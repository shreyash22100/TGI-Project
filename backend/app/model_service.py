"""
model_service.py
-----------------
This is the "model server" half of the dashboard's CPU/demo mode — it is
the part of the project that stands in for TGI's Python model server
(see PROJECT_REPORT.md, Architecture section).

It is responsible for:
  * loading the Hugging Face model/tokenizer once (singleton)
  * plain (non-streaming) generation, with real measured latency
  * token-by-token streaming generation using TextIteratorStreamer
    (this is a genuine streaming implementation, not a fake typewriter
    effect — tokens are yielded as the model actually produces them)
  * a KV-cache on/off comparison (use_cache=True vs False), so the
    speed-up from caching can be *measured*, not just asserted
  * a CPU dynamic-quantization comparison. Real TGI quantization
    (bitsandbytes / GPTQ / AWQ) needs a CUDA GPU, which a student laptop
    usually doesn't have. torch.quantization.quantize_dynamic is the
    closest CPU-native equivalent: it also produces a smaller, INT8
    model with a real (if different) latency/size trade-off, so it is
    used here as a *concept-equivalent*, clearly labelled as such.

Every number this module returns is measured with time.perf_counter()
around real model calls — nothing here is a fabricated benchmark figure.
"""

import copy
import threading
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from . import config

_lock = threading.Lock()
_tokenizer = None
_model = None
_quantized_model = None  # lazily built the first time the quantization demo is used


@dataclass
class GenerationResult:
    generated_text: str
    new_tokens: int
    latency_seconds: float
    tokens_per_second: float


def load_model() -> None:
    """Loads tokenizer + model once. Safe to call multiple times."""
    global _tokenizer, _model
    with _lock:
        if _model is not None:
            return
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
        _model.to(config.DEVICE)
        _model.eval()


def is_loaded() -> bool:
    return _model is not None


def get_model_name() -> str:
    return config.MODEL_NAME


def _ensure_loaded():
    if _model is None:
        load_model()


def _format_prompt(prompt: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
    if not getattr(_tokenizer, "chat_template", None):
        return prompt
    chat_messages = messages or [{"role": "user", "content": prompt}]
    return _tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )


def generate(
    prompt: str,
    max_new_tokens: int,
    use_cache: bool = True,
    messages: Optional[List[Dict[str, str]]] = None,
) -> GenerationResult:
    """Plain, non-streaming generation with real timing."""
    _ensure_loaded()
    formatted_prompt = _format_prompt(prompt, messages)
    inputs = _tokenizer(formatted_prompt, return_tensors="pt").to(config.DEVICE)
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=_tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - start

    new_tokens = output_ids.shape[1] - input_len
    text = _tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    tps = new_tokens / latency if latency > 0 else 0.0

    return GenerationResult(
        generated_text=text,
        new_tokens=new_tokens,
        latency_seconds=latency,
        tokens_per_second=tps,
    )


def stream_generate(
    prompt: str,
    max_new_tokens: int,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """
    Real token streaming: runs generation on a background thread and
    yields each decoded piece as TextIteratorStreamer receives it. This
    mirrors what TGI does over Server-Sent Events, just over a Python
    generator that the FastAPI route turns into SSE.
    """
    _ensure_loaded()
    formatted_prompt = _format_prompt(prompt, messages)
    inputs = _tokenizer(formatted_prompt, return_tensors="pt").to(config.DEVICE)
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=_tokenizer.pad_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=_model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        if new_text:
            yield new_text

    thread.join()


def sequential_batch(prompts: List[str], max_new_tokens: int):
    """One-request-at-a-time baseline: each prompt gets its own forward
    call, run back-to-back. This is the 'normal inference' comparison
    point (no batching)."""
    _ensure_loaded()
    results = []
    overall_start = time.perf_counter()
    for p in prompts:
        r = generate(p, max_new_tokens=max_new_tokens)
        results.append({"prompt": p, "generated_text": r.generated_text, "latency_seconds": r.latency_seconds})
    overall_latency = time.perf_counter() - overall_start
    return results, overall_latency


def batched_generate(prompts: List[str], max_new_tokens: int):
    """
    True batched forward pass: all prompts are padded and generated in a
    single model.generate() call with a batch dimension, similar in
    spirit to how TGI's continuous batching packs multiple in-flight
    requests into one forward pass rather than serving them one at a
    time. On CPU the speed-up will be modest (a CPU has far less
    parallel compute headroom than a GPU, which is exactly why TGI's
    continuous batching matters most on GPU servers) — but the
    comparison is real and measured, not simulated.
    """
    _ensure_loaded()
    _tokenizer.padding_side = "left"  # required for correct generation with padded batches
    formatted_prompts = [_format_prompt(prompt) for prompt in prompts]
    inputs = _tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(config.DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=_tokenizer.pad_token_id,
        )
    overall_latency = time.perf_counter() - start

    results = []
    input_len = inputs["input_ids"].shape[1]
    for i, p in enumerate(prompts):
        text = _tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
        results.append({"prompt": p, "generated_text": text, "latency_seconds": None})

    return results, overall_latency


def kv_cache_comparison(prompt: str, max_new_tokens: int):
    """Runs the same prompt with use_cache=True and use_cache=False and
    returns both real, measured latencies side by side."""
    with_cache = generate(prompt, max_new_tokens=max_new_tokens, use_cache=True)
    without_cache = generate(prompt, max_new_tokens=max_new_tokens, use_cache=False)
    return {
        "with_cache": {
            "latency_seconds": round(with_cache.latency_seconds, 4),
            "tokens_per_second": round(with_cache.tokens_per_second, 2),
        },
        "without_cache": {
            "latency_seconds": round(without_cache.latency_seconds, 4),
            "tokens_per_second": round(without_cache.tokens_per_second, 2),
        },
    }


def _model_size_mb(model) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 * 1024)


def _get_quantized_model():
    global _quantized_model
    _ensure_loaded()
    if _quantized_model is None:
        with _lock:
            if _quantized_model is None:
                fp32_copy = copy.deepcopy(_model)
                _quantized_model = torch.quantization.quantize_dynamic(
                    fp32_copy, {torch.nn.Linear}, dtype=torch.qint8
                )
                _quantized_model.eval()
    return _quantized_model


def quantization_comparison(prompt: str, max_new_tokens: int):
    """
    Compares the full-precision model against a CPU INT8 dynamically
    quantized copy: real model size (MB) and real measured latency for
    both. Labelled clearly in the API response as a CPU-native stand-in
    for TGI's GPU quantization backends (bitsandbytes/GPTQ/AWQ).
    """
    _ensure_loaded()
    formatted_prompt = _format_prompt(prompt)
    inputs = _tokenizer(formatted_prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # Full precision
    start = time.perf_counter()
    with torch.no_grad():
        fp32_ids = _model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
        )
    fp32_latency = time.perf_counter() - start
    fp32_text = _tokenizer.decode(fp32_ids[0][input_len:], skip_special_tokens=True)
    fp32_size = _model_size_mb(_model)

    # Dynamically quantized (INT8) copy
    q_model = _get_quantized_model()
    start = time.perf_counter()
    with torch.no_grad():
        int8_ids = q_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
        )
    int8_latency = time.perf_counter() - start
    int8_text = _tokenizer.decode(int8_ids[0][input_len:], skip_special_tokens=True)
    int8_size = _model_size_mb(q_model)

    return {
        "note": (
            "TGI's own quantization backends (bitsandbytes/GPTQ/AWQ/FP8) require an "
            "NVIDIA GPU. This comparison uses PyTorch's CPU-native dynamic INT8 "
            "quantization as a concept-equivalent demonstration of the same idea: "
            "lower-precision weights, smaller memory footprint."
        ),
        "full_precision": {
            "latency_seconds": round(fp32_latency, 4),
            "approx_model_size_mb": round(fp32_size, 2),
            "generated_text": fp32_text,
        },
        "int8_dynamic_quantized": {
            "latency_seconds": round(int8_latency, 4),
            "approx_model_size_mb": round(int8_size, 2),
            "generated_text": int8_text,
        },
    }
