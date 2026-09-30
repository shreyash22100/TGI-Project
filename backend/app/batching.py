"""
batching.py
-----------
A deliberately simplified stand-in for TGI's router/scheduler, used by the
"Batching Demo" page to show *why* continuous batching helps, not just
assert that it does.

TGI's real scheduler re-evaluates the active batch after every decoding
step and admits new requests mid-flight (see PROJECT_REPORT.md). Building
that exact iteration-level scheduler around a single Transformers
`generate()` call is impractical for a CPU demo (Transformers' `generate`
does not expose per-token batch-admission hooks in a simple way), so this
module simulates staggered request arrival and shows the *scheduling*
behaviour — requests queue, get grouped into a batch window, and are then
sent together — while the actual token generation for the grouped batch
still calls the real `batched_generate()` in model_service.py, so the
generated text and the overall latency are real, measured values.

This distinction (simulated arrival timing + real batched generation) is
stated explicitly in the API response and in PROJECT_REPORT.md so it is
never presented as more than it is.
"""

import asyncio
import time
from typing import List

from . import config, model_service


async def simulate_continuous_batching(prompts: List[str], max_new_tokens: int) -> dict:
    """
    Simulates requests arriving with small staggered delays (as real
    concurrent users would), collects everything that arrives inside a
    BATCH_WINDOW_SECONDS window, then runs one real batched forward pass
    over whatever arrived in that window.
    """
    timeline = []
    arrival_start = time.perf_counter()

    async def arrive(i, prompt):
        # stagger arrivals by a few hundredths of a second to emulate
        # independent concurrent clients rather than one caller
        await asyncio.sleep(0.02 * i)
        t = time.perf_counter() - arrival_start
        timeline.append({"request_id": i, "prompt": prompt, "arrival_time_seconds": round(t, 4)})

    await asyncio.gather(*[arrive(i, p) for i, p in enumerate(prompts)])

    # wait out the batch-collection window (mirrors a scheduler tick)
    await asyncio.sleep(config.BATCH_WINDOW_SECONDS)

    batch_start = time.perf_counter()
    results, batch_latency = model_service.batched_generate(prompts, max_new_tokens=max_new_tokens)
    batch_end = time.perf_counter()

    for i, r in enumerate(results):
        timeline[i]["generated_text"] = r["generated_text"]
        timeline[i]["batch_started_at_seconds"] = round(batch_start - arrival_start, 4)
        timeline[i]["batch_completed_at_seconds"] = round(batch_end - arrival_start, 4)

    return {
        "batch_window_seconds": config.BATCH_WINDOW_SECONDS,
        "requests_in_batch": len(prompts),
        "total_batch_latency_seconds": round(batch_latency, 4),
        "timeline": timeline,
    }
