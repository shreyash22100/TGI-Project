"""
metrics.py
----------
A minimal, thread-safe, in-memory metrics store. Real TGI exposes
Prometheus metrics; this module keeps the same *idea* (request count,
latency, tokens/sec) without needing a Prometheus server for the demo.

Every number here comes from an actual measured request handled by this
process since it started — nothing is simulated or hard-coded.
"""

import threading
import time

_lock = threading.Lock()

_state = {
    "start_time": time.time(),
    "request_count": 0,
    "total_tokens_generated": 0,
    "total_latency_seconds": 0.0,
    "last_request_latency": None,
    "last_tokens_per_second": None,
}


def record_request(new_tokens: int, latency_seconds: float) -> None:
    with _lock:
        _state["request_count"] += 1
        _state["total_tokens_generated"] += new_tokens
        _state["total_latency_seconds"] += latency_seconds
        _state["last_request_latency"] = latency_seconds
        _state["last_tokens_per_second"] = (
            new_tokens / latency_seconds if latency_seconds > 0 else 0.0
        )


def snapshot() -> dict:
    with _lock:
        uptime = time.time() - _state["start_time"]
        avg_latency = (
            _state["total_latency_seconds"] / _state["request_count"]
            if _state["request_count"] > 0
            else 0.0
        )
        avg_tps = (
            _state["total_tokens_generated"] / _state["total_latency_seconds"]
            if _state["total_latency_seconds"] > 0
            else 0.0
        )
        return {
            "uptime_seconds": round(uptime, 2),
            "request_count": _state["request_count"],
            "total_tokens_generated": _state["total_tokens_generated"],
            "average_latency_seconds": round(avg_latency, 4),
            "average_tokens_per_second": round(avg_tps, 2),
            "last_request_latency_seconds": (
                round(_state["last_request_latency"], 4)
                if _state["last_request_latency"] is not None
                else None
            ),
            "last_tokens_per_second": (
                round(_state["last_tokens_per_second"], 2)
                if _state["last_tokens_per_second"] is not None
                else None
            ),
        }
