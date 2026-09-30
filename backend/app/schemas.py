"""
schemas.py
----------
Pydantic models used to validate request bodies and shape responses.
Keeping these in one file makes it easy to see the whole API "contract"
at a glance.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(40, ge=1, le=200)
    use_cache: bool = Field(True, description="Toggle KV cache on/off (for the KV-cache demo)")
    stream: bool = Field(False, description="If true, response is text/event-stream")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    new_tokens: int
    latency_seconds: float
    tokens_per_second: float
    use_cache: bool
    model: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "tgi-demo"
    messages: List[ChatMessage]
    max_tokens: int = Field(40, ge=1, le=200)
    stream: bool = False
    temperature: float = 0.8


class BatchDemoRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=8)
    mode: Literal["sequential", "batched"] = "batched"
    max_new_tokens: int = Field(30, ge=1, le=100)


class KvCacheDemoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_new_tokens: int = Field(60, ge=1, le=200)


class QuantizationDemoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_new_tokens: int = Field(30, ge=1, le=150)
