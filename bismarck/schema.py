# bismarck/schema.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type, Literal, get_args
from pydantic import BaseModel


ModelCapability = Literal["tools", "vision", "streaming", "structured_output"]
MODEL_CAPABILITIES: frozenset[str] = frozenset(get_args(ModelCapability))

@dataclass(frozen=True)
class ModelPricing:
    input: float
    output: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    context_window: int
    pricing: ModelPricing
    capabilities: frozenset[ModelCapability]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    messages: tuple[Message, ...]
    system: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    schema: Optional[Type[BaseModel]] = None

    def __post_init__(self):
        if not self.messages:
            raise ValueError("Request must include messages.")
        if self.messages[0].role != "user":
            raise ValueError(f"Messages must start with a user message, got: {self.messages[0].role}")
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"Temperature out of range: {self.temperature}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got: {self.max_tokens}")


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    raw: Optional[str]
    data: Optional[Any]
    model: str
    usage: ModelUsage
