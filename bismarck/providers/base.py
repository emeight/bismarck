# bismarck/providers/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Type
from pydantic import BaseModel

from bismarck.schema import LLMResponse, GenerationRequest, Message


class BaseProvider(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    def generate(
        self,
        messages: list[Message],
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] =None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        """Public entrypoint that for structured requests."""
        request = GenerationRequest(
            # copy to tuple so that request messages remain immutable
            messages=tuple(messages),
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            schema=schema
        )
        return self._generate(request)
    
    @abstractmethod
    def _generate(self, request: GenerationRequest) -> LLMResponse:
        """Normalized generation implemented by provider subclasses."""
        pass