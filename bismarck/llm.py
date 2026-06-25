# bismarck/llm.py

import time
import threading

from bismarck.schema import LLMResponse, Message
from bismarck.errors import AllCandidatesExcludedError, ProviderError, ModelError 
from bismarck.routing import ModelRegistry, ProviderRouter


class LLM:
    DEFAULT_PROVIDER_COOLDOWN_SECONDS: float = 30.

    def __init__(
            self,
            model: str,
            fallbacks: list[str] | None = None,
            provider_cooldown_seconds: float = DEFAULT_PROVIDER_COOLDOWN_SECONDS,
            ):
        self.spec = ModelRegistry.get(model)
        self.fallbacks = [
            ModelRegistry.get(name) for name in ModelRegistry.validate(model, fallbacks or [])
        ]

        if provider_cooldown_seconds < 0:
            raise ValueError(
                f"provider_cooldown_seconds must be non-negative, got: {provider_cooldown_seconds}"
            )
        self.provider_cooldown = provider_cooldown_seconds

        self._excluded_until: dict[str, float] = {} # provider name: unblock time
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        fallback_names = [f.name for f in self.fallbacks]
        excluded = [p for p in self._excluded_until if self._is_excluded(p)]
        excluded_part = f" excluded={excluded!r}" if excluded else ""
        return f"<LLM model={self.spec.name!r} fallbacks={fallback_names!r}{excluded_part}>"
    
    def reset(self) -> None:
        """Clear all provider exclusions."""
        with self._lock:
            self._excluded_until.clear()

    def _is_excluded(self, provider_name: str) -> bool:
        unblock_at = self._excluded_until.get(provider_name)
        return unblock_at is not None and time.monotonic() < unblock_at

    def _exclude_provider(self, provider_name: str) -> None:
        with self._lock:
            self._excluded_until[provider_name] = time.monotonic() + self.provider_cooldown

    def generate(
            self,
            user: str,
            system: str | None = None,
            schema=None,
            retry: bool = False,
        ) -> LLMResponse:
        messages = (Message(role="user", content=user),)
        candidates = [self.spec] + (self.fallbacks if retry else [])

        last_error: Exception | None = None
        attempted = False

        for candidate in candidates:
            if self._is_excluded(candidate.provider):
                continue

            attempted = True
            try:
                provider = ProviderRouter.get(candidate.provider)
                return provider.generate(
                    messages=messages,
                    system=system,
                    model=candidate.name,
                    schema=schema,
                )
            except ProviderError as e:
                # provider is unhealthy, skip all associated models
                self._exclude_provider(candidate.provider)
                last_error = e
            except ModelError as e:
                # model-level issue, move to another model
                last_error = e
        
        if not attempted:
            raise AllCandidatesExcludedError(c.provider for c in candidates)

        raise last_error