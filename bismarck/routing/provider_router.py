# bismark/src/routing/provider_router.py

import threading
from types import MappingProxyType
from typing import Type

from bismarck.errors import UnknownProviderError
from bismarck.providers.base import BaseProvider
from bismarck.providers.openai import OpenAIProvider
from bismarck.providers.anthropic import AnthropicProvider
from bismarck.providers.google import GoogleProvider


class ProviderRouter:
    """Resolves provider names to cached, shared provider instances."""

    _providers: MappingProxyType[str, Type[BaseProvider]] = MappingProxyType({
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider
    })

    _instances: dict[str, BaseProvider] = {}
    _lock = threading.Lock()    # prevents race conditions

    @classmethod
    def get(cls, name):
        """Return a cached, shared instance for `name`."""

        instance = cls._instances.get(name)

        if instance is None:
            with cls._lock:
                # re-check inside the lock incase another thread created an instance while we were waiting
                instance = cls._instances.get(name)
                if instance is None:
                    try:
                        provider_cls = cls._providers[name]
                    except KeyError:
                        raise UnknownProviderError(name) from None
                    
                    instance = provider_cls()
                    cls._instances[name] = instance

        return instance
