# bismarck/tests/routing/test_provider_router.py

import threading
from types import MappingProxyType

import pytest

from bismarck.routing import ProviderRouter
from bismarck.errors import UnknownProviderError


class FakeProviderA:
    pass


class FakeProviderB:
    pass


class CountingProvider:
    """Tracks how many times it's actually been instantiated."""
    instances_created = 0

    def __init__(self):
        CountingProvider.instances_created += 1


@pytest.fixture
def fake_providers(monkeypatch):
    providers = MappingProxyType({
        "fake-a": FakeProviderA,
        "fake-b": FakeProviderB,
        "counting": CountingProvider,
    })
    monkeypatch.setattr(ProviderRouter, "_providers", providers)
    CountingProvider.instances_created = 0
    yield


# ---------------------------------------------------------------------------
# basic resolution / caching
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_instance_of_correct_class(self, fake_providers):
        instance = ProviderRouter.get("fake-a")
        assert isinstance(instance, FakeProviderA)

    def test_repeated_calls_return_the_same_cached_instance(self, fake_providers):
        first = ProviderRouter.get("fake-a")
        second = ProviderRouter.get("fake-a")
        assert first is second

    def test_different_providers_get_independent_instances(self, fake_providers):
        a = ProviderRouter.get("fake-a")
        b = ProviderRouter.get("fake-b")
        assert a is not b
        assert isinstance(a, FakeProviderA)
        assert isinstance(b, FakeProviderB)

    def test_unknown_provider_raises(self, fake_providers):
        with pytest.raises(UnknownProviderError):
            ProviderRouter.get("does_not_exist")

    def test_unknown_provider_does_not_pollute_cache(self, fake_providers):
        with pytest.raises(UnknownProviderError):
            ProviderRouter.get("does_not_exist")
        assert "does_not_exist" not in ProviderRouter._instances

    def test_provider_is_only_instantiated_once_across_multiple_get_calls(self, fake_providers):
        ProviderRouter.get("counting")
        ProviderRouter.get("counting")
        ProviderRouter.get("counting")
        assert CountingProvider.instances_created == 1


# ---------------------------------------------------------------------------
# thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_first_access_instantiates_only_once(self, fake_providers):
        results = []

        def call():
            results.append(ProviderRouter.get("counting"))

        threads = [threading.Thread(target=call) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # exactly one underlying instance was ever constructed...
        assert CountingProvider.instances_created == 1
        # ...and every thread got that same instance back
        assert len({id(r) for r in results}) == 1

    def test_concurrent_access_to_different_providers_does_not_interfere(self, fake_providers):
        results = {"fake-a": [], "fake-b": []}

        def call(name):
            results[name].append(ProviderRouter.get(name))

        threads = [
            threading.Thread(target=call, args=(name,))
            for name in ("fake-a", "fake-b") * 25
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len({id(r) for r in results["fake-a"]}) == 1
        assert len({id(r) for r in results["fake-b"]}) == 1


# ---------------------------------------------------------------------------
# default registration / immutability contract
# ---------------------------------------------------------------------------

class TestDefaultProviders:
    def test_default_providers_are_registered(self):
        from bismarck.providers.openai import OpenAIProvider
        from bismarck.providers.anthropic import AnthropicProvider
        from bismarck.providers.google import GoogleProvider

        assert dict(ProviderRouter._providers) == {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
        }

    def test_providers_mapping_is_immutable(self):
        with pytest.raises(TypeError):
            ProviderRouter._providers["new"] = object