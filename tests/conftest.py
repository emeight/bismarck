# tests/conftest.py

import pytest

from bismarck.routing import ModelRegistry, ProviderRouter


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure each test starts with a clean registry state."""
    ModelRegistry._models = {}
    yield
    ModelRegistry._models = {}


@pytest.fixture(autouse=True)
def reset_provider_router():
    """Ensure each test starts with a clean provider instance cache."""
    ProviderRouter._instances = {}
    yield
    ProviderRouter._instances = {}