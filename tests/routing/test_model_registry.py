# bismarck/tests/routing/test_model_registry.py

import pytest

import yaml as yaml_module

from bismarck.routing import ModelRegistry
from bismarck.errors import UnknownModelError, UnknownCapabilityError
from bismarck.schema import ModelSpec, ModelPricing, MODEL_CAPABILITIES


def make_spec(name, capabilities=frozenset(), context_window=8000):
    return ModelSpec(
        name=name,
        provider="test-provider",
        context_window=context_window,
        pricing=ModelPricing(input=0.0, output=0.0),
        capabilities=frozenset(capabilities),
    )


@pytest.fixture
def populated_registry():
    ModelRegistry._models = {
        "primary": make_spec("primary", {"tools", "streaming"}),
        "full_fallback": make_spec("full_fallback", {"tools", "streaming", "vision"}),
        "exact_fallback": make_spec("exact_fallback", {"tools", "streaming"}),
        "partial_fallback": make_spec("partial_fallback", {"tools"}),
        "no_caps_fallback": make_spec("no_caps_fallback", set()),
        "no_caps_primary": make_spec("no_caps_primary", set()),
    }
    return ModelRegistry


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_spec_for_known_model(self, populated_registry):
        spec = populated_registry.get("primary")
        assert spec.name == "primary"
        assert spec.capabilities == {"tools", "streaming"}

    def test_raises_for_unknown_model(self, populated_registry):
        with pytest.raises(UnknownModelError):
            populated_registry.get("does_not_exist")

    def test_triggers_load_when_models_empty(self, monkeypatch):
        ModelRegistry._models = {}

        def fake_load(cls):
            cls._models = {"loaded-model": make_spec("loaded-model")}

        monkeypatch.setattr(ModelRegistry, "load", classmethod(fake_load))

        spec = ModelRegistry.get("loaded-model")
        assert spec.name == "loaded-model"

    def test_does_not_reload_when_models_already_populated(self, monkeypatch):
        ModelRegistry._models = {"already-here": make_spec("already-here")}

        def fail_if_called(pkg):
            raise AssertionError("files() should not be called when already loaded")

        monkeypatch.setattr("bismarck.routing.model_registry.files", fail_if_called)

        spec = ModelRegistry.get("already-here")
        assert spec.name == "already-here"


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_populates_models_from_yaml(self, tmp_path, monkeypatch):
        config = {
            "models": {
                "test-model": {
                    "provider": "test-provider",
                    "context_window": 4096,
                    "pricing": {"input": 0.001, "output": 0.002},
                    "capabilities": {
                        "tools": True,
                        "vision": False,
                        "streaming": True,
                        "structured_output": False,
                    },
                },
            }
        }
        (tmp_path / "models.yaml").write_text(yaml_module.dump(config))
        monkeypatch.setattr(
            "bismarck.routing.model_registry.files", lambda pkg: tmp_path
        )

        ModelRegistry.load()

        assert set(ModelRegistry._models) == {"test-model"}
        spec = ModelRegistry._models["test-model"]
        assert spec.name == "test-model"
        assert spec.provider == "test-provider"
        assert spec.context_window == 4096
        assert spec.pricing.input == 0.001
        assert spec.pricing.output == 0.002
        assert spec.capabilities == {"tools", "streaming"}

    def test_multiple_models_load_independently(self, tmp_path, monkeypatch):
        config = {
            "models": {
                "model-a": {
                    "provider": "provider-a",
                    "context_window": 1000,
                    "pricing": {"input": 0.1, "output": 0.2},
                    "capabilities": {
                        "tools": True,
                        "vision": False,
                        "streaming": False,
                        "structured_output": False,
                    },
                },
                "model-b": {
                    "provider": "provider-b",
                    "context_window": 2000,
                    "pricing": {"input": 0.3, "output": 0.4},
                    "capabilities": {
                        "tools": False,
                        "vision": True,
                        "streaming": True,
                        "structured_output": True,
                    },
                },
            }
        }
        (tmp_path / "models.yaml").write_text(yaml_module.dump(config))
        monkeypatch.setattr(
            "bismarck.routing.model_registry.files", lambda pkg: tmp_path
        )

        ModelRegistry.load()

        assert set(ModelRegistry._models) == {"model-a", "model-b"}
        assert ModelRegistry._models["model-a"].capabilities == {"tools"}
        assert ModelRegistry._models["model-b"].capabilities == {
            "vision", "streaming", "structured_output"
        }

    def test_is_idempotent_and_does_not_reread_file(self, tmp_path, monkeypatch):
        ModelRegistry._models = {"existing": make_spec("existing")}

        def fail_if_called(pkg):
            raise AssertionError("files() should not be called when already loaded")

        monkeypatch.setattr("bismarck.routing.model_registry.files", fail_if_called)

        ModelRegistry.load()

        assert list(ModelRegistry._models) == ["existing"]

    def test_real_config_loads_and_has_well_formed_specs(self):
        """Smoke test against the actual production data/models.yaml."""
        ModelRegistry._models = {}
        ModelRegistry.load()

        assert len(ModelRegistry._models) > 0
        for name, spec in ModelRegistry._models.items():
            assert spec.name == name
            assert isinstance(spec.context_window, int) and spec.context_window > 0
            assert spec.pricing.input >= 0
            assert spec.pricing.output >= 0
            assert isinstance(spec.capabilities, frozenset)
            assert spec.capabilities <= MODEL_CAPABILITIES


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:
    def test_fallback_with_all_required_capabilities_is_valid(self, populated_registry):
        result = populated_registry.validate("primary", ["full_fallback"])
        assert result == ["full_fallback"]

    def test_fallback_with_exact_capability_match_is_valid(self, populated_registry):
        result = populated_registry.validate("primary", ["exact_fallback"])
        assert result == ["exact_fallback"]

    def test_fallback_missing_required_capability_is_excluded(self, populated_registry):
        result = populated_registry.validate("primary", ["partial_fallback"])
        assert result == []

    def test_fallback_with_no_capabilities_is_excluded(self, populated_registry):
        result = populated_registry.validate("primary", ["no_caps_fallback"])
        assert result == []

    def test_primary_with_no_required_capabilities_accepts_any_fallback(self, populated_registry):
        # an empty requirement set is trivially satisfied by anything
        result = populated_registry.validate("no_caps_primary", ["no_caps_fallback", "partial_fallback"])
        assert result == ["no_caps_fallback", "partial_fallback"]

    def test_unknown_fallback_is_silently_skipped(self, populated_registry):
        result = populated_registry.validate("primary", ["does_not_exist"])
        assert result == []

    def test_empty_fallback_list_returns_empty_list(self, populated_registry):
        result = populated_registry.validate("primary", [])
        assert result == []

    def test_preserves_fallback_order(self, populated_registry):
        result = populated_registry.validate(
            "primary", ["exact_fallback", "full_fallback"]
        )
        assert result == ["exact_fallback", "full_fallback"]

    def test_mixed_valid_invalid_and_unknown_fallbacks(self, populated_registry):
        result = populated_registry.validate(
            "primary",
            ["full_fallback", "partial_fallback", "does_not_exist", "exact_fallback"],
        )
        assert result == ["full_fallback", "exact_fallback"]

    def test_duplicate_fallbacks_are_deduped_preserving_first_occurrence(self, populated_registry):
        result = populated_registry.validate(
            "primary", ["full_fallback", "partial_fallback", "full_fallback", "exact_fallback"]
        )
        assert result == ["full_fallback", "exact_fallback"]

    def test_explicit_required_capabilities_overrides_primary(self, populated_registry):
        result = populated_registry.validate(
            "primary", ["partial_fallback"], required_capabilities=["tools"]
        )
        assert result == ["partial_fallback"]

    def test_explicit_empty_required_capabilities_accepts_any_fallback(self, populated_registry):
        result = populated_registry.validate(
            "primary", ["no_caps_fallback"], required_capabilities=[]
        )
        assert result == ["no_caps_fallback"]

    def test_unknown_required_capability_raises(self, populated_registry):
        with pytest.raises(UnknownCapabilityError):
            populated_registry.validate(
                "primary", ["full_fallback"], required_capabilities=["not_a_real_capability"]
            )

    def test_unknown_capability_error_contains_offending_names(self, populated_registry):
        with pytest.raises(UnknownCapabilityError) as exc_info:
            populated_registry.validate(
                "primary",
                ["full_fallback"],
                required_capabilities=["tools", "made_up_cap"],
            )
        assert exc_info.value.names == {"made_up_cap"}

    def test_unknown_primary_raises_before_checking_fallbacks(self, populated_registry):
        with pytest.raises(UnknownModelError):
            populated_registry.validate("does_not_exist", ["full_fallback"])