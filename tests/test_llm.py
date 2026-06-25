# tests/test_llm.py

import threading
import time

import pytest

from bismarck.llm import LLM
from bismarck.routing import ModelRegistry, ProviderRouter
from bismarck.errors import (
    AllCandidatesExcludedError,
    ModelError,
    ProviderError,
    UnknownModelError,
    UnknownProviderError,
)
from bismarck.schema import ModelSpec, ModelPricing, LLMResponse, ModelUsage


def make_spec(name, provider="provider-a", capabilities=frozenset(), context_window=8000):
    return ModelSpec(
        name=name,
        provider=provider,
        context_window=context_window,
        pricing=ModelPricing(input=0.0, output=0.0),
        capabilities=frozenset(capabilities),
    )


def make_response(model="m", text="ok"):
    return LLMResponse(raw=text, data=None, model=model, usage=ModelUsage(1, 1))


def register_model(name, provider="provider-a", **kwargs):
    """Seed a spec directly into the real ModelRegistry — same approach
    test_model_registry.py's populated_registry fixture uses."""
    ModelRegistry._models[name] = make_spec(name, provider=provider, **kwargs)


def register_provider(name, provider):
    """Seed ProviderRouter's instance cache directly, bypassing the
    name -> class lookup in _providers (which only knows the real
    openai/anthropic/google classes — we never want those instantiated
    here). Provider names used in these tests are deliberately fake
    ("provider-a", "provider-b", ...) so there's no risk of colliding
    with a real entry even if the cache weren't pre-seeded.
    """
    ProviderRouter._instances[name] = provider


class ScriptedProvider:
    """A fake provider whose .generate() replays a queued script of
    results/exceptions, one per call, and records every call's kwargs."""

    def __init__(self, name):
        self.name = name
        self._script = []
        self.calls = []

    def queue(self, *actions):
        self._script.extend(actions)
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError(
                f"provider {self.name!r}.generate() called more times than scripted "
                f"({len(self.calls)} calls so far)"
            )
        action = self._script.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_resolves_primary_spec(self):
        register_model("primary", provider="provider-a")
        llm = LLM("primary")
        assert llm.spec.name == "primary"
        assert llm.spec.provider == "provider-a"

    def test_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            LLM("does-not-exist")

    def test_no_fallbacks_by_default(self):
        register_model("primary")
        assert LLM("primary").fallbacks == []

    def test_resolves_fallbacks_in_order(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")
        register_model("fb2", provider="provider-c")
        llm = LLM("primary", fallbacks=["fb1", "fb2"])
        assert [f.name for f in llm.fallbacks] == ["fb1", "fb2"]

    def test_unknown_fallback_names_are_dropped(self):
        # ModelRegistry.validate()'s own filtering logic is covered in
        # test_model_registry.py — this just checks LLM wires it through.
        register_model("primary")
        register_model("fb1", provider="provider-b")
        llm = LLM("primary", fallbacks=["fb1", "ghost"])
        assert [f.name for f in llm.fallbacks] == ["fb1"]

    def test_default_cooldown_is_30_seconds(self):
        register_model("primary")
        assert LLM("primary").provider_cooldown == 30.0

    def test_custom_cooldown_is_stored(self):
        register_model("primary")
        assert LLM("primary", provider_cooldown_seconds=5.0).provider_cooldown == 5.0

    def test_zero_cooldown_is_allowed(self):
        register_model("primary")
        assert LLM("primary", provider_cooldown_seconds=0.0).provider_cooldown == 0.0

    def test_negative_cooldown_raises_value_error(self):
        register_model("primary")
        with pytest.raises(ValueError, match="non-negative"):
            LLM("primary", provider_cooldown_seconds=-1)

    def test_starts_with_no_exclusions(self):
        register_model("primary")
        assert LLM("primary")._excluded_until == {}


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_with_no_fallbacks_no_exclusions(self):
        register_model("primary", provider="provider-a")
        assert repr(LLM("primary")) == "<LLM model='primary' fallbacks=[]>"

    def test_repr_lists_fallback_names_in_order(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")
        register_model("fb2", provider="provider-c")
        llm = LLM("primary", fallbacks=["fb1", "fb2"])
        assert repr(llm) == "<LLM model='primary' fallbacks=['fb1', 'fb2']>"

    def test_repr_shows_currently_excluded_provider(self):
        register_model("primary", provider="provider-a")
        llm = LLM("primary")
        llm._exclude_provider("provider-a")
        assert "excluded=['provider-a']" in repr(llm)

    def test_repr_omits_expired_exclusion(self, monkeypatch):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=10)

        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        llm._exclude_provider("provider-a")
        assert "excluded=" in repr(llm)

        clock["t"] += 11  # past the 10s cooldown
        assert "excluded=" not in repr(llm)


# ---------------------------------------------------------------------------
# cooldown bookkeeping
# ---------------------------------------------------------------------------

class TestExclusionMechanics:
    def test_not_excluded_before_any_failure(self):
        register_model("primary", provider="provider-a")
        assert LLM("primary")._is_excluded("provider-a") is False

    def test_excluded_immediately_after_exclude_provider(self):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=60)
        llm._exclude_provider("provider-a")
        assert llm._is_excluded("provider-a") is True

    def test_exclusion_expires_after_cooldown_elapses(self, monkeypatch):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=30)

        clock = {"t": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        llm._exclude_provider("provider-a")
        assert llm._is_excluded("provider-a") is True

        clock["t"] = 29.999
        assert llm._is_excluded("provider-a") is True

        clock["t"] = 30.001
        assert llm._is_excluded("provider-a") is False

    def test_unrelated_provider_is_unaffected_by_exclusion(self):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=60)
        llm._exclude_provider("provider-a")
        assert llm._is_excluded("provider-b") is False

    def test_reset_clears_exclusions(self):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=60)
        llm._exclude_provider("provider-a")
        llm.reset()
        assert llm._is_excluded("provider-a") is False
        assert llm._excluded_until == {}

    def test_zero_cooldown_means_immediately_unexcluded(self, monkeypatch):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=0)

        clock = {"t": 5.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        llm._exclude_provider("provider-a")
        assert llm._is_excluded("provider-a") is False


# ---------------------------------------------------------------------------
# generate() — happy path / argument plumbing
# ---------------------------------------------------------------------------

class TestGenerateHappyPath:
    def test_returns_provider_response_on_success(self):
        register_model("primary", provider="provider-a")
        provider = ScriptedProvider("provider-a").queue(make_response(model="primary"))
        register_provider("provider-a", provider)

        result = LLM("primary").generate("hello")
        assert result.model == "primary"
        assert len(provider.calls) == 1

    def test_builds_single_user_message_from_input(self):
        register_model("primary", provider="provider-a")
        provider = ScriptedProvider("provider-a").queue(make_response())
        register_provider("provider-a", provider)

        LLM("primary").generate("hello world")

        sent = provider.calls[0]["messages"]
        assert len(sent) == 1
        assert sent[0].role == "user"
        assert sent[0].content == "hello world"

    def test_passes_system_and_schema_through_untouched(self):
        register_model("primary", provider="provider-a")
        provider = ScriptedProvider("provider-a").queue(make_response())
        register_provider("provider-a", provider)

        sentinel_schema = object()
        LLM("primary").generate("hello", system="be terse", schema=sentinel_schema)

        call = provider.calls[0]
        assert call["system"] == "be terse"
        assert call["schema"] is sentinel_schema

    def test_uses_candidate_name_not_always_primary_name(self):
        # Regression check: the model passed to provider.generate() must be
        # the *candidate's* name, not unconditionally self.spec.name.
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        provider_a = ScriptedProvider("provider-a").queue(ModelError("nope"))
        provider_b = ScriptedProvider("provider-b").queue(make_response(model="fb1"))
        register_provider("provider-a", provider_a)
        register_provider("provider-b", provider_b)

        result = LLM("primary", fallbacks=["fb1"]).generate("hello", retry=True)

        assert result.model == "fb1"
        assert provider_b.calls[0]["model"] == "fb1"

    def test_default_model_param_is_primary_spec_name(self):
        register_model("primary", provider="provider-a")
        provider = ScriptedProvider("provider-a").queue(make_response())
        register_provider("provider-a", provider)

        LLM("primary").generate("hello")
        assert provider.calls[0]["model"] == "primary"


# ---------------------------------------------------------------------------
# generate() — retry / fallback traversal
# ---------------------------------------------------------------------------

class TestGenerateRetry:
    def test_retry_false_never_touches_fallbacks_even_on_failure(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        register_provider("provider-a", ScriptedProvider("provider-a").queue(ModelError("boom")))
        provider_b = ScriptedProvider("provider-b").queue(make_response())
        register_provider("provider-b", provider_b)

        llm = LLM("primary", fallbacks=["fb1"])
        with pytest.raises(ModelError):
            llm.generate("hello")  # retry defaults to False

        assert provider_b.calls == []  # never consulted

    def test_retry_true_falls_back_after_model_error(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        register_provider("provider-a", ScriptedProvider("provider-a").queue(ModelError("boom")))
        register_provider(
            "provider-b", ScriptedProvider("provider-b").queue(make_response(model="fb1"))
        )

        llm = LLM("primary", fallbacks=["fb1"])
        result = llm.generate("hello", retry=True)
        assert result.model == "fb1"

    def test_retry_true_falls_back_after_provider_error_and_excludes_provider(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        register_provider("provider-a", ScriptedProvider("provider-a").queue(ProviderError("down")))
        register_provider(
            "provider-b", ScriptedProvider("provider-b").queue(make_response(model="fb1"))
        )

        llm = LLM("primary", fallbacks=["fb1"])
        result = llm.generate("hello", retry=True)

        assert result.model == "fb1"
        assert llm._is_excluded("provider-a") is True
        assert llm._is_excluded("provider-b") is False

    def test_model_error_does_not_exclude_the_provider(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        register_provider("provider-a", ScriptedProvider("provider-a").queue(ModelError("boom")))
        register_provider("provider-b", ScriptedProvider("provider-b").queue(make_response()))

        llm = LLM("primary", fallbacks=["fb1"])
        llm.generate("hello", retry=True)

        assert llm._is_excluded("provider-a") is False

    def test_excluded_provider_is_skipped_on_a_later_call(self):
        # First call burns provider-a via a ProviderError and falls back
        # successfully. A second call (still retry=True) should skip
        # straight past the still-excluded primary.
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        provider_a = ScriptedProvider("provider-a").queue(ProviderError("down"))
        provider_b = ScriptedProvider("provider-b").queue(
            make_response(model="fb1"), make_response(model="fb1")
        )
        register_provider("provider-a", provider_a)
        register_provider("provider-b", provider_b)

        llm = LLM("primary", fallbacks=["fb1"], provider_cooldown_seconds=9999)
        llm.generate("hello", retry=True)

        llm.generate("hello", retry=True)
        assert len(provider_a.calls) == 1  # not called again
        assert len(provider_b.calls) == 2

    def test_two_fallback_models_sharing_an_excluded_provider_are_both_skipped(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-a")  # same provider as primary
        register_model("fb2", provider="provider-b")

        provider_a = ScriptedProvider("provider-a").queue(ProviderError("down"))
        provider_b = ScriptedProvider("provider-b").queue(make_response(model="fb2"))
        register_provider("provider-a", provider_a)
        register_provider("provider-b", provider_b)

        llm = LLM("primary", fallbacks=["fb1", "fb2"])
        result = llm.generate("hello", retry=True)

        assert result.model == "fb2"
        # only one attempt ever reached provider-a (for the primary); fb1
        # was skipped via _is_excluded before being attempted at all
        assert len(provider_a.calls) == 1

    def test_stops_at_first_success_does_not_consult_remaining_candidates(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        register_provider("provider-a", ScriptedProvider("provider-a").queue(make_response()))
        # provider-b deliberately never registered in ProviderRouter._instances
        # (and isn't a real provider name either) — if it were consulted,
        # ProviderRouter.get would raise UnknownProviderError and fail this test.
        llm = LLM("primary", fallbacks=["fb1"])
        llm.generate("hello", retry=True)  # should not raise


# ---------------------------------------------------------------------------
# generate() — what gets raised when nothing succeeds
# ---------------------------------------------------------------------------

class TestGenerateErrorRaising:
    def test_raises_last_model_error_when_all_candidates_fail(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        first_error = ModelError("first")
        second_error = ModelError("second")
        register_provider("provider-a", ScriptedProvider("provider-a").queue(first_error))
        register_provider("provider-b", ScriptedProvider("provider-b").queue(second_error))

        llm = LLM("primary", fallbacks=["fb1"])
        with pytest.raises(ModelError) as exc_info:
            llm.generate("hello", retry=True)

        assert exc_info.value is second_error  # the *last* error, not the first

    def test_raises_last_provider_error_when_all_candidates_fail(self):
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")

        first_error = ProviderError("first")
        second_error = ProviderError("second")
        register_provider("provider-a", ScriptedProvider("provider-a").queue(first_error))
        register_provider("provider-b", ScriptedProvider("provider-b").queue(second_error))

        llm = LLM("primary", fallbacks=["fb1"])
        with pytest.raises(ProviderError) as exc_info:
            llm.generate("hello", retry=True)

        assert exc_info.value is second_error
        assert llm._is_excluded("provider-a") is True
        assert llm._is_excluded("provider-b") is True

    def test_unknown_provider_error_is_not_caught_and_propagates(self):
        # UnknownProviderError is a BaseError, not a ProviderError/ModelError,
        # so the except clauses in generate() must NOT swallow it.
        register_model("primary", provider="totally-unregistered-provider")
        with pytest.raises(UnknownProviderError):
            LLM("primary").generate("hello")

    def test_all_candidates_pre_excluded_raises_all_candidates_excluded_error(self):
        register_model("primary", provider="provider-a")
        register_provider("provider-a", ScriptedProvider("provider-a"))  # never queried

        llm = LLM("primary", provider_cooldown_seconds=9999)
        llm._exclude_provider("provider-a")

        with pytest.raises(AllCandidatesExcludedError) as exc_info:
            llm.generate("hello")

        assert exc_info.value.providers == ["provider-a"]

    def test_partial_exclusion_still_attempts_remaining_candidates(self):
        # Only the primary's provider is excluded; the fallback's provider
        # is healthy, so generate() should succeed via the fallback rather
        # than raising AllCandidatesExcludedError.
        register_model("primary", provider="provider-a")
        register_model("fb1", provider="provider-b")
        register_provider(
            "provider-b", ScriptedProvider("provider-b").queue(make_response(model="fb1"))
        )

        llm = LLM("primary", fallbacks=["fb1"], provider_cooldown_seconds=9999)
        llm._exclude_provider("provider-a")

        result = llm.generate("hello", retry=True)
        assert result.model == "fb1"


# ---------------------------------------------------------------------------
# basic thread-safety sanity check
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_exclude_provider_calls_do_not_corrupt_state(self):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=60)

        def hammer():
            for _ in range(200):
                llm._exclude_provider("provider-a")
                llm._is_excluded("provider-a")

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert llm._is_excluded("provider-a") is True
        assert isinstance(llm._excluded_until["provider-a"], float)

    def test_reset_during_concurrent_exclusion_does_not_raise(self):
        register_model("primary", provider="provider-a")
        llm = LLM("primary", provider_cooldown_seconds=60)
        errors = []

        def excluder():
            try:
                for _ in range(200):
                    llm._exclude_provider("provider-a")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        def resetter():
            try:
                for _ in range(200):
                    llm.reset()
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=excluder), threading.Thread(target=resetter)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []