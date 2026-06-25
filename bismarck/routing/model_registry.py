# bismarck/routing/model_registry.py

import yaml
from importlib.resources import files
from typing import Optional, Sequence

from bismarck.errors import UnknownModelError, UnknownCapabilityError
from bismarck.schema import ModelSpec, ModelPricing, ModelCapability, MODEL_CAPABILITIES


class ModelRegistry:

    _models: dict[str, ModelSpec] = {}

    @classmethod
    def load(cls):
        if cls._models:
            return
        with open(files("bismarck.data") / "models.yaml") as f:
            config = yaml.safe_load(f)

        cls._models = {
            name: ModelSpec(
                name=name,
                provider=spec["provider"],
                context_window=spec["context_window"],
                pricing=ModelPricing(
                    input=spec["pricing"]["input"],
                    output=spec["pricing"]["output"],
                ),
                capabilities=frozenset(
                    cap for cap, enabled in spec["capabilities"].items() if enabled
                )
            )
            for name, spec in config["models"].items()
        }

    @classmethod
    def get(cls, model: str) -> ModelSpec:
        cls.load()
        try:
            return cls._models[model]
        except KeyError:
            raise UnknownModelError(model)
        
    @classmethod
    def validate(
            cls,
            primary: str,
            fallbacks: list[str],
            required_capabilities: Optional[Sequence[ModelCapability]] = None
        ) -> list[str]:
        base_spec = cls.get(primary)
        required = frozenset(
            base_spec.capabilities
            if required_capabilities is None
            else required_capabilities
        )

        # ensure all capabilities are real
        if invalid := set(required) - MODEL_CAPABILITIES:
            raise UnknownCapabilityError(invalid)

        valid_fallbacks = []
        # dict.fromkeys dedupes and preseves order of fallbacks
        for fallback in dict.fromkeys(fallbacks):
            try:
                candidate_spec = cls.get(fallback)
            except UnknownModelError:
                continue
    
            if required.issubset(candidate_spec.capabilities):
                # fallback satisfies reqs of primary
                valid_fallbacks.append(fallback)

        return valid_fallbacks
