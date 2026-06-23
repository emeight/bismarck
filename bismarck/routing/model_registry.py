# bismark/src/routing/model_registry.py

import yaml
from importlib.resources import files

from bismarck.errors import UnknownModelError
from bismarck.schema import ModelSpec, ModelPricing


class ModelRegistry:

    _models: dict[str, ModelSpec] = {}

    @classmethod
    def load(cls):
        if cls._models:
            return
        with open(files("data") / "models.yaml") as f:
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
