# bismarck/src/llm.py

from bismarck.schema import LLMResponse, Message
from bismarck.routing.model_registry import ModelRegistry
from bismarck.routing.provider_router import ProviderRouter


class LLM:
    def __init__(self, model: str):
        self.spec = ModelRegistry.get(model)
        self.provider = ProviderRouter.get(self.spec.provider)

    def __repr__(self) -> str:
        return f"<LLM model={self.spec.name!r} provider={self.spec.provider!r}>"

    def generate(
            self,
            user: str,
            system: str | None = None,
            schema=None,
        ) -> LLMResponse:
        messages = (Message(role="user", content=user),)
        return self.provider.generate(
            messages=messages,
            system=system,
            model=self.spec.name,
            schema=schema,
        )