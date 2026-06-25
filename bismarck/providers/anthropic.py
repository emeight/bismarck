# bismarck/providers/anthropic.py

from typing import Any

from anthropic import Anthropic, APIError
from pydantic import TypeAdapter, ValidationError

from bismarck.errors import ProviderError, ProviderRefusalError, SchemaValidationError
from bismarck.providers import BaseProvider
from bismarck.schema import GenerationRequest, LLMResponse, ModelUsage

_ENVELOPE_KEY = "result"


class AnthropicProvider(BaseProvider):
    def __init__(self):
        self.client = Anthropic()

    @property
    def name(self) -> str:
        return "anthropic"

    def _to_anthropic_messages(self, request: GenerationRequest) -> list[dict]:
        # anthropic takes messages as a top-level param
        return [{"role": m.role, "content": m.content} for m in request.messages]

    def _build_output_format(self, schema: Any) -> tuple[dict, bool]:
        json_schema = TypeAdapter(schema).json_schema()
        enveloped = json_schema.get("type") != "object"
        if enveloped:
            json_schema = {
                "type": "object",
                "properties": {_ENVELOPE_KEY: json_schema},
                "required": [_ENVELOPE_KEY],
                "additionalProperties": False,
            }
        output_format = {"type": "json_schema", "schema": json_schema}
        return output_format, enveloped

    def _extract_text(self, response) -> str:
        # content is a list of blocks; concatenate any text blocks.
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    def _normalize(self, response, schema: Any, enveloped: bool) -> LLMResponse:
        if response.stop_reason == "refusal":
            raise ProviderRefusalError(f"{self.name} refused to respond")

        raw_text = self._extract_text(response)

        data = None
        if schema is not None:
            import json
            try:
                payload = raw_text
                if enveloped:
                    payload = json.dumps(json.loads(raw_text)[_ENVELOPE_KEY])
                data = TypeAdapter(schema).validate_json(payload)
            except (ValueError, ValidationError) as e:
                raise SchemaValidationError(
                    f"{self.name} returned output that didn't match the requested schema: {e}"
                ) from e

        return LLMResponse(
            raw=raw_text,
            data=data,
            model=response.model,
            usage=ModelUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        )

    def _generate(self, request: GenerationRequest) -> LLMResponse:
        if request.max_tokens is None:
            # Unlike OpenAI, Anthropic's API has no default and requires this explicitly.
            raise ValueError(f"{self.name}: max_tokens is required")

        kwargs = dict(
            model=request.model,
            messages=self._to_anthropic_messages(request),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        if request.system:
            kwargs["system"] = request.system

        enveloped = False
        if request.schema is not None:
            output_format, enveloped = self._build_output_format(request.schema)
            kwargs["output_config"] = {"format": output_format}

        try:
            response = self.client.messages.create(**kwargs)
        except APIError as e:
            raise ProviderError(f"{self.name} request failed: {e}") from e

        return self._normalize(response, request.schema, enveloped)