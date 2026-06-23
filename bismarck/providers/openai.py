# bismarck/src/providers/openai.py

import json
from openai import OpenAI, OpenAIError
from pydantic import TypeAdapter, ValidationError
from typing import Any

from bismarck.errors import ProviderError, ProviderRefusalError, SchemaValidationError
from bismarck.schema import GenerationRequest, LLMResponse, ModelUsage
from bismarck.providers.base import BaseProvider

_ENVELOPE_KEY = "result"


class OpenAIProvider(BaseProvider):
    
    def __init__(self):
        self.client = OpenAI()

    @property
    def name(self) -> str:
        return "openai"
    
    def _to_openai_messages(self, request: GenerationRequest) -> list[dict]:
        msgs = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        msgs.extend(
            {"role": m.role, "content": m.content} for m in request.messages
        )
        return msgs
    
    def _build_response_format(self, schema: Any) -> dict:
        json_schema = TypeAdapter(schema).json_schema()
        needs_envelope = json_schema.get("type") != "object"
        if needs_envelope:
            json_schema = {
                "type": "object",
                "properties": {_ENVELOPE_KEY: json_schema},
                "required": [_ENVELOPE_KEY],
                "additionalProperties": False,
            }
        return {
            "type": "json_schema",
            "name": getattr(schema, "__name__", "response"),
            "schema": json_schema,
            "strict": True,
        }, needs_envelope
    
    def _normalize(self, response, schema: Any, enveloped: bool) -> LLMResponse:
        if getattr(response, "refusal", None):
            raise ProviderRefusalError(f"{self.name} refused to respond: {response.refusal}")
        
        data = None
        if schema is not None:
            raw_json = response.output_text
            try:
                if enveloped:
                    raw_json = json.dumps(json.loads(raw_json)[_ENVELOPE_KEY])
                data = TypeAdapter(schema).validate_json(raw_json)
            except (ValueError, ValidationError) as e:
                raise SchemaValidationError(
                    f"{self.name} returned output that didn't match the requested schema: {e}"
                ) from e

        return LLMResponse(
            raw=response.output_text,
            data=data,
            model=response.model,
            usage=ModelUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        )

    def _generate(self, request: GenerationRequest) -> LLMResponse:
        input_messages = self._to_openai_messages(request)
        kwargs = dict(
            model=request.model,
            input=input_messages,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )
        enveloped = False
        if request.schema is not None:
            response_format, enveloped = self._build_response_format(request.schema)
            kwargs["text"] = {"format": response_format}

        try:
            response = self.client.responses.create(**kwargs)
        except OpenAIError as e:
            raise ProviderError(f"{self.name} request failed: {e}") from e

        return self._normalize(response, request.schema, enveloped)