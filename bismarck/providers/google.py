# bismarck/src/providers/google.py

from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import TypeAdapter, ValidationError

from bismarck.errors import ProviderError, ProviderRefusalError, SchemaValidationError
from bismarck.providers.base import BaseProvider
from bismarck.schema import GenerationRequest, LLMResponse, ModelUsage

_ENVELOPE_KEY = "result"

# gemini uses "model" instead of "assistant" in messages
_ROLE_MAP = {"user": "user", "assistant": "model"}


class GoogleProvider(BaseProvider):
    def __init__(self):
        self.client = genai.Client()

    @property
    def name(self) -> str:
        return "google"

    def _to_google_contents(self, request: GenerationRequest) -> list[types.Content]:
        return [
            types.Content(role=_ROLE_MAP[m.role], parts=[types.Part.from_text(text=m.content)])
            for m in request.messages
        ]

    def _build_schema_config(self, schema: Any) -> tuple[dict, bool]:
        json_schema = TypeAdapter(schema).json_schema()
        # using an envelope despite google's ability to accept arrays here
        enveloped = json_schema.get("type") != "object"
        if enveloped:
            json_schema = {
                "type": "object",
                "properties": {_ENVELOPE_KEY: json_schema},
                "required": [_ENVELOPE_KEY],
            }
        return json_schema, enveloped

    def _extract_text(self, response) -> str:
        return response.text or ""

    def _normalize(self, response, schema: Any, enveloped: bool) -> LLMResponse:
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None and str(finish_reason).upper() in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"):
            raise ProviderRefusalError(f"{self.name} refused to respond: {finish_reason}")

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

        usage = response.usage_metadata
        return LLMResponse(
            raw=raw_text,
            data=data,
            model=response.model_version if hasattr(response, "model_version") else None,
            usage=ModelUsage(
                input_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count,
            )
        )

    def _generate(self, request: GenerationRequest) -> LLMResponse:
        config_kwargs = dict(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )
        if request.system:
            config_kwargs["system_instruction"] = request.system

        enveloped = False
        if request.schema is not None:
            json_schema, enveloped = self._build_schema_config(request.schema)
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = json_schema

        try:
            response = self.client.models.generate_content(
                model=request.model,
                contents=self._to_google_contents(request),
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except APIError as e:
            raise ProviderError(f"{self.name} request failed: {e}") from e

        return self._normalize(response, request.schema, enveloped)