# Bismarck

![Otto](assets/otto.jpg)
> Otto von Bismarck unified the German states into a single empire in 1871. This package continues his legacy through the unification of major LLM providers into a single API.

A single unified interface for working with multiple LLM providers. Developers can write their code once and swap models freely.

## Motivation
SDKs, request formats, and output structures vary across LLM providers. Switching between models usually means rewriting integration code. I wanted to remove that friction with one interface that supports switching between any major model.

## Features
* **Provider Agnostic** OpenAI, Anthropic, Google, and others behind a single class.
* **Automatic Fallbacks** Register backup models so that your requests are unaffected by provider outages.
* **Structured Outputs** Consistent typed response regardless of provider.
* **Model Swapping** Change a single parameter to use a different model.
* **Usage Transparency** Clear usage metrics.

## Installation

```bash
pip install git+https://github.com/emeight/bismarck.git
```

Pin to a release instead of tracking `main`:
```bash
pip install git+https://github.com/emeight/bismarck.git@v0.1.0
```

## Example
```python
from bismarck import LLM

llm = LLM(model="claude-sonnet-4-6")

response = llm.generate("Where was Teddy Roosevelt born?")
```

## Retries & Fallbacks

Pass `fallbacks` to register backup models, in priority order, and opt
into them per call with `retry=True`:

```python
from bismarck import LLM

llm = LLM(
    model="gpt-5",
    fallbacks=["claude-sonnet-4-6", "gemini-3-pro"],
)

response = llm.generate("Where was Teddy Roosevelt born?", retry=True)
```

Bismarck distinguishes two kinds of failure:

- **Provider errors** (outages, auth failures, rate limits) take the
  *entire provider* out of rotation for a cooldown window — any other
  fallback model on that same provider is skipped too, not just the one
  that failed.
- **Model errors** (a single model's refusal, a schema mismatch) only
  rule out that specific model — Bismarck moves to the next fallback
  immediately, no cooldown applied.

The cooldown defaults to 30 seconds and is configurable:

```python
llm = LLM(model="gpt-5", fallbacks=["claude-sonnet-4-6"],
          provider_cooldown_seconds=60)
```

## Supported Providers
* OpenAI
* Anthropic
* Google

## License
Standard MIT open-source licensing.