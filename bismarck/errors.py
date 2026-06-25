# bismarck/errors.py

class BaseError(Exception):
    """Base exception for all errors."""
    pass


class UnknownProviderError(BaseError):
    def __init__(self, name: str):
        super().__init__(f"Unknown provider: {name!r}")
        self.name = name


class UnknownModelError(BaseError):
    def __init__(self, name: str):
        super().__init__(f"Unknown model: {name!r}")
        self.name = name


class UnknownCapabilityError(BaseError):
    def __init__(self, names: set[str]):
        super().__init__(f"Unknown capabilities: {sorted(names)!r}")
        self.names = names


class AllCandidatesExcludedError(BaseError):
    """Raised when unable to attempt generation due to no available models."""
    def __init__(self, providers):
        providers = sorted(set(providers))
        super().__init__(
            f"All candidate providers are currently on cooldown: {providers!r}"
        )
        self.providers = providers


class ProviderError(BaseError):
    """Raised when a provider's underlying API call fails."""
    pass


class ModelError(BaseError):
    """Raised when a specific model can't fulfill the request, but provider is healthy."""
    pass


class SchemaValidationError(ModelError):
    """Raised when a provider's output couldn't be validated against the requested schema."""
    pass


class ProviderRefusalError(ModelError):
    """Raised when the model declined to produce a response (e.g. safety refusal)."""
    pass