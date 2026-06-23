# bismarck/src/errors.py


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


class ProviderError(BaseError):
    """Raised when a provider's underlying API call fails."""
    pass


class SchemaValidationError(ProviderError):
    """Raised when a provider's output couldn't be validated against the requested schema."""
    pass


class ProviderRefusalError(ProviderError):
    """Raised when the model declined to produce a response (e.g. safety refusal)."""
    pass