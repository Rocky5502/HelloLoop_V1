"""Provider-neutral model configuration and clients."""

from .openai_compatible import (
    CredentialError,
    ModelCallResult,
    ModelConfig,
    ModelInfrastructureError,
    OpenAICompatibleClient,
    load_model_config,
    model_preflight,
    write_model_catalog,
    write_model_preflight,
    write_model_smoke,
)

__all__ = [
    "CredentialError",
    "ModelCallResult",
    "ModelConfig",
    "ModelInfrastructureError",
    "OpenAICompatibleClient",
    "load_model_config",
    "model_preflight",
    "write_model_catalog",
    "write_model_preflight",
    "write_model_smoke",
]
