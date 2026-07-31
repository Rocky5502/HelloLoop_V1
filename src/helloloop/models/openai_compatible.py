from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ModelConfigurationError(ValueError):
    """Raised when a model request is not fully configured."""


class ModelInfrastructureError(RuntimeError):
    """Raised for provider or transport failures, distinct from task failure."""


class CredentialError(ModelInfrastructureError):
    """Raised when runtime credentials are missing or rejected."""


def _token_limit_parameter(model: str) -> str:
    model_id = model.rsplit("/", 1)[-1].lower()
    modern_prefixes = ("gpt-5", "o1", "o3", "o4")
    return "max_completion_tokens" if model_id.startswith(modern_prefixes) else "max_tokens"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    endpoint: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 20260711
    max_tokens: int = 4_096
    timeout_seconds: float = 120.0
    retries: int = 2
    retry_backoff_seconds: float = 1.0
    input_price_per_million_usd: float | None = None
    output_price_per_million_usd: float | None = None
    context_limit: int | None = None
    tokenizer: str = "provider_reported"
    prompt_template_version: str = "helloloop.swebench.v1"

    def validate(self, *, require_runtime: bool = False) -> None:
        for field_name in ("name", "provider", "model", "endpoint", "api_key_env"):
            if not str(getattr(self, field_name)).strip():
                raise ModelConfigurationError(f"{field_name} is required")
        if self.provider != "openai_compatible":
            raise ModelConfigurationError(f"unsupported provider: {self.provider}")
        if not self.endpoint.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ModelConfigurationError("endpoint must use HTTPS unless it is local")
        if not 0.0 <= self.temperature <= 2.0:
            raise ModelConfigurationError("temperature must be in [0, 2]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ModelConfigurationError("top_p must be in [0, 1]")
        if _token_limit_parameter(self.model) == "max_completion_tokens" and (
            self.temperature != 1.0 or self.top_p != 1.0
        ):
            raise ModelConfigurationError(
                "modern completion models require temperature=1.0 and top_p=1.0"
            )
        if self.max_tokens < 1 or self.timeout_seconds <= 0 or self.retries < 0:
            raise ModelConfigurationError("token, timeout, and retry limits are invalid")
        for price in (self.input_price_per_million_usd, self.output_price_per_million_usd):
            if price is not None and price < 0:
                raise ModelConfigurationError("prices must be non-negative")
        if require_runtime and self.model.upper() == "UNRESOLVED":
            raise ModelConfigurationError("model ID is unresolved")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_minisweagent_config(self) -> dict[str, Any]:
        self.validate(require_runtime=True)
        model_name = self.model
        if "/" not in model_name:
            model_name = f"openai/{model_name}"
        kwargs: dict[str, Any] = {
            "api_base": self.endpoint,
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            _token_limit_parameter(self.model): self.max_tokens,
        }
        if _token_limit_parameter(self.model) == "max_tokens":
            kwargs["top_p"] = self.top_p
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.input_price_per_million_usd is not None:
            kwargs["input_cost_per_token"] = self.input_price_per_million_usd / 1_000_000
        if self.output_price_per_million_usd is not None:
            kwargs["output_cost_per_token"] = self.output_price_per_million_usd / 1_000_000
        return {
            "model_name": model_name,
            "model_kwargs": kwargs,
            "cost_tracking": (
                "default"
                if self.input_price_per_million_usd is not None
                and self.output_price_per_million_usd is not None
                else "ignore_errors"
            ),
        }


@dataclass(frozen=True)
class ModelCallResult:
    content: str
    returned_model_id: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    finish_reason: str | None
    attempts: int
    latency_seconds: float


def load_model_config(path: str | Path) -> ModelConfig:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("PyYAML is required to load model configuration") from error
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelConfigurationError("model configuration must be a mapping")
    config = ModelConfig(**data)
    config.validate()
    return config


def model_preflight(config: ModelConfig) -> dict[str, Any]:
    blockers = []
    try:
        config.validate(require_runtime=True)
    except ModelConfigurationError as error:
        blockers.append(str(error))
    key_present = bool(os.getenv(config.api_key_env))
    if not key_present:
        blockers.append(f"runtime variable {config.api_key_env} is unset")
    return {
        "name": config.name,
        "provider": config.provider,
        "configured_model": config.model,
        "endpoint": config.endpoint,
        "api_key_env": config.api_key_env,
        "api_key_present": key_present,
        "ready": not blockers,
        "blockers": blockers,
    }


def write_model_preflight(config: ModelConfig, path: str | Path) -> dict[str, Any]:
    result = model_preflight(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


class OpenAICompatibleClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        allow_unresolved_model: bool = False,
    ) -> None:
        config.validate(require_runtime=not allow_unresolved_model)
        api_key = os.getenv(config.api_key_env)
        if sdk_client is None and not api_key:
            raise CredentialError(f"runtime variable {config.api_key_env} is unset")
        if sdk_client is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - optional dependency guard
                raise ModelInfrastructureError("OpenAI SDK is not installed") from error
            sdk_client = OpenAI(
                api_key=api_key,
                base_url=config.endpoint,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        self.config = config
        self._client = sdk_client
        self._sleep = sleep

    def list_models(self) -> list[str]:
        try:
            response = self._client.models.list()
        except Exception as error:
            self._raise_classified(error)
        return sorted(str(item.id) for item in response.data)

    def chat(self, messages: list[dict[str, str]]) -> ModelCallResult:
        if not messages:
            raise ModelConfigurationError("at least one message is required")
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            _token_limit_parameter(self.config.model): self.config.max_tokens,
        }
        if self.config.seed is not None:
            request["seed"] = self.config.seed
        started = time.monotonic()
        response = None
        attempts = 0
        for attempts in range(1, self.config.retries + 2):
            try:
                response = self._client.chat.completions.create(**request)
                break
            except Exception as error:
                if self._is_credential_error(error):
                    raise CredentialError(
                        f"provider rejected credentials: {type(error).__name__}"
                    ) from error
                if attempts > self.config.retries:
                    raise ModelInfrastructureError(
                        f"model endpoint failed after {attempts} attempts: {type(error).__name__}"
                    ) from error
                self._sleep(self.config.retry_backoff_seconds * attempts)
        if response is None or not getattr(response, "choices", None):
            raise ModelInfrastructureError("provider returned no completion choices")
        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return ModelCallResult(
            content=str(content),
            returned_model_id=str(getattr(response, "model", self.config.model)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=self.estimate_cost(input_tokens, output_tokens),
            finish_reason=getattr(choice, "finish_reason", None),
            attempts=attempts,
            latency_seconds=round(time.monotonic() - started, 6),
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        input_price = self.config.input_price_per_million_usd
        output_price = self.config.output_price_per_million_usd
        if input_price is None or output_price is None:
            return None
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    @staticmethod
    def _is_credential_error(error: Exception) -> bool:
        name = type(error).__name__.lower()
        return "authentication" in name or "permission" in name or "unauthorized" in name

    def _raise_classified(self, error: Exception) -> None:
        if self._is_credential_error(error):
            raise CredentialError(
                f"provider rejected credentials: {type(error).__name__}"
            ) from error
        raise ModelInfrastructureError(f"model endpoint failure: {type(error).__name__}") from error


def write_model_catalog(config: ModelConfig, path: str | Path) -> dict[str, Any]:
    client = OpenAICompatibleClient(config, allow_unresolved_model=True)
    payload = {
        "schema_id": "helloloop.model_catalog.v1",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "provider": config.provider,
        "endpoint": config.endpoint,
        "models": client.list_models(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_model_smoke(config: ModelConfig, path: str | Path) -> dict[str, Any]:
    result = OpenAICompatibleClient(config).chat(
        [
            {"role": "system", "content": "Return the requested literal text only."},
            {"role": "user", "content": "Return exactly HELLOLOOP_SMOKE_OK"},
        ]
    )
    payload = {
        "schema_id": "helloloop.model_smoke.v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "configured_model": config.model,
        "returned_model_id": result.returned_model_id,
        "content_match": result.content.strip() == "HELLOLOOP_SMOKE_OK",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "finish_reason": result.finish_reason,
        "attempts": result.attempts,
        "latency_seconds": result.latency_seconds,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
