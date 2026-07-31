from __future__ import annotations

from types import SimpleNamespace

import pytest

from helloloop.models.openai_compatible import (
    CredentialError,
    ModelConfig,
    ModelConfigurationError,
    ModelInfrastructureError,
    OpenAICompatibleClient,
    load_model_config,
    model_preflight,
)


def _config(**overrides) -> ModelConfig:
    values = {
        "name": "fixture",
        "provider": "openai_compatible",
        "model": "fixture-model",
        "endpoint": "https://example.invalid/v1",
        "api_key_env": "HELLOLOOP_TEST_KEY",
        "retries": 2,
        "retry_backoff_seconds": 0.0,
        "input_price_per_million_usd": 2.0,
        "output_price_per_million_usd": 8.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


class _Completions:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0
        self.last_request = None

    def create(self, **request):
        self.calls += 1
        self.last_request = request
        if self.calls <= self.failures:
            raise TimeoutError("fixture timeout")
        return SimpleNamespace(
            model="returned-model-20260719",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


def _sdk(failures: int = 0):
    completions = _Completions(failures=failures)
    models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="b"), SimpleNamespace(id="a")])
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions), models=models
    ), completions


def test_model_config_loads_and_never_contains_key_value() -> None:
    config = load_model_config("configs/models/example_model.yaml")
    assert config.api_key_env == "OPENAI_API_KEY"
    assert "api_key" not in config.public_dict()
    assert (
        "api_key" not in config.to_minisweagent_config()["model_kwargs"]
        if config.model != "UNRESOLVED"
        else True
    )


def test_model_preflight_reports_missing_runtime_without_secret(monkeypatch) -> None:
    monkeypatch.delenv("HELLOLOOP_TEST_KEY", raising=False)
    result = model_preflight(_config())
    assert not result["ready"]
    assert not result["api_key_present"]
    assert "HELLOLOOP_TEST_KEY" in result["blockers"][0]


def test_chat_logs_returned_model_usage_cost_and_bounded_retry() -> None:
    sdk, completions = _sdk(failures=1)
    client = OpenAICompatibleClient(_config(), sdk_client=sdk, sleep=lambda _: None)
    result = client.chat([{"role": "user", "content": "hello"}])
    assert completions.calls == 2
    assert result.attempts == 2
    assert result.returned_model_id == "returned-model-20260719"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.estimated_cost_usd == pytest.approx(0.0006)
    assert client.list_models() == ["a", "b"]
    assert completions.last_request["max_tokens"] == 4_096
    assert "max_completion_tokens" not in completions.last_request


def test_modern_gpt_uses_max_completion_tokens_for_both_clients() -> None:
    config = _config(model="gpt-5.4-2026-03-05", temperature=1.0, max_tokens=32)
    sdk, completions = _sdk()
    OpenAICompatibleClient(config, sdk_client=sdk).chat([{"role": "user", "content": "hello"}])
    assert completions.last_request["max_completion_tokens"] == 32
    assert "max_tokens" not in completions.last_request
    kwargs = config.to_minisweagent_config()["model_kwargs"]
    assert kwargs["max_completion_tokens"] == 32
    assert "max_tokens" not in kwargs
    assert "top_p" not in kwargs


def test_legacy_minisweagent_config_keeps_top_p() -> None:
    kwargs = _config(model="fixture-model").to_minisweagent_config()["model_kwargs"]
    assert kwargs["top_p"] == 1.0


def test_modern_models_reject_unsupported_sampling_values() -> None:
    with pytest.raises(ModelConfigurationError, match="temperature=1.0"):
        _config(model="gpt-5.4-2026-03-05", temperature=0.0).validate()


def test_endpoint_failure_is_distinct_and_bounded() -> None:
    sdk, completions = _sdk(failures=5)
    client = OpenAICompatibleClient(_config(retries=1), sdk_client=sdk, sleep=lambda _: None)
    with pytest.raises(ModelInfrastructureError, match="after 2 attempts"):
        client.chat([{"role": "user", "content": "hello"}])
    assert completions.calls == 2


def test_missing_key_and_unresolved_model_are_explicit(monkeypatch) -> None:
    monkeypatch.delenv("HELLOLOOP_TEST_KEY", raising=False)
    with pytest.raises(CredentialError):
        OpenAICompatibleClient(_config())
    with pytest.raises(ModelConfigurationError, match="unresolved"):
        OpenAICompatibleClient(_config(model="UNRESOLVED"), sdk_client=object())


def test_catalog_mode_allows_unresolved_model_with_injected_sdk() -> None:
    sdk, _ = _sdk()
    client = OpenAICompatibleClient(
        _config(model="UNRESOLVED"),
        sdk_client=sdk,
        allow_unresolved_model=True,
    )
    assert client.list_models() == ["a", "b"]
