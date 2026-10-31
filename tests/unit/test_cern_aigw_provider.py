from src.archi.providers import get_provider
from src.archi.providers.base import ProviderConfig, ProviderType


def test_cern_aigw_provider_uses_its_own_key_and_base_url(monkeypatch):
    monkeypatch.setenv("CERN_AIGW_API_KEY", "aigw-key")
    monkeypatch.setenv("CERN_LITELLM_API_KEY", "litellm-key")
    provider = get_provider(
        "cern_aigw",
        ProviderConfig(provider_type=ProviderType.CERN_AIGW, base_url="https://aigw.cern.ch/v1"),
        use_cache=False,
    )
    assert provider.provider_type is ProviderType.CERN_AIGW
    assert provider.is_configured
    model = provider.get_chat_model("qwen3.8-27b-fp16")
    assert model.model_name == "qwen3.8-27b-fp16"
    assert model.openai_api_key.get_secret_value() == "aigw-key"
    assert str(model.openai_api_base).startswith("https://aigw.cern.ch/v1")
    assert not getattr(model, "use_responses_api", False)
