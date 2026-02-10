"""Unit tests for VLLMProvider."""

import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from src.archi.providers.base import ModelInfo, ProviderConfig, ProviderType
from src.archi.providers.vllm_provider import VLLMProvider, DEFAULT_VLLM_BASE_URL


class TestVLLMProviderInit(unittest.TestCase):
    """Test VLLMProvider initialization."""

    def test_default_config(self):
        provider = VLLMProvider()
        assert provider.config.base_url == DEFAULT_VLLM_BASE_URL
        assert provider.config.provider_type == ProviderType.VLLM
        assert provider._api_key == "not-needed"

    def test_custom_base_url(self):
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            base_url="http://gpu-node:9000/v1",
        )
        provider = VLLMProvider(config)
        assert provider.config.base_url == "http://gpu-node:9000/v1"

    @patch.dict("os.environ", {"VLLM_BASE_URL": "http://env-host:8000/v1"})
    def test_env_overrides_default(self):
        provider = VLLMProvider()
        assert provider.config.base_url == "http://env-host:8000/v1"

    @patch.dict("os.environ", {"VLLM_BASE_URL": "http://env-host:8000/v1"})
    def test_env_overrides_config(self):
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            base_url="http://config-host:8000/v1",
        )
        provider = VLLMProvider(config)
        assert provider.config.base_url == "http://env-host:8000/v1"

    def test_api_key_defaults_to_not_needed(self):
        config = ProviderConfig(provider_type=ProviderType.VLLM)
        provider = VLLMProvider(config)
        assert provider._api_key == "not-needed"

    def test_base_url_defaults_when_config_has_none(self):
        config = ProviderConfig(provider_type=ProviderType.VLLM, base_url=None)
        provider = VLLMProvider(config)
        assert provider.config.base_url == DEFAULT_VLLM_BASE_URL


class TestVLLMProviderGetChatModel(unittest.TestCase):
    """Test get_chat_model returns ChatOpenAI with correct params."""

    @patch("langchain_openai.ChatOpenAI", autospec=True)
    def test_returns_chat_openai_with_defaults(self, mock_chat_openai):
        provider = VLLMProvider()
        provider.get_chat_model("my-model")

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "my-model"
        assert call_kwargs["base_url"] == DEFAULT_VLLM_BASE_URL
        assert call_kwargs["api_key"] == "not-needed"
        assert call_kwargs["streaming"] is True

    @patch("langchain_openai.ChatOpenAI", autospec=True)
    def test_custom_base_url_passed_through(self, mock_chat_openai):
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            base_url="http://custom:8000/v1",
        )
        provider = VLLMProvider(config)
        provider.get_chat_model("Qwen/Qwen2.5-7B")

        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["base_url"] == "http://custom:8000/v1"


class TestVLLMProviderListModels(unittest.TestCase):
    """Test list_models with mocked /v1/models endpoint."""

    def _mock_response(self, data, status=200):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("src.archi.providers.vllm_provider.urllib.request.urlopen")
    def test_fetches_models_from_server(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({
            "data": [
                {"id": "Qwen/Qwen2.5-7B-Instruct-1M"},
                {"id": "meta-llama/Llama-3-8B"},
            ]
        })

        provider = VLLMProvider()
        models = provider.list_models()

        assert len(models) == 2
        assert models[0].id == "Qwen/Qwen2.5-7B-Instruct-1M"
        assert models[1].id == "meta-llama/Llama-3-8B"
        assert all(isinstance(m, ModelInfo) for m in models)

    @patch("src.archi.providers.vllm_provider.urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused"))
    def test_falls_back_to_config_models(self, mock_urlopen):
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            base_url=DEFAULT_VLLM_BASE_URL,
            models=[ModelInfo(id="fallback-model", name="fallback-model", display_name="Fallback")],
        )
        provider = VLLMProvider(config)
        models = provider.list_models()

        assert len(models) == 1
        assert models[0].id == "fallback-model"

    @patch("src.archi.providers.vllm_provider.urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused"))
    def test_returns_empty_when_no_config_models(self, mock_urlopen):
        provider = VLLMProvider()
        models = provider.list_models()
        assert models == []


class TestVLLMProviderValidateConnection(unittest.TestCase):
    """Test validate_connection."""

    def _mock_response(self, status=200):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("src.archi.providers.vllm_provider.urllib.request.urlopen")
    def test_returns_true_on_200(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response(200)
        provider = VLLMProvider()
        assert provider.validate_connection() is True

    @patch("src.archi.providers.vllm_provider.urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused"))
    def test_returns_false_on_failure(self, mock_urlopen):
        provider = VLLMProvider()
        assert provider.validate_connection() is False


class TestVLLMProviderRegistration(unittest.TestCase):
    """Test that vLLM is properly registered in the provider system."""

    def test_provider_type_enum_exists(self):
        assert ProviderType.VLLM == "vllm"

    def test_get_provider_returns_vllm(self):
        from src.archi.providers import (
            _PROVIDER_REGISTRY, _PROVIDER_INSTANCES,
            register_provider, get_provider,
        )
        # Manually register only VLLMProvider to avoid importing all providers
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_INSTANCES.clear()
        register_provider(ProviderType.VLLM, VLLMProvider)

        provider = get_provider("vllm")
        assert isinstance(provider, VLLMProvider)

        _PROVIDER_REGISTRY.clear()
        _PROVIDER_INSTANCES.clear()

    def test_get_provider_by_name_returns_vllm(self):
        from src.archi.providers import (
            _PROVIDER_REGISTRY, _PROVIDER_INSTANCES,
            register_provider, get_provider_by_name,
        )
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_INSTANCES.clear()
        register_provider(ProviderType.VLLM, VLLMProvider)

        provider = get_provider_by_name("vllm")
        assert isinstance(provider, VLLMProvider)

        _PROVIDER_REGISTRY.clear()
        _PROVIDER_INSTANCES.clear()


if __name__ == "__main__":
    unittest.main()
