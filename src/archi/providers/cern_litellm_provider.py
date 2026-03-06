"""CERN LiteLLM provider implementation.

Wraps the CERN LLM Gateway (a LiteLLM proxy) which exposes an
OpenAI-compatible API and serves multiple model families (GPT, Mistral,
Qwen, etc.).
"""

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from src.archi.providers.base import (
    BaseProvider,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CERNLiteLLMProvider(BaseProvider):
    """Provider for CERN LLM Gateway (LiteLLM proxy).

    The gateway speaks the OpenAI chat-completions protocol, so we re-use
    ``ChatOpenAI`` from *langchain-openai* under the hood.  Models are
    defined entirely via the YAML config – there are no hardcoded defaults
    because the gateway's catalogue changes independently.
    """

    provider_type = ProviderType.CERN_LITELLM
    display_name = "CERN LiteLLM"

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.CERN_LITELLM,
                api_key_env="CERN_LITELLM_API_KEY",
            )
        super().__init__(config)

    @property
    def is_configured(self) -> bool:
        """The CERN gateway requires a base_url.

        An API key is optional – some deployments authenticate via kerberos
        or network policy instead.
        """
        return bool(self.config.base_url)

    def get_chat_model(self, model_name: str, **kwargs) -> ChatOpenAI:
        """Return a ``ChatOpenAI`` instance pointing at the CERN gateway."""
        config_stream_options = self.config.extra_kwargs.get("stream_options")
        request_stream_options = kwargs.get("stream_options")

        merged_stream_options: Dict[str, Any] = {"include_usage": True}
        if isinstance(config_stream_options, dict):
            merged_stream_options.update(config_stream_options)
        if isinstance(request_stream_options, dict):
            merged_stream_options.update(request_stream_options)

        model_kwargs: Dict[str, Any] = {
            "model": model_name,
            "streaming": True,
            **self.config.extra_kwargs,
            **kwargs,
        }

        if (
            isinstance(model_kwargs.get("stream_options"), dict)
            or "stream_options" not in model_kwargs
        ):
            model_kwargs["stream_options"] = merged_stream_options

        if self._api_key:
            model_kwargs["api_key"] = self._api_key

        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url

        return ChatOpenAI(**model_kwargs)

    def list_models(self) -> List[ModelInfo]:
        """Return models declared in the configuration.

        Because the gateway catalogue is managed externally, we rely on
        whatever is specified in the YAML ``models`` list rather than
        querying the gateway at runtime.
        """
        return self.config.models if self.config.models else []
