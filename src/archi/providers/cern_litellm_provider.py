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
from src.archi.providers.openai_provider import needs_responses_api
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

        # The CERN gateway proxies OpenAI models; the same chat-completions
        # limitation applies to gpt-5.5+/5.6 (tools rejected unless
        # reasoning_effort='none'). Route those through /v1/responses.
        if "use_responses_api" not in model_kwargs and needs_responses_api(model_name):
            logger.info(
                "Routing model '%s' through the Responses API (use_responses_api=True)",
                model_name,
            )
            model_kwargs["use_responses_api"] = True

        # /v1/responses rejects stream_options.include_usage (400). The
        # include_usage injection above is a chat-completions-only concern,
        # so drop stream_options entirely on the Responses path.
        if model_kwargs.get("use_responses_api"):
            model_kwargs["stream_options"] = None

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


class CERNAIGatewayProvider(CERNLiteLLMProvider):
    """Provider for the CERN AI Gateway (aigw.cern.ch).

    A separate OpenAI-compatible gateway from the LiteLLM one, with its own
    (team-scoped) API key and CERN-hosted open models (e.g. qwen3.8-27b-fp16).
    Same client as CERNLiteLLMProvider; only the provider type and the key
    variable differ, so the two gateways' keys are never mixed up.
    """

    provider_type = ProviderType.CERN_AIGW
    display_name = "CERN AI Gateway"

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.CERN_AIGW,
                api_key_env="CERN_AIGW_API_KEY",
            )
        super().__init__(config)
