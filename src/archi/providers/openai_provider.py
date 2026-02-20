"""OpenAI provider implementation."""

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


# Default models available from OpenAI
DEFAULT_OPENAI_MODELS = [
    ModelInfo(
        id="gpt-5",
        name="gpt-5",
        display_name="GPT-5",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=16384,
    ),
    ModelInfo(
        id="gpt-4o",
        name="gpt-4o",
        display_name="GPT-4o",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=16384,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="gpt-4o-mini",
        display_name="GPT-4o Mini",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=16384,
    ),
    ModelInfo(
        id="gpt-4-turbo",
        name="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=4096,
    ),
    ModelInfo(
        id="gpt-4",
        name="gpt-4",
        display_name="GPT-4",
        context_window=8192,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=False,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="gpt-3.5-turbo",
        name="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        context_window=16385,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=False,
        max_output_tokens=4096,
    ),
    ModelInfo(
        id="o1",
        name="o1",
        display_name="o1 (Reasoning)",
        context_window=200000,
        supports_tools=False,  # o1 doesn't support tools yet
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=100000,
    ),
    ModelInfo(
        id="o1-mini",
        name="o1-mini",
        display_name="o1 Mini (Reasoning)",
        context_window=128000,
        supports_tools=False,
        supports_streaming=True,
        supports_vision=False,
        max_output_tokens=65536,
    ),
]


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models."""
    
    provider_type = ProviderType.OPENAI
    display_name = "OpenAI"
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key_env="OPENAI_API_KEY",
                models=DEFAULT_OPENAI_MODELS,
                default_model="gpt-4o",
            )
        super().__init__(config)
    
    def get_chat_model(self, model_name: str, **kwargs) -> ChatOpenAI:
        """Get an OpenAI chat model instance."""
        config_stream_options = self.config.extra_kwargs.get("stream_options")
        request_stream_options = kwargs.get("stream_options")

        # Request usage in streamed responses so downstream token accounting/UI can rely on it.
        merged_stream_options = {"include_usage": True}
        if isinstance(config_stream_options, dict):
            merged_stream_options.update(config_stream_options)
        if isinstance(request_stream_options, dict):
            merged_stream_options.update(request_stream_options)

        model_kwargs = {
            "model": model_name,
            "streaming": True,
            **self.config.extra_kwargs,
            **kwargs,
        }

        if isinstance(model_kwargs.get("stream_options"), dict) or "stream_options" not in model_kwargs:
            model_kwargs["stream_options"] = merged_stream_options
        
        if self._api_key:
            model_kwargs["api_key"] = self._api_key
            
        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url
            
        return ChatOpenAI(**model_kwargs)
    
    def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models."""
        if self.config.models:
            return self.config.models
        return DEFAULT_OPENAI_MODELS
