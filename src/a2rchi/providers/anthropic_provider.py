"""Anthropic provider implementation."""

from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic

from src.a2rchi.providers.base import (
    BaseProvider,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default models available from Anthropic
DEFAULT_ANTHROPIC_MODELS = [
    ModelInfo(
        id="claude-sonnet-4-20250514",
        name="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=64000,
    ),
    ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="claude-3-opus-20240229",
        name="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=4096,
    ),
]


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""
    
    provider_type = ProviderType.ANTHROPIC
    display_name = "Anthropic"
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key_env="ANTHROPIC_API_KEY",
                models=DEFAULT_ANTHROPIC_MODELS,
                default_model="claude-sonnet-4-20250514",
            )
        super().__init__(config)
    
    def get_chat_model(self, model_name: str, **kwargs) -> ChatAnthropic:
        """Get an Anthropic chat model instance."""
        model_kwargs = {
            "model": model_name,
            "streaming": True,
            **self.config.extra_kwargs,
            **kwargs,
        }
        
        if self._api_key:
            model_kwargs["api_key"] = self._api_key
        
        # Anthropic requires max_tokens to be set
        if "max_tokens" not in model_kwargs:
            model_info = self.get_model_info(model_name)
            if model_info and model_info.max_output_tokens:
                model_kwargs["max_tokens"] = model_info.max_output_tokens
            else:
                model_kwargs["max_tokens"] = 8192
            
        return ChatAnthropic(**model_kwargs)
    
    def list_models(self) -> List[ModelInfo]:
        """List available Anthropic models."""
        if self.config.models:
            return self.config.models
        return DEFAULT_ANTHROPIC_MODELS
