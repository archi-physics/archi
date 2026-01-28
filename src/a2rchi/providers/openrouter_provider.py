"""OpenRouter provider implementation."""

import os
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from src.a2rchi.providers.base import (
    BaseProvider,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Popular models available via OpenRouter
DEFAULT_OPENROUTER_MODELS = [
    ModelInfo(
        id="anthropic/claude-sonnet-4",
        name="anthropic/claude-sonnet-4",
        display_name="Claude Sonnet 4 (via OpenRouter)",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=64000,
    ),
    ModelInfo(
        id="anthropic/claude-3.5-sonnet",
        name="anthropic/claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet (via OpenRouter)",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="openai/gpt-4o",
        name="openai/gpt-4o",
        display_name="GPT-4o (via OpenRouter)",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=16384,
    ),
    ModelInfo(
        id="openai/gpt-4o-mini",
        name="openai/gpt-4o-mini",
        display_name="GPT-4o Mini (via OpenRouter)",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=16384,
    ),
    ModelInfo(
        id="google/gemini-2.0-flash-001",
        name="google/gemini-2.0-flash-001",
        display_name="Gemini 2.0 Flash (via OpenRouter)",
        context_window=1048576,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="meta-llama/llama-3.3-70b-instruct",
        name="meta-llama/llama-3.3-70b-instruct",
        display_name="Llama 3.3 70B (via OpenRouter)",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=False,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="deepseek/deepseek-r1",
        name="deepseek/deepseek-r1",
        display_name="DeepSeek R1 (via OpenRouter)",
        context_window=64000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=False,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="qwen/qwen-2.5-72b-instruct",
        name="qwen/qwen-2.5-72b-instruct",
        display_name="Qwen 2.5 72B (via OpenRouter)",
        context_window=32768,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=False,
        max_output_tokens=8192,
    ),
]


class OpenRouterProvider(BaseProvider):
    """
    Provider for OpenRouter, which provides access to many models via a unified API.
    
    OpenRouter uses an OpenAI-compatible API, so we use ChatOpenAI with a custom base URL.
    """
    
    provider_type = ProviderType.OPENROUTER
    display_name = "OpenRouter"
    
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OPENROUTER,
                api_key_env="OPENROUTER_API_KEY",
                base_url=self.OPENROUTER_BASE_URL,
                models=DEFAULT_OPENROUTER_MODELS,
                default_model="anthropic/claude-3.5-sonnet",
            )
        super().__init__(config)
    
    def get_chat_model(self, model_name: str, **kwargs) -> ChatOpenAI:
        """Get an OpenRouter chat model instance."""
        model_kwargs = {
            "model": model_name,
            "base_url": self.config.base_url or self.OPENROUTER_BASE_URL,
            "streaming": True,
            **self.config.extra_kwargs,
            **kwargs,
        }
        
        if self._api_key:
            model_kwargs["api_key"] = self._api_key
        
        # Add OpenRouter-specific headers
        headers = {}
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME", "A2rchi")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        if headers:
            model_kwargs["default_headers"] = headers
            
        return ChatOpenAI(**model_kwargs)
    
    def list_models(self) -> List[ModelInfo]:
        """List available OpenRouter models."""
        if self.config.models:
            return self.config.models
        return DEFAULT_OPENROUTER_MODELS
